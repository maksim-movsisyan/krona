#pragma once

#include <krona/transport/viscous/plan.hpp>
#include <krona/transport/block_assembly_view.hpp>
#include <krona/transport/face_range.hpp>
#include <krona/transport/viscous/detail/kernels/boundary.hpp>
#include <krona/transport/viscous/detail/kernels/inner.hpp>
#include <krona/transport/viscous/detail/validation.hpp>

namespace krona::transport::viscous {

/// @brief Adds viscous momentum and energy fluxes or their compact block Jacobian.
/// @tparam Exec Kokkos execution space used by the plan, boundary manager and assembly status.
/// @tparam Real Scalar type used for face states, fluxes and their derivatives.
/// @tparam Properties Device-copyable policy whose evaluate<Real>(state) returns dynamic
/// viscosity and thermal conductivity at the given face state.
/// @tparam EOS Equation of state used by the inner Jacobian to recover density and internal
/// energy from p and T and to form derivatives of primitives with respect to conservative variables.
/// @tparam Accumulator Scalar type used to multiply fluxes and derivatives by face area
/// and sum contributions in each owned row.
/// @details The residual subtracts integrated outward viscous flux. The Jacobian differentiates
/// the positive outward flux sum and acts on conservative increments. Residual components
/// and block rows and columns follow mass, x/y/z momentum and total energy order.
///
/// The input packs use local cell IDs. Values are ordered p, u, v, w, T. Gradients use
/// 3 * primitive_index + axis_index with axes ordered x, y, z. Supply unlimited cell
/// gradients. The flux uses velocity and temperature derivatives.
///
/// Each add_*_async call queues additive work on the plan's execution instance.
/// The caller initializes targets and makes the fields and required MPI ghosts ready.
/// Keep all borrowed storage alive until the work completes, and keep inputs disjoint
/// from writable targets. Open a batch with AssemblyStatus::begin_async() and call
/// AssemblyStatus::check() after the last submission to wait for and check the batch.
template<class Exec, class Real, class Properties, class EOS, class Accumulator = AccumulationReal>
class ViscousOperator {
public:
    using Plan = OperatorPlan<Exec, Real>; ///< Mesh plan type borrowed by the operator.
    using Space = typename Exec::memory_space; ///< Memory space used by the stencil.

    /// @brief Borrows a mesh plan and stores copies of the material policy and equation of state.
    /// @param plan Validated mesh and cell-to-face lists with a host patch catalog.
    /// The plan must outlive the operator.
    /// @param properties Model used to evaluate viscosity and thermal conductivity on faces.
    /// @param eos Equation of state for the inner Jacobian. Use the same EOS in the boundary model.
    /// @param bulk Uniform, finite, nonnegative bulk viscosity in the isotropic stress term.
    ViscousOperator(const Plan& plan, Properties properties, EOS eos, Real bulk = 0)
        : plan_(&plan), properties_(properties), eos_(eos), bulk_(bulk) {
        mpi::require(plan.has_patch_catalog(), "Viscous: operator plan requires a host patch catalog");
        mpi::require(std::isfinite(bulk_) && bulk_ >= Real{0}, "Viscous: bulk viscosity must be finite and nonnegative");
        mpi::guarded([&] {
            if constexpr (requires { properties_.validate(); }) properties_.validate();
            if constexpr (requires { eos_.validate(); }) eos_.validate();
        });
    }

    /// @brief Rejects temporary plans because the operator borrows the plan.
    ViscousOperator(Plan&&, Properties, EOS, Real = 0) = delete;

    /// @brief Queues the viscous residual contributions from the selected inner faces.
    /// @tparam Values Readable five-component pack of cell-centred primitive variables.
    /// @tparam Gradients Readable 15-component pack of Cartesian cell gradients.
    /// @tparam Output Writable five-component residual pack indexed by owned cell ID.
    /// @param values Primitive values ready in owned cells. Also supply current MPI ghost
    /// values when the range reaches the interface faces.
    /// @param gradients Cell gradients with the same owned and ghost coverage as values.
    /// @param output Residual to increment by minus the integrated outward viscous flux.
    /// @param status Open assembly batch on the plan's execution instance.
    /// @param faces Half-open range of local inner face IDs. The default covers all inner
    /// faces, including MPI interfaces. Use end = -1 to extend the range to the end.
    template<class Values, class Gradients, class Output>
    void add_inner_residual_async(const Values& values, const Gradients& gradients, Output output,
        AssemblyStatus<Exec>& status, FaceRange faces = {}) const {
        submit_inner<false>(values, gradients, output, status, faces);
    }

    /// @brief Queues viscous residual contributions from the physical boundary faces.
    /// @tparam Values Readable five-component pack of cell-centred primitive variables.
    /// @tparam Gradients Readable 15-component pack of Cartesian cell gradients.
    /// @tparam Output Writable five-component residual pack indexed by owned cell ID.
    /// @tparam Boundary Finalized physical boundary manager with a compressible thermal model.
    /// @param values Primitive values in owned cells.
    /// @param gradients Cell gradients in owned cells.
    /// @param output Residual to increment atomically at each boundary face's owner.
    /// @param boundary Manager supplying conditions and geometry from the same mesh and patch
    /// catalog, on the same execution instance as the plan.
    /// @param status Open assembly batch on the plan's execution instance.
    /// @param time Finite physical time at which to sample boundary profiles.
    template<class Values, class Gradients, class Output, class Boundary>
    void add_boundary_residual_async(const Values& values, const Gradients& gradients, Output output,
        const Boundary& boundary, AssemblyStatus<Exec>& status, Real time = 0) const {
        submit_boundary<false>(values, gradients, output, boundary, status, WallLinearization{}, time);
    }

    /// @brief Queues both inner-face and physical boundary residual contributions.
    /// @tparam Values Readable five-component pack of cell-centred primitive variables.
    /// @tparam Gradients Readable 15-component pack of Cartesian cell gradients.
    /// @tparam Output Writable five-component residual pack indexed by owned cell ID.
    /// @tparam Boundary Finalized physical boundary manager with a compressible thermal model.
    /// @param values Primitive values in owned cells and current MPI ghosts.
    /// @param gradients Cell gradients in owned cells and current MPI ghosts.
    /// @param output Residual to increment by minus the integrated outward viscous flux.
    /// @param boundary Manager supplying boundary conditions and geometry on the plan's
    /// mesh, patch numbering and execution instance.
    /// @param status Open assembly batch shared by both submissions.
    /// @param time Finite physical time at which to sample boundary profiles.
    template<class Values, class Gradients, class Output, class Boundary>
    void add_residual_async(const Values& values, const Gradients& gradients, Output output,
        const Boundary& boundary, AssemblyStatus<Exec>& status, Real time = 0) const {
        add_inner_residual_async(values, gradients, output, status);
        add_boundary_residual_async(values, gradients, output, boundary, status, time);
    }

    /// @brief Queues compact inner-face Jacobian contributions to the owned matrix rows.
    /// The linearization keeps the face material coefficients and cell gradients fixed.
    /// @tparam Values Readable five-component pack of cell-centred primitive variables.
    /// @tparam Gradients Readable 15-component pack of Cartesian cell gradients.
    /// @tparam Target Writable 5x5 block view with its cell-to-diagonal and face-to-entry maps.
    /// @param values Primitive values in owned cells and current MPI ghosts when the range
    /// reaches the interface faces.
    /// @param gradients Cell gradients with the same owned and ghost coverage as values.
    /// @param target Matrix entries to increment. Each inner face contributes to the diagonal
    /// and neighbour blocks of its locally owned rows.
    /// @param status Open assembly batch on the plan's execution instance.
    /// @param faces Half-open range of local inner face IDs, including MPI interfaces if selected.
    /// The default covers all inner faces. Use end = -1 to extend the range to the end.
    template<class Values, class Gradients, class Target>
    void add_inner_jacobian_async(const Values& values, const Gradients& gradients, Target target,
        AssemblyStatus<Exec>& status, FaceRange faces = {}) const {
        submit_inner<true>(values, gradients, target, status, faces);
    }

    /// @brief Queues one Jacobian contribution to the owner's diagonal block per boundary face.
    /// @tparam Values Readable five-component pack of cell-centred primitive variables.
    /// @tparam Gradients Readable 15-component pack of Cartesian cell gradients.
    /// @tparam Target Writable 5x5 block view with entry maps for the plan's mesh.
    /// @tparam Boundary Finalized physical boundary manager with a compressible thermal model.
    /// @tparam Linearization Device-callable policy returning the outward flux derivative per
    /// unit area with respect to the owner's conservative variables.
    /// @param values Primitive values in owned cells.
    /// @param gradients Cell gradients in owned cells.
    /// @param target Matrix whose owner diagonal blocks are incremented atomically.
    /// @param boundary Manager supplying conditions and geometry from the same mesh and patch
    /// catalog, on the same execution instance as the plan.
    /// @param status Open assembly batch on the plan's execution instance.
    /// @param time Finite physical time at which to sample boundary profiles. Use the same
    /// time as for the corresponding residual.
    /// @param linearize Boundary derivative policy. WallLinearization uses analytic relations
    /// on walls and symmetry faces and finite differences for other conditions.
    template<class Values, class Gradients, class Target, class Boundary, class Linearization = WallLinearization>
    void add_boundary_jacobian_async(const Values& values, const Gradients& gradients, Target target,
        const Boundary& boundary, AssemblyStatus<Exec>& status, Real time = 0, Linearization linearize = {}) const {
        submit_boundary<true>(values, gradients, target, boundary, status, linearize, time);
    }

    /// @brief Queues the inner and boundary Jacobian contributions of the outward viscous flux.
    /// @tparam Values Readable five-component pack of cell-centred primitive variables.
    /// @tparam Gradients Readable 15-component pack of Cartesian cell gradients.
    /// @tparam Target Writable 5x5 block view with its cell-to-diagonal and face-to-entry maps.
    /// @tparam Boundary Finalized physical boundary manager with a compressible thermal model.
    /// @tparam Linearization Device-callable policy returning a boundary flux Jacobian per unit area.
    /// @param values Primitive values in owned cells and current MPI ghosts.
    /// @param gradients Cell gradients in owned cells and current MPI ghosts.
    /// @param target Matrix whose locally owned diagonal and neighbour blocks are incremented.
    /// @param boundary Manager supplying boundary conditions and geometry on the plan's
    /// mesh, patch numbering and execution instance.
    /// @param status Open assembly batch shared by both submissions.
    /// @param time Finite physical time at which to sample boundary profiles.
    /// @param linearize Policy used for the physical boundary contributions.
    template<class Values, class Gradients, class Target, class Boundary, class Linearization = WallLinearization>
    void add_jacobian_async(const Values& values, const Gradients& gradients, Target target,
        const Boundary& boundary, AssemblyStatus<Exec>& status, Real time = 0, Linearization linearize = {}) const {
        add_inner_jacobian_async(values, gradients, target, status);
        add_boundary_jacobian_async(values, gradients, target, boundary, status, time, linearize);
    }

private:
    const Plan* plan_;      ///< Borrowed plan supplying the mesh, patch catalog and execution instance.
    Properties properties_; ///< Stored policy for face transport coefficients.
    EOS eos_;               ///< Stored equation of state for inner-face Jacobian columns.
    Real bulk_;            ///< Uniform bulk viscosity.

    /// @brief Checks input pack coverage and the selected destination before launch.
    /// @tparam Jacobian True to validate matrix blocks and entry maps, false to validate a residual pack.
    /// @param values Cell primitive pack.
    /// @param gradients Cartesian cell gradient pack.
    /// @param target Matrix block view or residual pack to validate.
    /// @param count Minimum number of readable cells, including ghosts when the range requires them.
    template<bool Jacobian, class Values, class Gradients, class Target>
    void validate(const Values& values, const Gradients& gradients, const Target& target, LocalIndex count) const {
        const auto n = static_cast<std::size_t>(count);
        const auto sizes = plan_->mesh_view().sizes;
        transport::detail::validate_pack<Exec, 5, false>(values, n);
        transport::detail::validate_pack<Exec, 15, false>(gradients, n);
        if constexpr (Jacobian)
            detail::validate_target<Exec>(target.blocks, target.entries, sizes);
        else {
            transport::detail::validate_pack<Exec, 5, true>(target, static_cast<std::size_t>(sizes.n_own));
            transport::detail::reject_exact_alias(values, target);
            transport::detail::reject_exact_alias(gradients, target);
        }
    }

    /// @brief Validates a face range and launches one work item per owned cell.
    /// Each work item gathers the selected inner faces incident to its cell.
    /// @tparam Jacobian True for block assembly, false for residual accumulation.
    /// @param values Cell primitive pack with the owned and ghost coverage for the range.
    /// @param gradients Cell gradient pack with the same coverage as values.
    /// @param target Destination for the owned rows.
    /// @param status Open assembly batch supplying the device error flags.
    /// @param faces Half-open local inner face range to resolve and validate.
    template<bool Jacobian, class Values, class Gradients, class Target>
    void submit_inner(const Values& values, const Gradients& gradients, Target target, AssemblyStatus<Exec>& status,
                      FaceRange faces) const {
        const auto mesh = plan_->mesh_view();
        faces = detail::inner_range(faces, mesh.sizes.n_inner_faces);
        const LocalIndex count = faces.end > mesh.sizes.n_internal_faces ? mesh.sizes.n_cells : mesh.sizes.n_own;
        validate<Jacobian>(values, gradients, target, count);
        const auto& exec = plan_->execution_space_instance();
        const auto flags = status.device_flags(exec);
        if (faces.begin == faces.end) return;

        const detail::Fields inputs{values, gradients};
        using Rows = detail::InnerRows<Jacobian, Space, Real, Accumulator, Target, decltype(inputs),
                                       Properties, EOS, typename AssemblyStatus<Exec>::Flags>;
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
        mpi::guarded([&] {
            Kokkos::parallel_for(Jacobian ? "viscous/inner_jacobian" : "viscous/inner_residual",
                Policy(exec, 0, mesh.sizes.n_own),
                Rows{detail::stencil_data(mesh, plan_->connectivity_view()), target, inputs,
                     properties_, eos_, flags, faces, bulk_});
        });
    }

    /// @brief Validates the boundary inputs and queues one work item per physical boundary face.
    /// @tparam Jacobian True for diagonal block assembly, false for residual accumulation.
    /// @param values Cell primitive pack covering the owned cells.
    /// @param gradients Cell gradient pack covering the owned cells.
    /// @param target Destination for the atomic owner updates.
    /// @param boundary Manager that provides the face contexts, conditions and boundary model.
    /// @param status Open assembly batch supplying the device error flags.
    /// @param linearize Derivative policy used by the Jacobian work items.
    /// @param time Time passed to the boundary profiles.
    template<bool Jacobian, class Values, class Gradients, class Target, class Boundary, class Linearization>
    void submit_boundary(const Values& values, const Gradients& gradients, Target target, const Boundary& boundary,
        AssemblyStatus<Exec>& status, Linearization linearize, Real time) const {
        validate<Jacobian>(values, gradients, target, plan_->mesh_view().sizes.n_own);
        detail::validate_boundary(*plan_, boundary, time);
        const auto flags = status.device_flags(plan_->execution_space_instance());

        const detail::Fields inputs{values, gradients};
        const auto areas = plan_->mesh_view().geometry.face_area;
        using Faces = detail::BoundaryFaces<Jacobian, Real, Accumulator, Target, decltype(inputs), Properties,
            typename Boundary::model_type, Linearization, typename AssemblyStatus<Exec>::Flags, decltype(areas)>;
        boundary.for_each_face_async(Faces{target, inputs, properties_, boundary.model(), linearize, flags, bulk_, areas},
                                     time);
    }
};

/// @brief Deduces Exec and Real from the plan and uses the default accumulator precision.
template<class Exec, class Real, class Properties, class EOS>
ViscousOperator(const OperatorPlan<Exec, Real>&, Properties, EOS, Real = {}) ->
    ViscousOperator<Exec, Real, Properties, EOS>;

} // namespace krona::transport::viscous