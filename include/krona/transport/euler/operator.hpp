#pragma once
#include <krona/transport/euler/detail/jacobian_kernels.hpp>
#include <krona/transport/euler/detail/kernels.hpp>
#include <krona/numerics/riemann/hllc.hpp>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace krona::transport::euler {

/// @brief Adds inviscid fluxes to the residual and optionally assembles their Jacobian.
/// Residual entries are minus the outward face flux integrated over each owned cell's faces.
/// The Jacobian has the opposite sign and acts on conservative increments.
/// @tparam StateLaw Converts a primitive state (p, u, v, w, T) to a state the Riemann solver accepts.
/// @tparam Solver Computes flux per unit area from two states and a unit normal.
/// @tparam Exec Kokkos execution space used by the plan and the assembly status.
/// @tparam Real Scalar used for thermodynamics, reconstruction and face fluxes.
/// @tparam Accumulator Scalar used to multiply by face area and sum contributions.
template<class StateLaw, class Solver = numerics::riemann::Hllc<>, class Exec = execution::ExecutionSpace,
    class Real = ComputeReal, class Accumulator = AccumulationReal>
class EulerOperator {
    static_assert(std::is_floating_point_v<Real> && std::is_floating_point_v<Accumulator>);
    using Space = typename Exec::memory_space;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    using FacePolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
public:
    using execution_space = Exec;
    using memory_space = Space;
    using plan_type = EulerPlan<Exec, Real>;

    /// @brief Stores the mesh plan, state preparation policy and flux solver.
    /// @param plan Validated mesh, incident face lists and patch ranges.
    /// @param law State preparation policy used by the residual and Jacobian.
    /// @param solver Numerical flux used by the residual and default boundary Jacobian.
    EulerOperator(plan_type plan, StateLaw law, Solver solver = {})
        : plan_(std::move(plan)), law_(std::move(law)), solver_(std::move(solver)) {
        if constexpr (requires { law_.validate(); }) law_.validate();
    }

    /// @brief Adds contributions from the selected inner faces without waiting.
    /// @tparam Reconstruction Bound face profiles for pressure, velocity and temperature.
    /// @tparam Output Writable pack ordered as mass, three momentum components and energy.
    /// @param reconstruction Cell values and formulas with all inputs ready for the selected faces.
    /// @param output Residual to increment. Entries outside the selected owned rows are untouched.
    /// @param status Active assembly batch on this operator's execution instance.
    /// @param faces Half-open range of inner faces, including MPI interface faces if selected.
    /// @param cells Half-open range of owned rows.
    template<class Reconstruction, class Output>
    void add_inner_residual_async(Reconstruction reconstruction, Output output, AssemblyStatus<Exec>& status,
                                  FaceRange faces = {}, CellRange cells = {}) const {
        validate_residual(reconstruction, output, false, false);
        submit_inner_residual(reconstruction, output, status.device_flags(execution_space_instance()),
                              inner_range(faces), cell_range(cells));
    }

    /// @brief Adds exterior boundary contributions without waiting, one launch per nonempty patch.
    /// @tparam Reconstruction Bound face profiles supplying the interior trace.
    /// @tparam Output Writable residual pack in conservative component order.
    /// @tparam Boundary Physical boundary manager for the mesh held by the plan.
    /// @tparam Closure Policy that supplies the exterior trace to the Riemann solver.
    /// @param reconstruction Cell values and boundary reconstruction inputs.
    /// @param output Residual to increment.
    /// @param boundary Manager with the patch conditions and face geometry.
    /// @param status Active assembly batch on this operator's execution instance.
    /// @param closure Exterior trace policy; the default reads prepared boundary values for inlets and outlets.
    /// @param time Time at which to sample boundary profiles.
    /// @param cells Only boundary faces owned by a cell in this range contribute.
    template<class Reconstruction, class Output, class Boundary, class Closure = PreparedExterior>
    void add_boundary_residual_async(Reconstruction reconstruction, Output output, const Boundary& boundary,
                                     AssemblyStatus<Exec>& status, Closure closure = {}, Real time = 0, CellRange cells = {}) const {
        validate_boundary_request<Closure>(boundary, time);
        validate_residual(reconstruction, output, true, detail::uses_prepared_face<Closure>);
        submit_boundary_residual(reconstruction, output, boundary, status.device_flags(execution_space_instance()),
                                 closure, time, cell_range(cells));
    }

    /// @brief Adds all inner-face and boundary contributions without waiting or clearing the residual.
    /// The caller makes halo data ready before this call.
    /// @copydetails add_boundary_residual_async
    template<class Reconstruction, class Output, class Boundary, class Closure = PreparedExterior>
    void add_residual_async(Reconstruction reconstruction, Output output, const Boundary& boundary,
                            AssemblyStatus<Exec>& status, Closure closure = {}, Real time = 0, CellRange cells = {}) const {
        // Check the full request before either kernel can change the residual.
        validate_boundary_request<Closure>(boundary, time);
        validate_residual(reconstruction, output, true, detail::uses_prepared_face<Closure>);
        cells = cell_range(cells);
        const auto flags = status.device_flags(execution_space_instance());
        submit_inner_residual(reconstruction, output, flags, inner_range({}), cells);
        submit_boundary_residual(reconstruction, output, boundary, flags, closure, time, cells);
    }

    /// @brief Adds the compact inner-face Jacobian of outward flux to owned BSR rows.
    /// @tparam Target Writable BSR blocks and the face-to-entry map.
    /// @tparam Primitives Pack of pressure, velocity and temperature at cell centres.
    /// @tparam Linearization Policy that returns the owner and neighbour derivative blocks.
    /// @param target Matrix entries to increment.
    /// @param values Primitive states including all halo values used by the selected faces.
    /// @param status Active assembly batch.
    /// @param linearization Derivative policy such as FrozenRusanov.
    /// @param faces Half-open inner-face range.
    /// @param cells Half-open owned-row range.
    template<class Target, class Primitives, class Linearization>
    void add_inner_jacobian_async(Target target, Primitives values, AssemblyStatus<Exec>& status,
                                  Linearization linearization, FaceRange faces = {}, CellRange cells = {}) const {
        validate_jacobian_target(target.blocks, target.entries);
        detail::validate_pack<Exec, false>(values, static_cast<std::size_t>(plan_.mesh_view().sizes.n_cells));
        submit_inner_jacobian(target, values, status, linearization, inner_range(faces), cell_range(cells));
    }

    /// @brief Adds physical boundary blocks to owned diagonals using the selected closure.
    /// Use the same reconstruction, closure and time as for the boundary residual.
    /// @tparam Target Writable BSR blocks and their entry map.
    /// @tparam Reconstruction Bound face profiles of pressure, velocity and temperature.
    /// @tparam Boundary Physical boundary manager for the mesh held by the plan.
    /// @tparam Closure Exterior trace policy used in the residual.
    /// @tparam Linearization Policy that returns a boundary flux derivative for one face.
    /// @param target Matrix whose owner diagonal blocks are incremented.
    /// @param reconstruction Primitive fields and interior trace formulas.
    /// @param boundary Manager supplying the conditions and face geometry.
    /// @param status Active assembly batch.
    /// @param closure Policy used to obtain each exterior trace.
    /// @param time Time at which to sample boundary profiles.
    /// @param cells Only boundary faces owned by a cell in this range contribute.
    /// @param linearization Boundary derivative policy. The default uses finite differences.
    template<class Target, class Reconstruction, class Boundary, class Closure,
             class Linearization = FiniteDifferenceBoundary>
    void add_boundary_jacobian_async(Target target, Reconstruction reconstruction, const Boundary& boundary,
                                     AssemblyStatus<Exec>& status, Closure closure,
                                     Real time = 0, CellRange cells = {},
                                     Linearization linearization = {}) const {
        validate_jacobian_target(target.blocks, target.entries);
        validate_boundary_request<Closure>(boundary, time);
        validate_reconstruction(reconstruction, true, detail::jacobian_uses_prepared_face<Closure, Linearization>);
        submit_boundary_jacobian(target, reconstruction, boundary, status, closure, time, cell_range(cells), linearization);
    }

    /// @brief Adds both inner and exterior boundary blocks to the matrix without clearing it.
    /// @tparam Target Writable BSR blocks and their entry map.
    /// @tparam Reconstruction Bound face profiles of pressure, velocity and temperature.
    /// @tparam Boundary Physical boundary manager for the mesh held by the plan.
    /// @tparam InnerLinearization Policy that differentiates inner-face fluxes.
    /// @tparam Closure Exterior trace policy used in the residual.
    /// @tparam BoundaryLinearization Policy that differentiates the exterior boundary flux.
    /// @param target Matrix entries to increment.
    /// @param reconstruction Primitive fields and interior trace formulas.
    /// @param boundary Manager supplying the conditions and face geometry.
    /// @param status Active assembly batch.
    /// @param inner Inner-face derivative policy such as FrozenRusanov.
    /// @param closure Policy used to obtain each exterior trace.
    /// @param time Time at which to sample boundary profiles.
    /// @param cells Half-open owned-row range.
    /// @param outer Boundary derivative policy. The default uses finite differences.
    template<class Target, class Reconstruction, class Boundary, class InnerLinearization, class Closure,
             class BoundaryLinearization = FiniteDifferenceBoundary>
    void add_jacobian_async(Target target, Reconstruction reconstruction, const Boundary& boundary,
                            AssemblyStatus<Exec>& status, InnerLinearization inner, Closure closure, Real time = 0,
                            CellRange cells = {}, BoundaryLinearization outer = {}) const {
        validate_jacobian_target(target.blocks, target.entries);
        validate_boundary_request<Closure>(boundary, time);
        validate_reconstruction(reconstruction, true,
                                detail::jacobian_uses_prepared_face<Closure, BoundaryLinearization>);
        cells = cell_range(cells);
        submit_inner_jacobian(target, reconstruction.values(), status, inner, inner_range({}), cells);
        submit_boundary_jacobian(target, reconstruction, boundary, status, closure, time, cells, outer);
    }

    /// @brief Returns the mesh plan used by all assembly paths.
    [[nodiscard]] const plan_type& plan() const noexcept { return plan_; }
    /// @brief Returns the instance on which contributions are queued.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return plan_.execution_space_instance(); }
private:
    plan_type plan_;
    StateLaw law_;
    Solver solver_;

    FaceRange inner_range(FaceRange range) const {
        const auto count = plan_.mesh_view().sizes.n_inner_faces;
        if (range.end == -1) range.end = count;
        mpi::require(range.begin >= 0 && range.end >= range.begin && range.end <= count, "Euler: invalid inner face range");
        return range;
    }

    template<class Blocks, class Entries>
    void validate_jacobian_target(const Blocks& blocks, const Entries& entries) const {
        using Value = typename Blocks::value_type;
        static_assert(Blocks::rank == 3);
        static_assert(std::is_floating_point_v<Value> && !std::is_const_v<Value>);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Blocks::memory_space>::accessible
                      && Kokkos::SpaceAccessibility<Exec, typename Entries::memory_space>::accessible);
        const auto sizes = plan_.mesh_view().sizes;
        mpi::require(blocks.extent(1) == 5 && blocks.extent(2) == 5
                     && entries.diagonal.extent(0) == static_cast<std::size_t>(sizes.n_own)
                     && entries.owner_entries.extent(0) == static_cast<std::size_t>(sizes.n_inner_faces)
                     && entries.neighbour_entries.extent(0) == static_cast<std::size_t>(sizes.n_internal_faces),
                     "Euler: Jacobian target dimensions do not match the mesh");
    }

    template<class Closure, class Boundary>
    void validate_boundary_request(const Boundary& boundary, Real time) const {
        using Model = typename Boundary::model_type;
        static_assert(Model::compressible && Model::thermal, "Euler requires a compressible thermal boundary model");
        static_assert(!std::is_arithmetic_v<Closure>, "Pass time after an explicit boundary closure policy");
        mpi::require(std::isfinite(time), "Euler: time must be finite");
        validate_boundary(boundary);
    }

    template<class Reconstruction>
    void validate_reconstruction(const Reconstruction& reconstruction, bool with_boundary, bool needs_prepared) const {
        const auto sizes = plan_.mesh_view().sizes;
        const auto cells = static_cast<std::size_t>(sizes.n_cells);
        const auto boundary_slots = needs_prepared ? static_cast<std::size_t>(sizes.n_faces - sizes.n_inner_faces) : 0;
        reconstruction.template validate<Exec>(cells, cells + boundary_slots, with_boundary);
    }

    // Submission helpers assume that all host checks have already passed.
    template<class Reconstruction, class Output, class Flags>
    void submit_inner_residual(Reconstruction reconstruction, Output output, Flags flags,
        FaceRange faces, CellRange cells) const {
        if (faces.begin == faces.end || cells.begin == cells.end) return;
        const auto physics = detail::Physics<Real, Reconstruction, StateLaw, Solver, Flags>{
                                                            reconstruction, law_, solver_, flags};
        using Rows = detail::InnerResidualRows<Space, Real, Accumulator, Output, decltype(physics)>;
        mpi::guarded([&] {
            Kokkos::parallel_for("euler/inner_rows", Policy(execution_space_instance(), cells.begin, cells.end),
                Rows{detail::stencil_data(plan_.mesh_view(), plan_.connectivity_view()), output, physics, faces});
        });
    }

    template<class Reconstruction, class Output, class Boundary, class Flags, class Closure>
    void submit_boundary_residual(Reconstruction reconstruction, Output output, const Boundary& boundary,
                                  Flags flags, Closure closure, Real time, CellRange cells) const {
        if (cells.begin == cells.end) return;
        const auto physics = detail::Physics<Real, Reconstruction, StateLaw, Solver, Flags>{
            reconstruction, law_, solver_, flags};
        const auto& exec = execution_space_instance();
        boundary.for_each_patch([&](auto range, const auto& condition) {
            if (range.begin == range.end) return;
            using Condition = std::remove_cvref_t<decltype(condition)>;
            using Faces = detail::BoundaryResidualFaces<Space, Real, Accumulator, Output, decltype(physics),
                typename Boundary::model_type, Condition, Closure>;
            Kokkos::parallel_for("euler/boundary_residual", FacePolicy(exec, range.begin, range.end),
                Faces{boundary.geometry_view(), output, physics, boundary.model(), condition, closure, cells, time});
        });
    }

    template<class Target, class Primitives, class Linearization>
    void submit_inner_jacobian(Target target, Primitives values, AssemblyStatus<Exec>& status,
                               Linearization linearization, FaceRange faces, CellRange cells) const {
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        if (faces.begin == faces.end || cells.begin == cells.end) return;
        using Rows = detail::InnerJacobianRows<Space, Real, Accumulator, Target, Primitives, StateLaw, Linearization,
                                               decltype(flags)>;
        mpi::guarded([&] {
            Kokkos::parallel_for("euler/inner_jacobian", Policy(exec, cells.begin, cells.end),
                Rows{detail::stencil_data(plan_.mesh_view(), plan_.connectivity_view()),
                     target, values, law_, linearization, flags, faces});
        });
    }

    template<class Target, class Reconstruction, class Boundary, class Closure, class Linearization>
    void submit_boundary_jacobian(Target target, Reconstruction reconstruction, const Boundary& boundary,
                                  AssemblyStatus<Exec>& status, Closure closure, Real time, 
                                  CellRange cells, Linearization linearization) const {
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        if (cells.begin == cells.end) return;
        const auto physics = detail::Physics<Real, Reconstruction, StateLaw, Solver, decltype(flags)>{
            reconstruction, law_, solver_, flags};
        boundary.for_each_patch([&](auto range, const auto& condition) {
            if (range.begin == range.end) return;
            using Condition = std::remove_cvref_t<decltype(condition)>;
            using Faces = detail::BoundaryJacobianFaces<Space, Real, Accumulator, Target, decltype(physics),
                typename Boundary::model_type, Condition, Closure, Linearization>;
            Kokkos::parallel_for("euler/boundary_jacobian", FacePolicy(exec, range.begin, range.end),
                Faces{boundary.geometry_view(), target, physics, boundary.model(),
                      condition, closure, linearization, cells, time});
        });
    }

    CellRange cell_range(CellRange range) const {
        const auto count = static_cast<std::size_t>(plan_.mesh_view().sizes.n_own);
        if (range.end == std::numeric_limits<std::size_t>::max()) range.end = count;
        mpi::require(range.begin <= range.end && range.end <= count, "Euler: invalid owned cell range");
        return range;
    }

    // Check shape and then reject obvious aliases between the data read and written by the kernels.
    template<class Reconstruction, class Output>
    void validate_residual(const Reconstruction& reconstruction, const Output& output,
                           bool with_boundary, bool needs_prepared) const {
        const auto owned = static_cast<std::size_t>(plan_.mesh_view().sizes.n_own);
        validate_reconstruction(reconstruction, with_boundary, needs_prepared);
        detail::validate_pack<Exec, true>(output, owned);
        if (owned == 0) return;
        detail::reject_exact_alias(reconstruction.values(), output);
        reconstruction.profiles.visit_inputs([&](const char*, const auto& input, bool = false) {
            if constexpr (requires { input.x; input.y; input.z; }) {
                detail::reject_exact_alias(input.x, output);
                detail::reject_exact_alias(input.y, output);
                detail::reject_exact_alias(input.z, output);
            } else {
                detail::reject_exact_alias(input, output);
            }
        }, with_boundary);
    }
    // Euler and the boundary manager must agree on device geometry and host patch ranges.
    template<class Boundary>
    void validate_boundary(const Boundary& boundary) const {
        const auto b = boundary.geometry_view();
        const auto m = plan_.mesh_view();
        const auto& g = m.geometry;
        mpi::require(b.n_cells == m.sizes.n_cells && b.n_inner_faces == m.sizes.n_inner_faces &&
            b.face_owner.data() == m.connectivity.face_owner.data() &&
            b.cell_x.data() == g.cell_centroid_x.data() && b.cell_y.data() == g.cell_centroid_y.data() &&
            b.cell_z.data() == g.cell_centroid_z.data() &&
            b.face_x.data() == g.face_centroid_x.data() && b.face_y.data() == g.face_centroid_y.data() &&
            b.face_z.data() == g.face_centroid_z.data() &&
            b.normal_x.data() == g.face_normal_x.data() && b.normal_y.data() == g.face_normal_y.data() &&
            b.normal_z.data() == g.face_normal_z.data() && b.face_area.data() == g.face_area.data(),
            "Euler: boundary manager must use the same compute mesh");
        const auto& catalog = plan_.catalog();
        const auto patches = catalog.patches();
        boundary.for_each_patch([&](auto range, const auto&) {
            const auto id = range.patch_id;
            mpi::require(id == -1 || (id >= 0 && static_cast<std::size_t>(id) < patches.size()),
                         "Euler: unknown boundary patch");
            const auto expected = id == -1 ? catalog.unlabelled() : patches[static_cast<std::size_t>(id)].range;
            mpi::require(expected.begin == range.begin && expected.end == range.end,
                         "Euler: boundary catalog differs from plan");
        });
    }
};
template<class Exec, class Real, class Law, class Solver>
EulerOperator(EulerPlan<Exec, Real>, Law, Solver) -> EulerOperator<Law, Solver, Exec, Real>;

template<class Exec, class Real, class Law>
EulerOperator(EulerPlan<Exec, Real>, Law) -> EulerOperator<Law, numerics::riemann::Hllc<>, Exec, Real>;

} // namespace krona::transport::euler
