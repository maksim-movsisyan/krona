#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/transport/scalar/convection/detail/kernels.hpp>
#include <krona/transport/scalar/convection/detail/validation.hpp>

#include <type_traits>
#include <utility>

namespace krona::transport::scalar::convection {

/// @brief Adds finite-volume convection of one scalar using a supplied mass flow rate.
/// The matrix uses donor-cell upwind with the mass rate held fixed. Reconstruction adds an explicit RHS term.
/// The residual follows r = b - A*phi, so each face contributes minus the outward scalar flux.
/// Each add call increments existing storage. The caller initializes it, manages the halo exchange,
/// and checks the diagnostic batch after submitting the work.
/// @tparam Exec Kokkos execution space used by the plan and its kernels.
/// @tparam Real Floating-point type used to evaluate face fluxes.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class ConvectionOperator {
    using Space = typename Exec::memory_space;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using Flags = typename AssemblyStatus<Exec>::flags_type;
public:
    using execution_space = Exec;
    using memory_space = Space;
    using plan_type = ConvectionPlan<Exec, Real>;

    /// @brief Retains the validated plan and its borrowed mesh views.
    /// @param plan Plan for the mesh on which the operator will run.
    explicit ConvectionOperator(plan_type plan) : plan_(std::move(plan)) {}

    /// @brief Adds the matrix and RHS contributions from all local faces.
    /// @tparam Target Writable scalar CSR or diagonal component of a block system.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Flux Callable returning the signed mass flow rate through a face.
    /// @tparam Reconstruction Policy providing the donor-side face increment.
    /// @tparam Inflow Policy supplying the transported scalar where mass enters the domain.
    /// @param target Initialized matrix and RHS with the matching face-entry map.
    /// @param boundary Conditions for the physical boundary patches.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param flux Integrated mass rate, positive along the owner-outward normal.
    /// @param reconstruction Optional high-order increment. Keep its inputs alive and current.
    /// @param inflow Values sampled only on boundary faces with inward flow.
    /// @param time Evaluation time passed to the face providers.
    /// @param add_rhs Include prescribed inflow and explicit reconstruction in the RHS. Use false for a matrix-only pass.
    template<class Target, class Boundary, class Flux, class Reconstruction = NoReconstruction,
             class Inflow = FixedValueInflow>
    void add_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status, Flux flux,
                          Reconstruction reconstruction = {}, const Inflow& inflow = {}, Real time = 0,
                          bool add_rhs = true) const {
        mpi::guarded([&] {
            validate_inputs(boundary, flux, reconstruction, time, add_rhs);
            detail::validate_system<Exec>(target, plan_.mesh_view().sizes, add_rhs);
            const auto flags = status.device_flags(execution_space_instance());
            submit_inner_system(target, flags, all_inner_faces(), flux, reconstruction, time, add_rhs);
            submit_boundary_system(target, boundary, flags, flux, reconstruction, inflow, time, add_rhs);
        });
    }

    /// @brief Adds the system contributions from a selected range of inner faces.
    /// @tparam Target Writable scalar CSR or diagonal component of a block system.
    /// @tparam Flux Callable returning the signed mass rate through each selected face.
    /// @tparam Reconstruction Policy providing the donor-side face increment.
    /// @param target Initialized matrix and RHS with the full face-entry map.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param faces Half-open local face interval. Use {} for all inner faces.
    /// @param flux Integrated mass rate in the face orientation.
    /// @param reconstruction Optional high-order increment. All inputs read by this range must be ready.
    /// @param time Evaluation time passed to the face providers.
    /// @param add_rhs Include the explicit reconstruction contribution in the RHS.
    template<class Target, class Flux, class Reconstruction = NoReconstruction>
    void add_inner_system_async(Target target, AssemblyStatus<Exec>& status, FaceRange faces, Flux flux,
                                Reconstruction reconstruction = {}, Real time = 0, bool add_rhs = true) const {
        mpi::guarded([&] {
            const auto sizes = plan_.mesh_view().sizes;
            faces = detail::checked_inner_range(faces, sizes.n_inner_faces);
            detail::validate_time(time);
            detail::validate_providers(plan_, flux, reconstruction, false, add_rhs);
            detail::validate_system<Exec>(target, sizes, add_rhs);
            const auto flags = status.device_flags(execution_space_instance());
            if (faces.begin == faces.end)
                return;
            submit_inner_system(target, flags, faces, flux, reconstruction, time, add_rhs);
        });
    }

    /// @brief Adds outflow matrix terms and boundary RHS terms from all physical patches.
    /// @tparam Target Writable scalar CSR or diagonal component of a block system.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Flux Callable returning the signed mass rate through each boundary face.
    /// @tparam Reconstruction Policy providing the cell-side increment at outflow.
    /// @tparam Inflow Policy supplying the transported scalar at inflow.
    /// @param target Initialized matrix and RHS with the matching face-entry map.
    /// @param boundary Conditions for the physical boundary patches.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param flux Integrated mass rate, positive out of the domain.
    /// @param reconstruction Optional high-order increment at outflow.
    /// @param inflow Values sampled on faces with inward flow.
    /// @param time Evaluation time passed to the face providers.
    /// @param add_rhs Include prescribed inflow and outflow reconstruction in the RHS.
    template<class Target, class Boundary, class Flux, class Reconstruction = NoReconstruction,
             class Inflow = FixedValueInflow>
    void add_boundary_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status, Flux flux,
                                   Reconstruction reconstruction = {}, const Inflow& inflow = {}, Real time = 0,
                                   bool add_rhs = true) const {
        mpi::guarded([&] {
            validate_inputs(boundary, flux, reconstruction, time, add_rhs);
            detail::validate_system<Exec>(target, plan_.mesh_view().sizes, add_rhs);
            const auto flags = status.device_flags(execution_space_instance());
            submit_boundary_system(target, boundary, flags, flux, reconstruction, inflow, time, add_rhs);
        });
    }

    /// @brief Adds the negative outward scalar flux from all local faces to the owned residual.
    /// @tparam Field Rank-one scalar view covering owned and MPI ghost cells.
    /// @tparam Output Writable rank-one residual view covering the owned cells.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Flux Callable returning the signed mass flow rate through a face.
    /// @tparam Reconstruction Policy providing the donor-side face increment.
    /// @tparam Inflow Policy supplying the transported scalar at inflow.
    /// @param field Cell-centre values with current MPI ghosts.
    /// @param output Owned residual to increment. Keep it disjoint from the field and provider inputs.
    /// @param boundary Conditions for the physical boundary patches.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param flux Integrated mass rate in the face orientation.
    /// @param reconstruction Optional high-order increment relative to the upwind value.
    /// @param inflow Values sampled on boundary faces with inward flow.
    /// @param time Evaluation time passed to the face providers.
    template<class Field, class Output, class Boundary, class Flux, class Reconstruction = NoReconstruction,
             class Inflow = FixedValueInflow>
    void add_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
                            Flux flux, Reconstruction reconstruction = {}, const Inflow& inflow = {},
                            Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, flux, reconstruction, time);
            detail::validate_residual<Exec>(field, output, cell_count(), owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            submit_inner_residual(field, output, flags, all_inner_faces(), flux, reconstruction, time);
            submit_boundary_residual(field, output, boundary, flags, flux, reconstruction, inflow, time);
        });
    }

    /// @brief Adds the negative outward scalar flux from a selected range of inner faces.
    /// @tparam Field Rank-one scalar view with storage for owned and MPI ghost cells.
    /// @tparam Output Writable rank-one residual view covering the owned cells.
    /// @tparam Flux Callable returning the signed mass rate through each selected face.
    /// @tparam Reconstruction Policy providing the donor-side face increment.
    /// @param field Cell-centre values. Only the entries read in the selected range must be ready.
    /// @param output Owned residual to increment. Keep it disjoint from the field and provider inputs.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param faces Half-open local face interval. Use {} for all inner faces.
    /// @param flux Integrated mass rate in the face orientation.
    /// @param reconstruction Optional high-order increment. Its inputs must be ready for the selected faces.
    /// @param time Evaluation time passed to the face providers.
    template<class Field, class Output, class Flux, class Reconstruction = NoReconstruction>
    void add_inner_residual_async(Field field, Output output, AssemblyStatus<Exec>& status, FaceRange faces,
                                  Flux flux, Reconstruction reconstruction = {}, Real time = 0) const {
        mpi::guarded([&] {
            faces = detail::checked_inner_range(faces, plan_.mesh_view().sizes.n_inner_faces);
            detail::validate_time(time);
            detail::validate_providers(plan_, flux, reconstruction, false);
            detail::validate_residual<Exec>(field, output, cell_count(), owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            if (faces.begin == faces.end)
                return;
            submit_inner_residual(field, output, flags, faces, flux, reconstruction, time);
        });
    }

    /// @brief Adds the negative outward scalar flux from all physical boundary patches.
    /// @tparam Field Rank-one scalar view covering the owned cells.
    /// @tparam Output Writable rank-one residual view covering the owned cells.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Flux Callable returning the signed mass rate through each boundary face.
    /// @tparam Reconstruction Policy providing the cell-side increment at outflow.
    /// @tparam Inflow Policy supplying the transported scalar at inflow.
    /// @param field Cell-centre values. The base field only needs to cover owned cells.
    /// @param output Owned residual to increment. Keep it disjoint from the field and provider inputs.
    /// @param boundary Conditions for the physical boundary patches.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param flux Integrated mass rate, positive out of the domain.
    /// @param reconstruction Optional high-order increment at outflow, with its own input views.
    /// @param inflow Values sampled on faces with inward flow.
    /// @param time Evaluation time passed to the face providers.
    template<class Field, class Output, class Boundary, class Flux, class Reconstruction = NoReconstruction,
             class Inflow = FixedValueInflow>
    void add_boundary_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
                                     Flux flux, Reconstruction reconstruction = {}, const Inflow& inflow = {},
                                     Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, flux, reconstruction, time);
            detail::validate_residual<Exec>(field, output, owned_rows(), owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            submit_boundary_residual(field, output, boundary, flags, flux, reconstruction, inflow, time);
        });
    }

    /// @brief Adds the change in RHS or residual due to reconstruction on all local faces.
    /// Start each correction update from the frozen upwind RHS or residual. Reassemble the base
    /// terms when the mass rate changes.
    /// @tparam Output Writable rank-one view covering the owned cells.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Flux Callable returning the mass rate used for the base terms.
    /// @tparam Reconstruction Policy providing the donor-side face increment.
    /// @param output Owned RHS or residual to increment.
    /// @param boundary Manager providing the physical patch ranges.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param flux Integrated mass rate used in the base terms.
    /// @param reconstruction High-order increment relative to the upwind value.
    /// @param time Evaluation time passed to the face providers.
    template<class Output, class Boundary, class Flux, class Reconstruction>
    void add_correction_async(Output output, const Boundary& boundary, AssemblyStatus<Exec>& status, Flux flux,
                              Reconstruction reconstruction, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, flux, reconstruction, time);
            scalar::detail::validate_output<Exec>(output, owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            submit_inner_correction(output, flags, all_inner_faces(), flux, reconstruction, time);
            submit_boundary_correction(output, boundary, flags, flux, reconstruction, time);
        });
    }

    /// @brief Adds the explicit reconstruction correction from a selected range of inner faces.
    /// @tparam Output Writable rank-one view covering the owned cells.
    /// @tparam Flux Callable returning the mass rate used for the base terms.
    /// @tparam Reconstruction Policy providing the donor-side face increment.
    /// @param output Owned RHS or residual to increment.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param faces Half-open local face interval. Use {} for all inner faces.
    /// @param flux Integrated mass rate used in the base terms.
    /// @param reconstruction High-order increment with current inputs for the selected faces.
    /// @param time Evaluation time passed to the face providers.
    template<class Output, class Flux, class Reconstruction>
    void add_inner_correction_async(Output output, AssemblyStatus<Exec>& status, FaceRange faces, Flux flux,
                                    Reconstruction reconstruction, Real time = 0) const {
        mpi::guarded([&] {
            faces = detail::checked_inner_range(faces, plan_.mesh_view().sizes.n_inner_faces);
            detail::validate_time(time);
            detail::validate_providers(plan_, flux, reconstruction, false);
            scalar::detail::validate_output<Exec>(output, owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            if (faces.begin == faces.end)
                return;
            submit_inner_correction(output, flags, faces, flux, reconstruction, time);
        });
    }

    /// @brief Adds the reconstruction correction on boundary faces with outward flow.
    /// @tparam Output Writable rank-one view covering the owned cells.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Flux Callable returning the mass rate used for the base terms.
    /// @tparam Reconstruction Policy providing the cell-side increment at outflow.
    /// @param output Owned RHS or residual to increment.
    /// @param boundary Manager providing the physical patch ranges.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param flux Integrated mass rate used in the base terms.
    /// @param reconstruction High-order increment from the adjacent cell to the face.
    /// @param time Evaluation time passed to the face providers.
    template<class Output, class Boundary, class Flux, class Reconstruction>
    void add_boundary_correction_async(Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
                                       Flux flux, Reconstruction reconstruction, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, flux, reconstruction, time);
            scalar::detail::validate_output<Exec>(output, owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            submit_boundary_correction(output, boundary, flags, flux, reconstruction, time);
        });
    }

    /// @brief Returns the validated plan used by this operator.
    [[nodiscard]] const plan_type& plan() const noexcept { return plan_; }

    /// @brief Returns the execution instance on which the kernels are queued.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return plan_.execution_space_instance(); }

private:
    plan_type plan_;

    auto stencil_data() const {
        return detail::stencil_data(plan_.mesh_view(), plan_.connectivity_view());
    }

    std::size_t owned_rows() const {
        return static_cast<std::size_t>(plan_.mesh_view().sizes.n_own);
    }

    std::size_t cell_count() const {
        return static_cast<std::size_t>(plan_.mesh_view().sizes.n_cells);
    }

    FaceRange all_inner_faces() const {
        return {0, plan_.mesh_view().sizes.n_inner_faces};
    }

    template<class Boundary, class Flux, class Reconstruction>
    void validate_inputs(const Boundary& boundary, const Flux& flux, const Reconstruction& reconstruction,
                         Real time, bool use_reconstruction = true) const {
        detail::validate_time(time);
        detail::validate_boundary(plan_.mesh_view(), boundary);
        detail::validate_providers(plan_, flux, reconstruction, true, use_reconstruction);
    }

    // Full and split calls use the same kernels. The full calls scan the inner faces once.
    template<bool AddRhs, class Target, class Flux, class Reconstruction>
    void launch_inner_system(const Target& target, const Flags& flags, FaceRange faces, const Flux& flux,
                             const Reconstruction& reconstruction, Real time) const {
        const auto& exec = execution_space_instance();
        Kokkos::parallel_for("convection/add_rows", Policy(exec, 0, plan_.mesh_view().sizes.n_own),
            detail::AddRows<AddRhs, Space, Real, Target, Flux, Reconstruction, Flags>{
                stencil_data(), target, flux, reconstruction, flags, time, faces});
    }

    template<class Target, class Flux, class Reconstruction>
    void submit_inner_system(const Target& target, const Flags& flags, FaceRange faces, const Flux& flux,
                             const Reconstruction& reconstruction, Real time, bool add_rhs) const {
        if constexpr (detail::reconstructed<Reconstruction>) {
            if (add_rhs) {
                launch_inner_system<true>(target, flags, faces, flux, reconstruction, time);
                return;
            }
        }
        launch_inner_system<false>(target, flags, faces, flux, reconstruction, time);
    }

    template<bool AddRhs, class Target, class Boundary, class Flux, class Reconstruction, class Inflow>
    void launch_boundary_system(const Target& target, const Boundary& boundary, const Flags& flags,
                                const Flux& flux, const Reconstruction& reconstruction, const Inflow& inflow,
                                Real time) const {
        const auto& exec = execution_space_instance();
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("convection/add_boundary", Policy(exec, range.begin, range.end),
                detail::AddBoundary<AddRhs, Space, Real, Target, C, Flux, Reconstruction, Inflow, Flags>{
                    boundary.geometry_view(), target, condition, flux, reconstruction, inflow, flags, time});
        });
    }

    template<class Target, class Boundary, class Flux, class Reconstruction, class Inflow>
    void submit_boundary_system(const Target& target, const Boundary& boundary, const Flags& flags,
                                const Flux& flux, const Reconstruction& reconstruction, const Inflow& inflow,
                                Real time, bool add_rhs) const {
        if (add_rhs)
            launch_boundary_system<true>(target, boundary, flags, flux, reconstruction, inflow, time);
        else
            launch_boundary_system<false>(target, boundary, flags, flux, reconstruction, inflow, time);
    }

    template<class Field, class Output, class Flux, class Reconstruction>
    void submit_inner_residual(const Field& field, const Output& output, const Flags& flags, FaceRange faces,
                               const Flux& flux, const Reconstruction& reconstruction, Real time) const {
        const auto& exec = execution_space_instance();
        Kokkos::parallel_for("convection/residual_rows", Policy(exec, 0, plan_.mesh_view().sizes.n_own),
            detail::ResidualRows<Space, Real, Field, Output, Flux, Reconstruction, Flags>{
                stencil_data(), field, output, flux, reconstruction, flags, time, faces});
    }

    template<class Field, class Output, class Boundary, class Flux, class Reconstruction, class Inflow>
    void submit_boundary_residual(const Field& field, const Output& output, const Boundary& boundary,
                                  const Flags& flags, const Flux& flux, const Reconstruction& reconstruction,
                                  const Inflow& inflow, Real time) const {
        const auto& exec = execution_space_instance();
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("convection/residual_boundary", Policy(exec, range.begin, range.end),
                detail::ResidualBoundary<Space, Real, Field, Output, C, Flux, Reconstruction, Inflow, Flags>{
                    boundary.geometry_view(), field, output, condition, flux, reconstruction, inflow, flags, time});
        });
    }

    template<class Output, class Flux, class Reconstruction>
    void submit_inner_correction(const Output& output, const Flags& flags, FaceRange faces, const Flux& flux,
                                 const Reconstruction& reconstruction, Real time) const {
        if constexpr (detail::reconstructed<Reconstruction>) {
            const auto& exec = execution_space_instance();
            Kokkos::parallel_for("convection/correction_rows", Policy(exec, 0, plan_.mesh_view().sizes.n_own),
                detail::CorrectionRows<Space, Real, Output, Flux, Reconstruction, Flags>{
                    stencil_data(), output, flux, reconstruction, flags, time, faces});
        }
    }

    template<class Output, class Boundary, class Flux, class Reconstruction>
    void submit_boundary_correction(const Output& output, const Boundary& boundary, const Flags& flags,
                                    const Flux& flux, const Reconstruction& reconstruction, Real time) const {
        if constexpr (detail::reconstructed<Reconstruction>) {
            const auto& exec = execution_space_instance();
            boundary.for_each_patch([&](auto range, const auto&) {
                Kokkos::parallel_for("convection/correction_boundary", Policy(exec, range.begin, range.end),
                    detail::CorrectionBoundary<Space, Real, Output, Flux, Reconstruction, Flags>{
                        boundary.geometry_view(), output, flux, reconstruction, flags, time});
            });
        }
    }
};

} // namespace krona::transport::scalar::convection
