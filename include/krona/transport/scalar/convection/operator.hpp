#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/transport/scalar/convection/detail/kernels.hpp>
#include <krona/transport/scalar/detail/views.hpp>

#include <cmath>
#include <utility>

namespace krona::transport::scalar::convection {
 
/// @brief Local additive div(mass_flux*phi), with r=b-A*phi and integrated mass/time input.
/// The matrix is donor-cell upwind; reconstruction increments are explicit RHS contributions.
/// Supplied mass flux is frozen in the linearization; its dependence on unknowns is not differentiated.
/// Outflow uses the interior trace, inflow uses inflow(context, condition), including backflow.
/// No allocations, halo exchanges, fences, resets or algebra-wrapper commits during contributions.
/// All providers are pure device-callable borrowed inputs; keep buffers alive and current.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class ConvectionOperator {
    using Space = typename Exec::memory_space;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
public:
    using execution_space = Exec;
    using memory_space = Space;
    using plan_type = ConvectionPlan<Exec, Real>;

    explicit ConvectionOperator(plan_type plan) : plan_(std::move(plan)) {}

    /// @brief Adds A_upwind and the boundary/high-order RHS to an initialized scalar SystemView.
    /// Scalar CSR and strided diagonal-component BSR slices use the same face-entry map.
    template<class Target, class Boundary, class Flux, class Reconstruction = NoReconstruction, class Inflow = FixedValueInflow>
    void add_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status, Flux flux,
        Reconstruction reconstruction = {}, Inflow inflow = {}, Real time = 0) const
    {
        validate_inputs(boundary, flux, reconstruction, inflow, time);
        const auto m = plan_.mesh_view();
        const auto rows = static_cast<std::size_t>(m.sizes.n_own);
        scalar::detail::validate_output<Exec>(target.values, target.values.extent(0));
        scalar::detail::validate_output<Exec>(target.rhs, rows);
        mpi::require(target.rhs.extent(0) == rows && target.entries.diagonal.extent(0) == rows &&
            target.entries.owner_entries.extent(0) == static_cast<std::size_t>(m.sizes.n_inner_faces) &&
            target.entries.neighbour_entries.extent(0) == static_cast<std::size_t>(m.sizes.n_internal_faces),
            "Scalar convection: incompatible system slice/entry map");
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        Kokkos::parallel_for("convection/add_rows", Policy(exec, 0, m.sizes.n_own),
            detail::AddRows<Space, Real, Target, Flux, Reconstruction, decltype(flags)>{
                row_data(), target, flux, reconstruction, flags, time});
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("convection/add_boundary", Policy(exec, range.begin, range.end),
                detail::AddBoundary<Space, Real, Target, C, Flux, Reconstruction, Inflow, decltype(flags)>{
                    boundary.geometry_view(), target, condition, flux, reconstruction, inflow, flags, time});
        });
    }

    /// @brief Adds the negative convective flux sum to an owned-cell output prefix.
    /// Field covers real cells including current MPI ghosts; boundary slots are not read.
    /// Field/reconstruction inputs must not overlap output, including strided/partial aliases.
    template<class Field, class Output, class Boundary, class Flux, class Reconstruction = NoReconstruction, class Inflow = FixedValueInflow>
    void add_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        Flux flux, Reconstruction reconstruction = {}, Inflow inflow = {}, Real time = 0) const
    {
        validate_inputs(boundary, flux, reconstruction, inflow, time);
        const auto m = plan_.mesh_view();
        scalar::detail::validate_view<Exec>(field, static_cast<std::size_t>(m.sizes.n_cells));
        scalar::detail::validate_output<Exec>(output, static_cast<std::size_t>(m.sizes.n_own));
        mpi::require(m.sizes.n_own == 0 || static_cast<const void*>(field.data()) != static_cast<const void*>(output.data()),
            "Scalar convection: in-place residual is unsupported");
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        Kokkos::parallel_for("convection/residual_rows", Policy(exec, 0, m.sizes.n_own),
            detail::ResidualRows<Space, Real, Field, Output, Flux, Reconstruction, decltype(flags)>{
                row_data(), field, output, flux, reconstruction, flags, time});
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("convection/residual_boundary", Policy(exec, range.begin, range.end),
                detail::ResidualBoundary<Space, Real, Field, Output, C, Flux, Reconstruction, Inflow, decltype(flags)>{
                    boundary.geometry_view(), field, output, condition, flux, reconstruction, inflow, flags, time});
        });
    }

    /// @brief Adds only reconstructed-RHS minus upwind-RHS, including the outflow trace correction.
    /// Does not repeat prescribed inflow/source terms or touch a matrix/revision.
    /// Restore the frozen base RHS/residual before each correction update. Rebuild A if mass flux changes.
    template<class Output, class Boundary, class Flux, class Reconstruction>
    void add_correction_async(Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        Flux flux, Reconstruction reconstruction, Real time = 0) const
    {
        validate_inputs(boundary, flux, reconstruction, FixedValueInflow{}, time);
        const auto m = plan_.mesh_view();
        scalar::detail::validate_output<Exec>(output, static_cast<std::size_t>(m.sizes.n_own));
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        if constexpr (detail::reconstructed<Reconstruction>) {
            Kokkos::parallel_for("convection/correction_rows", Policy(exec, 0, m.sizes.n_own),
                detail::CorrectionRows<Space, Real, Output, Flux, Reconstruction, decltype(flags)>{
                    row_data(), output, flux, reconstruction, flags, time});
            boundary.for_each_patch([&](auto range, const auto&) {
                Kokkos::parallel_for("convection/correction_boundary", Policy(exec, range.begin, range.end),
                    detail::CorrectionBoundary<Space, Real, Output, Flux, Reconstruction, decltype(flags)>{
                        boundary.geometry_view(), output, flux, reconstruction, flags, time});
            });
        }
    }

    [[nodiscard]] const plan_type& plan() const noexcept { return plan_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return plan_.execution_space_instance(); }
private:
    plan_type plan_;
    auto row_data() const { return detail::RowData<Space>{plan_.mesh_view(), plan_.connectivity_view()}; }

    template<class Flux> void validate_flux(const Flux&) const {}
    template<class View> void validate_flux(FaceMassFlux<View> flux) const
    { scalar::detail::validate_view<Exec>(flux.values, static_cast<std::size_t>(plan_.mesh_view().sizes.n_faces)); }
    template<class Inflow> void validate_inflow(const Inflow&) const {}
    template<class View> void validate_inflow(FaceInflowValues<View> inflow) const
    { scalar::detail::validate_view<Exec>(inflow.values, static_cast<std::size_t>(plan_.mesh_view().sizes.n_faces)); }

    template<class Boundary, class Flux, class Reconstruction, class Inflow>
    void validate_inputs(const Boundary& boundary, const Flux& flux, const Reconstruction& reconstruction,
        const Inflow& inflow, Real time) const
    {
        static_assert(!std::is_arithmetic_v<Reconstruction> && !std::is_arithmetic_v<Inflow>,
            "Pass time after reconstruction and inflow policies");
        mpi::require(std::isfinite(time), "Scalar convection: time must be finite");
        const auto m = plan_.mesh_view();
        const auto b = boundary.geometry_view();
        const auto& g = m.geometry;
        mpi::require(b.n_cells == m.sizes.n_cells && b.n_inner_faces == m.sizes.n_inner_faces &&
            b.face_owner.data() == m.connectivity.face_owner.data() &&
            b.cell_x.data() == g.cell_centroid_x.data() && b.cell_y.data() == g.cell_centroid_y.data() && b.cell_z.data() == g.cell_centroid_z.data() &&
            b.face_x.data() == g.face_centroid_x.data() && b.face_y.data() == g.face_centroid_y.data() && b.face_z.data() == g.face_centroid_z.data() &&
            b.normal_x.data() == g.face_normal_x.data() && b.normal_y.data() == g.face_normal_y.data() && b.normal_z.data() == g.face_normal_z.data() &&
            b.face_area.data() == g.face_area.data(), "Scalar convection: boundary manager must use the same compute mesh");
        validate_flux(flux);
        validate_inflow(inflow);
        detail::validate_reconstruction<Exec>(reconstruction, static_cast<std::size_t>(m.sizes.n_cells));
    }
};

} // namespace krona::transport::scalar::convection
