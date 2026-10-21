#pragma once

#include <krona/transport/scalar/diffusion/detail/assembly.hpp>

#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

namespace krona::transport::scalar::diffusion {

/// @brief Borrowed TPFA spatial operator. No matrix/vector/graph, work-array allocation, or halo exchange.
/// Integrated convention: r = b - A*phi; diffusion adds -outward_flux, sources are separate.
/// Coefficients are frozen provider values; no derivatives of nonlinear models are assembled.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionOperator {
    using Space = typename Exec::memory_space;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
public:
    using execution_space = Exec;
    using memory_space = Space;
    using plan_type = DiffusionPlan<Exec, Real>;

    explicit DiffusionOperator(plan_type plan) : plan_(std::move(plan)) {}

    /// @brief Adds A_diff and b_boundary into an initialized SystemView. No zeroing, fence, MPI, or commit.
    template<class Target, class Boundary, class Coefficient = UniformCoefficient<Real>>
    void add_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law = {}, Real time = 0) const
    {
        validate_inputs(boundary, law, time);
        const auto mesh = plan_.mesh_view();
        const auto rows = static_cast<std::size_t>(mesh.sizes.n_own);
        scalar::detail::validate_output<Exec>(target.values, target.values.extent(0));
        scalar::detail::validate_output<Exec>(target.rhs, rows);
        mpi::require(target.rhs.extent(0) == rows && target.entries.diagonal.extent(0) == rows &&
            target.entries.owner_entries.extent(0) == static_cast<std::size_t>(mesh.sizes.n_inner_faces) &&
            target.entries.neighbour_entries.extent(0) == static_cast<std::size_t>(mesh.sizes.n_internal_faces),
            "Scalar diffusion: incompatible system slice/entry map");
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        Kokkos::parallel_for("diffusion/add_rows", Policy(exec, 0, mesh.sizes.n_own),
            detail::AddRows<Space, Real, Target, decltype(law), decltype(flags)>{row_data(), target, law, flags, time});
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("diffusion/add_boundary", Policy(exec, range.begin, range.end),
                detail::AddBoundary<Space, Real, Target, C, decltype(law), decltype(flags)>{
                    boundary.geometry_view(), target, condition, law, flags, time});
        });
    }

    /// @brief Adds b_boundary-A_diff*phi to owned output slots; no extended boundary samples are read.
    /// Input real-cell ghosts must be current. Input/output must not overlap, including strided aliases.
    template<class Field, class Output, class Boundary, class Coefficient = UniformCoefficient<Real>>
    void add_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law = {}, Real time = 0) const
    {
        validate_inputs(boundary, law, time);
        const auto mesh = plan_.mesh_view();
        scalar::detail::validate_view<Exec>(field, static_cast<std::size_t>(mesh.sizes.n_cells));
        scalar::detail::validate_output<Exec>(output, static_cast<std::size_t>(mesh.sizes.n_own));
        mpi::require(mesh.sizes.n_own == 0 || static_cast<const void*>(field.data()) != static_cast<const void*>(output.data()),
                     "Scalar diffusion: in-place residual is unsupported");
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        Kokkos::parallel_for("diffusion/residual_rows", Policy(exec, 0, mesh.sizes.n_own),
            detail::ResidualRows<Space, Real, Field, Output, decltype(law), decltype(flags)>{row_data(), field, output, law, flags, time});
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("diffusion/residual_boundary", Policy(exec, range.begin, range.end),
                detail::ResidualBoundary<Space, Real, Field, Output, C, decltype(law), decltype(flags)>{
                    boundary.geometry_view(), field, output, condition, law, flags, time});
        });
    }

    /// @brief Optional blocking collective diagnostic of the supplied field; source is volume density.
    template<class Field, class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] BalanceResult balance(Field field, const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0) const
    {
        validate_inputs(boundary, law, time);
        const auto mesh = plan_.mesh_view();
        const auto& exec = execution_space_instance();
        scalar::detail::validate_view<Exec>(field, static_cast<std::size_t>(mesh.sizes.n_own));
        scalar::detail::validate_source<Exec>(source, static_cast<std::size_t>(mesh.sizes.n_own));
        AccumulationReal source_sum = 0, boundary_sum = 0;
        Kokkos::parallel_reduce("diffusion/integrate_source", Policy(exec, 0, mesh.sizes.n_own),
            detail::IntegrateSource<Space, Source>{mesh, source}, source_sum);
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            AccumulationReal sum = 0;
            Kokkos::parallel_reduce("diffusion/integrate_boundary", Policy(exec, range.begin, range.end),
                detail::IntegrateBoundary<Space, Real, C, decltype(law), Field>{boundary.geometry_view(), condition, law, field, time}, sum);
            boundary_sum += sum;
        });
        const std::array<double, 2> local{static_cast<double>(source_sum), static_cast<double>(boundary_sum)};
        std::array<double, 2> global{};
        mpi::check(MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_SUM, plan_.communicator().native()),
                   "MPI_Allreduce(diffusion balance)");
        mpi::require(std::isfinite(global[0]) && std::isfinite(global[1]), "Scalar diffusion: nonfinite balance");
        return {global[0], global[1], global[1]-global[0]};
    }
    [[nodiscard]] const plan_type& plan() const noexcept { return plan_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return plan_.execution_space_instance(); }
private:
    plan_type plan_;
    auto row_data() const { return detail::row_data(plan_.mesh_view(), plan_.connectivity_view(), plan_.geometry_view()); }
    template<class Boundary, class Law>
    void validate_inputs(const Boundary& boundary, const Law& law, Real time) const
    {
        detail::validate_boundary(plan_, boundary);
        mpi::require(std::isfinite(time), "Scalar diffusion: time must be finite");
        detail::validate_coefficient(plan_, law.coefficient);
    }
};

} // namespace krona::transport::scalar::diffusion
