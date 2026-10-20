#pragma once

#include <krona/linalg/csr_matrix.hpp>
#include <krona/transport/scalar/detail/assembly.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace krona::transport::scalar {

/// @brief Integrated TPFA system -div(D grad(phi)) = source, one unknown per owned cell.
/// Shares immutable plan data; owns only this equation's coefficients, RHS and status flags.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionOperator {
    using memory_space = typename Exec::memory_space;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using Flags = Kokkos::View<int*, memory_space>;
public:
    using matrix_type = linalg::CsrMatrix<Scalar, Exec>;
    using plan_type = DiffusionPlan<Exec, Real>;

    explicit DiffusionOperator(plan_type plan)
        : plan_(std::move(plan)), matrix_(plan_.pattern()), rhs_(plan_.pattern().distribution()),
          flags_(Kokkos::view_alloc(plan_.execution_space_instance(), Kokkos::WithoutInitializing, "diffusion/status"), detail::StatusCount),
          host_flags_(Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_)) {}
    DiffusionOperator(const DiffusionOperator&) = delete;
    DiffusionOperator& operator=(const DiffusionOperator&) = delete;
    ~DiffusionOperator() { if (pending_) drain(); }

    /// @brief Local async full overwrite. Providers and all captured storage must survive commit().
    /// The same physical law is used for interior coefficients and scalar boundary conditions.
    template<class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    void assemble_async(const Boundary& boundary, IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0)
    {
        mpi::require(!pending_, "Scalar diffusion: assembly pending; call commit()");
        detail::validate_boundary(plan_, boundary);
        mpi::require(std::isfinite(time), "Scalar diffusion: time must be finite");
        const auto mesh = plan_.mesh_view();
        const auto& exec = plan_.execution_space_instance();
        detail::validate_source<Exec>(source, static_cast<std::size_t>(mesh.sizes.n_own));
        detail::validate_coefficient(plan_, law.coefficient);
        mpi::guarded([&] {
            const auto values = matrix_.overwrite_values(exec);
            const auto rhs = rhs_.overwrite(exec);
            pending_ = true; ready_ = false;
            Kokkos::deep_copy(exec, values, Scalar{0});
            Kokkos::deep_copy(exec, rhs, Scalar{0});
            Kokkos::deep_copy(exec, flags_, 0);
            Kokkos::parallel_for("diffusion/assemble_rows", Policy(exec, 0, mesh.sizes.n_own),
                detail::AssembleRows<memory_space, Real, decltype(values), decltype(rhs), Source, decltype(law), Flags>{
                    detail::row_data(mesh, plan_.connectivity_view(), plan_.geometry_view()), plan_.row_plan(), values, rhs, source, law, flags_, time});
            boundary.for_each_patch([&](auto range, const auto& condition) {
                using C = std::remove_cvref_t<decltype(condition)>;
                const auto diagonal = plan_.row_plan().diagonal;
                Kokkos::parallel_for("diffusion/assemble_boundary", Policy(exec, range.begin, range.end),
                    detail::AssembleBoundary<memory_space, Real, C, decltype(law), decltype(values), decltype(rhs), decltype(diagonal), Flags>{
                        boundary.geometry_view(), condition, law, values, rhs, diagonal, flags_, time});
            });
            using EntryPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
            Kokkos::parallel_for("diffusion/validate_system", EntryPolicy(exec, 0, std::max(values.extent(0), rhs.extent(0))),
                detail::ValidateSystem<decltype(values), decltype(rhs), decltype(plan_.row_plan().diagonal), Flags>{
                    values, rhs, plan_.row_plan().diagonal, flags_});
        });
    }

    /// @brief Blocking collective validation. Connected domains require a value/Robin anchor.
    /// Pure Neumann systems, references and disconnected-component gauges are not provided here.
    void commit()
    {
        mpi::require(pending_, "Scalar diffusion: no pending assembly");
        Kokkos::deep_copy(plan_.execution_space_instance(), host_flags_, flags_);
        drain();
        std::array<int, detail::StatusCount> global{};
        mpi::check(MPI_Allreduce(host_flags_.data(), global.data(), detail::StatusCount, MPI_INT, MPI_MAX,
            plan_.communicator().native()), "MPI_Allreduce(diffusion status)");
        mpi::require(global[detail::BadSource] == 0, "Scalar diffusion: source density must be finite");
        mpi::require(global[detail::BadDiffusion] == 0, "Scalar diffusion: face coefficient must be finite and positive");
        mpi::require(global[detail::BadBoundary] == 0, "Scalar diffusion: invalid affine boundary flux or negative owner coefficient");
        mpi::require(global[detail::BadSystem] == 0, "Scalar diffusion: unrepresentable coefficients or nonpositive diagonal");
        mpi::require(plan_.pattern().distribution().global_size() == 0 || global[detail::HasAnchor] != 0,
            "Scalar diffusion: pure Neumann problem requires an explicit reference; not supported");
        ready_ = true;
    }

    template<class Component, class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] BalanceResult balance(Component field, const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0) const
    {
        require_ready(); detail::validate_boundary(plan_, boundary);
        mpi::require(std::isfinite(time), "Scalar diffusion: time must be finite");
        const auto mesh = plan_.mesh_view();
        const auto& exec = plan_.execution_space_instance();
        detail::validate_view<Exec>(field, static_cast<std::size_t>(mesh.sizes.n_own));
        detail::validate_source<Exec>(source, static_cast<std::size_t>(mesh.sizes.n_own));
        detail::validate_coefficient(plan_, law.coefficient);
        AccumulationReal source_sum = 0, boundary_sum = 0;
        Kokkos::parallel_reduce("diffusion/integrate_source", Policy(exec, 0, mesh.sizes.n_own),
            detail::IntegrateSource<memory_space, Source>{mesh, source}, source_sum);
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            AccumulationReal sum = 0;
            Kokkos::parallel_reduce("diffusion/integrate_boundary", Policy(exec, range.begin, range.end),
                detail::IntegrateBoundary<memory_space, Real, C, decltype(law), Component>{boundary.geometry_view(), condition, law, field, time}, sum);
            boundary_sum += sum;
        });
        const std::array<double, 2> local{static_cast<double>(source_sum), static_cast<double>(boundary_sum)};
        std::array<double, 2> global{};
        mpi::check(MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_SUM, plan_.communicator().native()),
            "MPI_Allreduce(diffusion balance)");
        mpi::require(std::isfinite(global[0]) && std::isfinite(global[1]), "Scalar diffusion: nonfinite balance");
        return {global[0], global[1], global[1]-global[0]};
    }

    [[nodiscard]] const matrix_type& matrix() const noexcept { return matrix_; }
    [[nodiscard]] const linalg::Vector<Scalar, Exec>& rhs() const noexcept { return rhs_; }
    [[nodiscard]] const plan_type& plan() const noexcept { return plan_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return plan_.execution_space_instance(); }
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    void require_ready() const { mpi::require(ready_ && !pending_, "Scalar diffusion: assemble and commit before use"); }

private:
    plan_type plan_;
    matrix_type matrix_;
    linalg::Vector<Scalar, Exec> rhs_;
    Flags flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_flags_;
    bool pending_ = false, ready_ = false;
    void drain() { matrix_.commit(); rhs_.commit(); pending_ = false; }
};

} // namespace krona::transport::scalar
