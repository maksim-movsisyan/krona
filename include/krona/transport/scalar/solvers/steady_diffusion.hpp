#pragma once

#include <krona/linalg/solver.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>

#include <algorithm>
#include <utility>

namespace krona::transport::scalar {

struct SteadyDiffusionOptions { linalg::SolverOptions linear; };

/// @brief Blocking convenience solver. Owns equation coefficients/RHS, shares the supplied algebra plan.
/// Publishes owned cells only; the application groups halo exchanges and boundary updates as needed.
/// The spatial DiffusionOperator itself remains allocation-free and independent of this solver.
template<class Storage = StorageReal, class Scalar = SolverReal, class Exec = execution::ExecutionSpace,
         class Real = ComputeReal, class PrecScalar = PreconditionerReal>
class SteadyDiffusion {
    using Space = typename Exec::memory_space;
    using Component = Kokkos::View<Storage*, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Operator = diffusion::DiffusionOperator<Exec, Real>;
public:
    using matrix_type = linalg::CsrMatrix<Scalar, Exec>;

    template<class View>
    SteadyDiffusion(diffusion::DiffusionPlan<Exec, Real> plan, linalg::FaceAssemblyPlan<Exec> assembly,
                    View component, SteadyDiffusionOptions options = {})
        : field_(borrow(component)), assembly_(std::move(assembly)), operation_(std::move(plan)),
          source_(operation_.plan().mesh_view(), execution_space()), status_(execution_space()),
          matrix_(assembly_.pattern()), rhs_(matrix_.distribution()), solution_(matrix_.distribution()),
          linear_(matrix_, std::move(options.linear))
    {
        const auto rows = static_cast<std::size_t>(operation_.plan().mesh_view().sizes.n_own);
        detail::validate_view<Exec>(field_, rows);
        mpi::require(matrix_.distribution().local_size() == rows, "Scalar diffusion: algebra/mesh owned count mismatch");
        int relation = MPI_UNEQUAL;
        mpi::check(MPI_Comm_compare(operation_.plan().communicator().native(), assembly_.communicator().native(), &relation),
                   "MPI_Comm_compare(diffusion assembly)");
        mpi::require(relation == MPI_IDENT || relation == MPI_CONGRUENT, "Scalar diffusion: algebra/mesh communicator mismatch");
    }
    SteadyDiffusion(const SteadyDiffusion&) = delete;
    SteadyDiffusion& operator=(const SteadyDiffusion&) = delete;

    /// @brief Collective solve using the current owned field as initial guess.
    /// Success publishes owned cells; nonconvergence preserves application storage.
    template<class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] linalg::SolveResult solve_owned(const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0)
    {
        const auto& exec = execution_space();
        // This solver owns a full rebuild; OverwriteAll avoids copying obsolete host values.
        const auto values = matrix_.overwrite_values(exec);
        const auto local_rhs = rhs_.overwrite(exec);
        const auto rhs_component = Kokkos::subview(local_rhs, Kokkos::ALL(), 0);
        const SystemView<decltype(values), decltype(rhs_component), typename linalg::FaceAssemblyPlan<Exec>::entry_map_type>
            target{values, rhs_component, assembly_.entry_map()};
        Kokkos::deep_copy(exec, target.values, Scalar{0});
        Kokkos::deep_copy(exec, target.rhs, Scalar{0});
        status_.begin_async();
        operation_.add_system_async(target, boundary, status_, law, time);
        source_.add_async(target.rhs, status_, source);
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        const auto flags = status_.device_flags(exec);
        Kokkos::parallel_for("diffusion/validate_steady_system",
            Policy(exec, 0, std::max(target.values.extent(0), target.rhs.extent(0))),
            diffusion::detail::ValidateSystem<decltype(target.values), decltype(target.rhs), decltype(target.entries.diagonal), decltype(flags)>{
                target.values, target.rhs, target.entries.diagonal, flags});
        const auto report = status_.check(operation_.plan().communicator());
        matrix_.commit(); rhs_.commit();
        mpi::require(matrix_.distribution().global_size() == 0 || report.has_diffusion_anchor,
            "Scalar diffusion: pure Neumann problem requires an explicit reference; not supported by SteadyDiffusion");
        solution_.copy_from_component(field_, exec); solution_.commit();
        const auto result = linear_.solve(rhs_, solution_);
        if (result.converged) { solution_.copy_to_component(field_, exec); solution_.commit(); }
        return result;
    }

    template<class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] diffusion::BalanceResult balance(const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0) const
    { return operation_.balance(field_, boundary, law, source, time); }

    [[nodiscard]] const Operator& operation() const noexcept { return operation_; }
    [[nodiscard]] const matrix_type& matrix() const noexcept { return matrix_; }
    [[nodiscard]] const linalg::Vector<Scalar, Exec>& rhs() const noexcept { return rhs_; }
    [[nodiscard]] const linalg::Vector<Scalar, Exec>& solution() const noexcept { return solution_; }
    [[nodiscard]] const linalg::FaceAssemblyPlan<Exec>& assembly_plan() const noexcept { return assembly_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return operation_.execution_space_instance(); }
    [[nodiscard]] Component component() const noexcept { return field_; }
    [[nodiscard]] linalg::Solver<matrix_type, PrecScalar>& linear_solver() noexcept { return linear_; }
private:
    Component field_;
    linalg::FaceAssemblyPlan<Exec> assembly_;
    Operator operation_;
    SourceOperator<Exec, Real> source_;
    AssemblyStatus<Exec> status_;
    matrix_type matrix_;
    linalg::Vector<Scalar, Exec> rhs_, solution_;
    linalg::Solver<matrix_type, PrecScalar> linear_;

    template<class View>
    static Component borrow(View view)
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(std::is_same_v<typename View::value_type, Storage>, "Supply a writable component of the selected storage scalar");
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
        return Component{view};
    }
};

} // namespace krona::transport::scalar
