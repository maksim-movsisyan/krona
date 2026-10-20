#pragma once

#include <krona/linalg/solver.hpp>
#include <krona/transport/scalar/diffusion.hpp>

#include <limits>
#include <utility>

namespace krona::transport::scalar {

struct SteadyDiffusionOptions { linalg::SolverOptions linear; };

/// @brief Reusable scalar solve with publication to owned cells only.
/// Halo exchange and boundary filling belong to the application, allowing grouped scalar exchanges.
/// Field, shared mesh/auxiliary storage, communicator and runtimes must outlive this object.
template<class Storage = StorageReal, class Scalar = SolverReal, class Exec = execution::ExecutionSpace,
         class Real = ComputeReal, class PrecScalar = PreconditionerReal>
class SteadyDiffusion {
    using Space = typename Exec::memory_space;
    using Component = Kokkos::View<Storage*, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Operator = DiffusionOperator<Scalar, Exec, Real>;
public:
    template<class View>
    SteadyDiffusion(DiffusionPlan<Exec, Real> plan, View component, SteadyDiffusionOptions options = {})
        : field_(borrow(component)), operation_(std::move(plan)), solution_(operation_.matrix().distribution()),
          linear_(operation_.matrix(), std::move(options.linear))
    {
        const auto s = operation_.plan().mesh_view().sizes;
        const std::size_t cells = static_cast<std::size_t>(s.n_cells), boundary = static_cast<std::size_t>(s.n_faces-s.n_inner_faces);
        mpi::require(boundary <= std::numeric_limits<std::size_t>::max()-cells, "Scalar diffusion: extended size overflow");
        detail::validate_view<Exec>(field_, cells+boundary);
    }
    SteadyDiffusion(const SteadyDiffusion&) = delete;
    SteadyDiffusion& operator=(const SteadyDiffusion&) = delete;

    /// @brief Collective/blocking, current owned field is the initial guess.
    /// Success updates owned cells; MPI ghosts/boundary slots are untouched.
    /// Nonconvergence preserves the whole field; solution() retains the latest algebraic iterate.
    template<class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] linalg::SolveResult solve_owned(const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0)
    {
        const auto& exec = operation_.execution_space();
        operation_.assemble_async(boundary, law, source, time);
        solution_.copy_from_component(field_, exec);
        operation_.commit(); solution_.commit();
        const auto result = linear_.solve(operation_.rhs(), solution_);
        if (result.converged) {
            solution_.copy_to_component(field_, exec);
            solution_.commit(); // Required even when the application has no halo neighbours.
        }
        return result;
    }

    /// @brief Optional blocking diagnostic; use the same law/source/time as the assembly.
    template<class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] BalanceResult balance(const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0) const
    { return operation_.balance(field_, boundary, law, source, time); }

    [[nodiscard]] const Operator& operation() const noexcept { return operation_; }
    [[nodiscard]] const linalg::Vector<Scalar, Exec>& solution() const noexcept { return solution_; }
    [[nodiscard]] Component component() const noexcept { return field_; }
    [[nodiscard]] linalg::Solver<typename Operator::matrix_type, PrecScalar>& linear_solver() noexcept { return linear_; }

private:
    Component field_;
    Operator operation_;
    linalg::Vector<Scalar, Exec> solution_;
    linalg::Solver<typename Operator::matrix_type, PrecScalar> linear_;
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
