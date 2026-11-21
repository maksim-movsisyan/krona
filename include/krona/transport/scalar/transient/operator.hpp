#pragma once

#include <krona/numerics/time/derivative.hpp>
#include <krona/transport/scalar/status.hpp>
#include <krona/transport/scalar/system_view.hpp>
#include <krona/transport/scalar/transient/detail/kernels.hpp>
#include <krona/transport/scalar/transient/detail/validation.hpp>
#include <krona/transport/scalar/transient/mass.hpp>

#include <cmath>
#include <type_traits>

namespace krona::transport::scalar::transient {

/// @brief Adds the integrated time derivative d(M*phi)/dt on owned cells, with M = rho*V.
/// Time t^(n+1) is the step being solved. Its mass is frozen during assembly.
/// Time t^n is the last accepted step. Each accepted field is paired with its own mass.
/// Keep all time levels in the same owned-cell order and separate from written outputs. The caller
/// keeps the inputs fixed until the queued work completes and rotates history after accepting a step.
/// @tparam Exec Kokkos execution space used by the operator's kernels.
/// @tparam Real Floating-point type used to compute the time contributions.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class TransientOperator {
    static_assert(std::is_floating_point_v<Real>);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    using Flags = typename AssemblyStatus<Exec>::flags_type;
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;

    /// @brief Creates an operator for the owned rows on this rank.
    /// @param owned Number of owned cells in the views and entry map.
    /// @param exec Execution instance shared with the diagnostic batch.
    explicit TransientOperator(std::size_t owned, Exec exec = {}) : owned_(owned), exec_(exec) {
        execution::require_kokkos();
    }

    /// @brief Queues an addition of w[0]*M^(n+1) to the diagonal and, optionally, the accepted terms to the RHS.
    /// Each diagonal ID must refer to the true diagonal of its row. Different rows must have distinct IDs.
    /// @tparam Target Writable system slice with a diagonal-entry map.
    /// @tparam Coefficient Floating-point type of the supplied derivative weights.
    /// @tparam N Number of accepted time levels used by the derivative.
    /// @tparam Mass Device-callable provider of integrated cell mass at t^(n+1).
    /// @tparam History Accepted scalar-and-mass pairs, in order t^n, t^(n-1), and so on.
    /// @param target Initialized matrix entries and, when add_rhs is true, the RHS to increment.
    /// @param status Open diagnostic batch on the operator's execution instance.
    /// @param derivative Weights including inverse time, with the unknown at index 0.
    /// @param mass_at_new_time Integrated mass for the step being solved.
    /// @param history Accepted values contributing -sum(w[j]*M_j*phi_j) for j >= 1.
    /// @param add_rhs Add the history term to the RHS. For a correction solve, use false and
    /// obtain the RHS contribution with add_residual_async() at the known iterate.
    template<class Target, class Coefficient, std::size_t N, class Mass, class History>
    void add_system_async(Target target, AssemblyStatus<Exec>& status,
                          numerics::time::Derivative<Coefficient, N> derivative,
                          Mass mass_at_new_time, History history, bool add_rhs = true) const {
        mpi::guarded([&] {
            const auto weights = validate(derivative, mass_at_new_time, history, add_rhs);
            scalar::detail::validate_output<Exec>(target.values, target.values.extent(0));
            if (add_rhs) {
                scalar::detail::validate_output<Exec>(target.rhs, owned_);
                mpi::require(target.rhs.extent(0) == owned_, "Scalar transient: incompatible system slice/entry map");
            }
            scalar::detail::validate_view<Exec>(target.entries.diagonal, owned_);
            mpi::require(target.entries.diagonal.extent(0) == owned_,
                         "Scalar transient: incompatible system slice/entry map");

            const auto flags = status.device_flags(exec_);
            if (add_rhs)
                launch_system<true>(target, weights, mass_at_new_time, history, flags);
            else
                launch_system<false>(target, weights, mass_at_new_time, history, flags);
        });
    }

    /// @brief Queues an addition of the negative integrated derivative to r = b - A*phi.
    /// Reads only owned cells. The diagonal term uses the supplied candidate at t^(n+1).
    /// @tparam Field Rank-one scalar view for the step being solved.
    /// @tparam Output Writable rank-one residual view indexed by owned cell ID.
    /// @tparam Coefficient Floating-point type of the supplied derivative weights.
    /// @tparam N Number of accepted time levels used by the derivative.
    /// @tparam Mass Device-callable provider of integrated cell mass at t^(n+1).
    /// @tparam History Accepted scalar-and-mass pairs, in order t^n, t^(n-1), and so on.
    /// @param field_at_new_time Candidate scalar at t^(n+1), before accepting the step.
    /// @param output Initialized residual to increment. Keep it disjoint from all inputs.
    /// @param status Open diagnostic batch on the operator's execution instance.
    /// @param derivative Weights including inverse time, with the candidate at index 0.
    /// @param mass_at_new_time Integrated mass paired with field_at_new_time.
    /// @param history Accepted scalar and mass values from before this step.
    template<class Field, class Output, class Coefficient, std::size_t N, class Mass, class History>
    void add_residual_async(Field field_at_new_time, Output output, AssemblyStatus<Exec>& status,
                            numerics::time::Derivative<Coefficient, N> derivative,
                            Mass mass_at_new_time, History history) const {
        mpi::guarded([&] {
            const auto weights = validate(derivative, mass_at_new_time, history);
            scalar::detail::validate_view<Exec>(field_at_new_time, owned_);
            scalar::detail::validate_output<Exec>(output, owned_);
            mpi::require(owned_ == 0 ||
                             static_cast<const void*>(field_at_new_time.data()) != static_cast<const void*>(output.data()),
                         "Scalar transient: in-place residual is unsupported");

            const auto flags = status.device_flags(exec_);
            Kokkos::parallel_for("transient/add_residual", Policy(exec_, 0, owned_),
                detail::AddResidual<Real, Field, Output, decltype(weights), Mass, History, decltype(flags)>{
                    field_at_new_time, output, weights, mass_at_new_time, history, flags});
        });
    }

    /// @brief Returns the number of owned cells processed by this operator.
    [[nodiscard]] std::size_t local_size() const noexcept { return owned_; }

    /// @brief Returns the execution instance used to queue the time term.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }

private:
    std::size_t owned_;
    Exec exec_;

    template<bool AddRhs, class Target, class Weights, class Mass, class History>
    void launch_system(const Target& target, const Weights& weights, const Mass& mass_at_new_time,
                       const History& history, const Flags& flags) const {
        Kokkos::parallel_for("transient/add_system", Policy(exec_, 0, owned_),
            detail::AddSystem<AddRhs, Real, Target, Weights, Mass, History, Flags>{
                target, weights, mass_at_new_time, history, flags});
    }

    /// @brief Converts the weights and checks the views used by the selected operation.
    template<class Coefficient, std::size_t N, class Mass, class History>
    auto validate(numerics::time::Derivative<Coefficient, N> derivative,
                  const Mass& mass_at_new_time, const History& history, bool use_history = true) const {
        static_assert(N == History::size, "Time derivative requires exactly one input layer per history coefficient");
        Kokkos::Array<Real, N + 1> weights{};
        const std::size_t count = use_history ? N + 1 : 1;
        for (std::size_t j = 0; j < count; ++j) {
            weights[j] = static_cast<Real>(derivative.weights[j]);
            mpi::require(std::isfinite(weights[j]) && (j != 0 || weights[j] > Real{0}),
                         "Scalar transient: invalid derivative coefficients in computation precision");
        }
        detail::validate_mass<Exec>(mass_at_new_time, owned_);
        if (use_history)
            detail::validate_history<Exec>(history, owned_);
        return weights;
    }
};

} // namespace krona::transport::scalar::transient
