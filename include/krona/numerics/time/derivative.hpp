#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace krona::numerics::time {

/// @brief Weights for approximating dq/dt at the new time from q^(n+1), q^n and older values.
/// To evaluate the derivative, multiply each time level by its weight and sum the products.
/// The weights include the inverse time step and have units of inverse time.
/// Custom weights should be finite, sum to zero to preserve constants and have a positive current weight.
/// @tparam Real Floating-point type used to store the weights.
/// @tparam History Number of old time levels required by the formula, at least one.
template<class Real, std::size_t History>
struct Derivative {
    static_assert(std::is_floating_point_v<Real> && History > 0);
    static constexpr std::size_t history_size = History; ///< Number of old values required.
    Kokkos::Array<Real, History+1> weights; ///< Weight for the current unknown, then for old values newest first.
};
 
namespace detail {
/// @brief Requires a finite, positive interval between neighbouring time levels.
template<class Real>
void require_step(Real dt) {
    static_assert(std::is_floating_point_v<Real>);
    mpi::require(std::isfinite(dt) && dt > Real{0},
        "Time derivative: time step must be finite and positive");
}

/// @brief Checks that the computed weights are finite and the current weight is positive.
template<class Real, std::size_t N>
Derivative<Real, N> checked(Derivative<Real, N> derivative) {
    for (const auto weight : derivative.weights) {
        mpi::require(std::isfinite(weight),
            "Time derivative: unrepresentable coefficient");
    }
    mpi::require(derivative.weights[0] > Real{0},
        "Time derivative: nonpositive current coefficient");
    return derivative;
}
} // namespace detail

/// @brief First-order backward derivative using the current unknown and one old value.
struct BackwardEuler {
    /// @brief Computes the weights (1/dt, -1/dt) on the host.
    /// @tparam Real Floating-point type of the time step and weights.
    /// @param dt Positive interval from the latest saved value to the new time.
    /// @return Weights that can be captured by value in a device kernel.
    template<class Real>
    [[nodiscard]] auto coefficients(Real dt) const {
        detail::require_step(dt);
        const Real inverse = Real{1}/dt;
        return detail::checked(Derivative<Real, 1>{{inverse, -inverse}});
    }
};

/// @brief Second-order backward derivative using the current unknown and two old values.
/// The caller supplies both history levels. Backward Euler can be used for the first step.
struct Bdf2 {
    /// @brief Computes BDF2 weights on the host when the last two intervals have the same length.
    /// @tparam Real Floating-point type of the time step and weights.
    /// @param dt Positive interval between each pair of adjacent time levels in the formula.
    /// @return Weights for the new value, the last saved value and the value before that.
    template<class Real>
    [[nodiscard]] auto coefficients(Real dt) const {
        detail::require_step(dt);
        return detail::checked(Derivative<Real, 2>{{Real{1.5}/dt, -Real{2}/dt, Real{0.5}/dt}});
    }

    /// @brief Computes BDF2 weights on the host for two possibly different step sizes.
    /// @tparam Real Floating-point type of both time steps and the resulting weights.
    /// @param dt Positive interval from t^n to t^(n+1).
    /// @param previous_dt Positive interval from t^(n-1) to t^n.
    /// @return Weights for the new value, the last saved value and the value before that.
    template<class Real>
    [[nodiscard]] auto coefficients(Real dt, Real previous_dt) const {
        detail::require_step(dt);
        detail::require_step(previous_dt);
        const Real ratio = dt/previous_dt;
        mpi::require(std::isfinite(ratio) && ratio > Real{0},
                     "Time derivative: unrepresentable step ratio");
        // The fraction is dt / (dt + previous_dt). Use the step ratio to avoid
        // forming the sum of the two steps or squaring the ratio in the weights.
        const Real fraction = ratio <= Real{1} ? ratio/(Real{1}+ratio) : Real{1}/(Real{1}+Real{1}/ratio);
        return detail::checked(Derivative<Real, 2>{{(Real{1}+fraction)/dt,
            -(Real{1}/dt+Real{1}/previous_dt), fraction/previous_dt}});
    }
};

} // namespace krona::numerics::time
