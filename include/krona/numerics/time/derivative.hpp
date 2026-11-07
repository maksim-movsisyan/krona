#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace krona::numerics::time {

/// @brief Dimensional coefficients: dq/dt = sum_j weights[j]*q^(n+1-j).
/// History is ordered newest first. Custom coefficients must preserve constants.
template<class Real, std::size_t History>
struct Derivative {
    static_assert(std::is_floating_point_v<Real> && History > 0);
    static constexpr std::size_t history_size = History;
    Kokkos::Array<Real, History+1> weights;
};
 
namespace detail {
template<class Real>
void require_step(Real dt)
{
    static_assert(std::is_floating_point_v<Real>);
    mpi::require(std::isfinite(dt) && dt > Real{0},
        "Time derivative: time step must be finite and positive");
}
template<class Real, std::size_t N>
Derivative<Real, N> checked(Derivative<Real, N> derivative)
{
    for (const auto weight : derivative.weights) {
        mpi::require(std::isfinite(weight),
            "Time derivative: unrepresentable coefficient");
    }
    mpi::require(derivative.weights[0] > Real{0},
        "Time derivative: nonpositive current coefficient");
    return derivative;
}
} // namespace detail

struct BackwardEuler {
    template<class Real>
    [[nodiscard]] auto coefficients(Real dt) const
    {
        detail::require_step(dt);
        const Real inverse = Real{1}/dt;
        return detail::checked(Derivative<Real, 1>{{inverse, -inverse}});
    }
};

/// @brief Implicit second-order backward derivative, including unequal adjacent steps.
/// No startup, error control, step acceptance or step-ratio policy is implied.
struct Bdf2 {
    template<class Real>
    [[nodiscard]] auto coefficients(Real dt) const
    {
        detail::require_step(dt);
        return detail::checked(Derivative<Real, 2>{{Real{1.5}/dt, -Real{2}/dt, Real{0.5}/dt}});
    }
    template<class Real>
    [[nodiscard]] auto coefficients(Real dt, Real previous_dt) const
    {
        detail::require_step(dt);
        detail::require_step(previous_dt);
        const Real ratio = dt/previous_dt;
        mpi::require(std::isfinite(ratio) && ratio > Real{0},
                     "Time derivative: unrepresentable step ratio");
        // Avoid forming ratio*ratio or dt+previous_dt.
        const Real fraction = ratio <= Real{1} ? ratio/(Real{1}+ratio) : Real{1}/(Real{1}+Real{1}/ratio);
        return detail::checked(Derivative<Real, 2>{{(Real{1}+fraction)/dt,
            -(Real{1}/dt+Real{1}/previous_dt), fraction/previous_dt}});
    }
};

} // namespace krona::numerics::time
