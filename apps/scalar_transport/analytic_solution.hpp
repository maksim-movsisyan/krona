#pragma once

#include "velocity.hpp"

#include <Kokkos_MathematicalFunctions.hpp>

#include <numbers>

namespace krona::apps::scalar_transport {

inline constexpr double gaussian_centre = 1.25;
inline constexpr double gaussian_width = 0.12;

[[nodiscard]] inline bool has_analytic_solution(Case problem) noexcept {
    return problem == Case::diffusion || problem == Case::gaussian;
}

/// @brief Exact centre-point value for the supplied example PDEs with zero source.
template<class Real>
KOKKOS_INLINE_FUNCTION Real analytic_solution(Case problem, Real x, Real, Real, Real time, Real diffusivity) {
    if (problem == Case::diffusion) {
        const Real pi = std::numbers::pi_v<Real>;
        return Kokkos::sin(pi*x) * Kokkos::exp(-pi*pi*diffusivity*time);
    }
    const Real sigma = static_cast<Real>(gaussian_width);
    const Real variance = sigma*sigma + Real{2}*diffusivity*time;
    const Real offset = x - static_cast<Real>(gaussian_centre) - static_cast<Real>(gaussian_speed)*time;
    return sigma/Kokkos::sqrt(variance) * Kokkos::exp(-offset*offset/(Real{2}*variance));
}

} // namespace krona::apps::scalar_transport
