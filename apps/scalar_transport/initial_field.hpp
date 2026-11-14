#pragma once

#include "analytic_solution.hpp"

namespace krona::apps::scalar_transport {

/// @brief Initial scalar value at a cell centroid. Customize the custom case here.
template<class Real>
KOKKOS_INLINE_FUNCTION Real initial_field(Case problem, Real x, Real y, Real z) {
    if (problem == Case::custom) return Real{0};
    return analytic_solution(problem, x, y, z, Real{0}, Real{0});
}

} // namespace krona::apps::scalar_transport
