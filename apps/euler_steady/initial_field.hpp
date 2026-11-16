#pragma once

#include "cli.hpp"

#include <krona/boundary/physical/state.hpp>

namespace krona::apps::euler_steady {

/// @brief Defines the initial primitive field at an owned cell centre.
template<class Real>
KOKKOS_INLINE_FUNCTION boundary::physical::Primitive<Real, true> initial_field(
    Case problem, boundary::physical::Primitive<Real, true> reference, Real x, Real /*y*/, Real /*z*/)
{
    if (problem == Case::channel) {
        // A pressure perturbation inside the example's unit-length channel.
        const Real coordinate = (x - Real{1} / Real{2}) * Real{8};
        reference.pressure *= Real{1} + Kokkos::exp(-coordinate * coordinate) / Real{20};
    }

    if (problem == Case::wedge) {
        reference.pressure = 100000;
        reference.temperature = 300;
        reference.velocity = {1042., 0., 0.};
    }
    
    // Customize Case::custom here to use other spatial initial conditions.
    return reference;
}

} // namespace krona::apps::euler_steady
