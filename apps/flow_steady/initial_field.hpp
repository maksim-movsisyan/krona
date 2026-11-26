#pragma once

#include "cli.hpp"

#include <krona/boundary/physical/state.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

namespace krona::apps::flow_steady {

/// @brief Defines the initial primitive field at an owned cell centre.
template<class Real>
struct InitialField {
    Case problem;
    boundary::physical::Primitive<Real, true> reference;

    KOKKOS_INLINE_FUNCTION auto operator()(Real x, Real /*y*/, Real /*z*/) const {
        auto state = reference;
        if (problem == Case::channel) {
            // A pressure perturbation inside the example's unit-length channel.
            const Real coordinate = (x - Real{1} / Real{2}) * Real{8};
            state.pressure *= Real{1} + Kokkos::exp(-coordinate * coordinate) / Real{20};
        }

        // Customize Case::custom here to use other spatial initial conditions.
        return state;
    }
};

} // namespace krona::apps::flow_steady
