#pragma once

#include <krona/numerics/riemann/types.hpp>
#include <krona/flow/state_jacobian.hpp>
#include <cstddef>

namespace krona::numerics::riemann {

/// @brief Differentiates physical_flux() at one state with respect to the five conservative densities.
/// Rows are the mass, three momentum and energy fluxes per unit area. Columns are the corresponding densities.
/// @tparam R Scalar type of the state, normal and matrix entries.
/// @tparam EOS Equation of state with rhoe_derivatives(density, specific_internal_energy).
/// @param state Admissible state at which to evaluate the derivatives.
/// @param normal Fixed unit normal pointing from left to right.
/// @param eos Supplies the pressure derivatives through the state conversion.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION math::Matrix5<R> physical_flux_jacobian(State<R> state, Vector<R> normal, const EOS& eos) {
    const flow::InternalEnergyState<R> local{state.density, state.velocity, state.internal_energy};
    const auto primitive = flow::primitive_from_conservative_jacobian(local, eos);
    const R un = state.velocity.dot(normal);
    const R mass_flux = state.density * un;
    const R energy_plus_pressure = conservative(state).total_energy_density + state.pressure;
    math::Matrix5<R> jacobian;

    // Each column varies one conserved density. The mass flux is momentum dotted with the fixed normal.
    for (std::size_t column = 0; column < 5; ++column) {
        const R dmass = column > 0 && column < 4 ? normal[column - 1] : R{0};
        const R dp = primitive(0, column);
        R dun = 0;
        jacobian(0, column) = dmass;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const R du = primitive(axis + 1, column);
            dun += normal[axis] * du;
            jacobian(axis + 1, column) = state.velocity[axis] * dmass + mass_flux * du + normal[axis] * dp;
        }

        // Differentiate the energy flux (rho*E + p) * un.
        const R denergy = column == 4 ? R{1} : R{0};
        jacobian(4, column) = un * (denergy + dp) + energy_plus_pressure * dun;
    }
    return jacobian;
}

} // namespace krona::numerics::riemann
