#pragma once

#include <krona/numerics/riemann/types.hpp>
#include <krona/flow/state_jacobian.hpp>
#include <Kokkos_Array.hpp>
#include <cstddef>

namespace krona::numerics::riemann {

using math::Matrix5;

namespace detail {
    
template<class R>
KOKKOS_INLINE_FUNCTION auto velocity_components(Vector<R> velocity)
{ return Kokkos::Array<R, 3>{velocity.x, velocity.y, velocity.z}; }

}

/// @brief Adapts a Riemann state to the shared conservative-to-primitive derivative.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION Matrix5<R> primitive_from_conservative_jacobian(State<R> state, const EOS& eos) {
    return flow::primitive_from_conservative_jacobian(
        flow::InternalEnergyState<R>{state.density, state.velocity, state.internal_energy}, eos);
}

/// @brief Adapts a Riemann state to the shared primitive-to-conservative derivative.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION Matrix5<R> conservative_from_primitive_jacobian(State<R> state, const EOS& eos) {
    return flow::conservative_from_primitive_jacobian(
        flow::InternalEnergyState<R>{state.density, state.velocity, state.internal_energy}, eos);
}

/// @brief Differentiates physical_flux() at one state with respect to the five conservative densities.
/// Rows are the mass, three momentum and energy fluxes per unit area. Columns are the corresponding densities.
/// @tparam R Scalar type of the state, normal and matrix entries.
/// @tparam EOS Equation of state with rhoe_derivatives(density, specific_internal_energy).
/// @param state Admissible state at which to evaluate the derivatives.
/// @param normal Fixed unit normal pointing from left to right.
/// @param eos Supplies the pressure derivatives through the state conversion.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION Matrix5<R> physical_flux_jacobian(State<R> state, Vector<R> normal, const EOS& eos) {
    const auto primitive = primitive_from_conservative_jacobian(state, eos);
    const auto velocity = detail::velocity_components(state.velocity);
    const auto n = detail::velocity_components(normal);
    const R un = state.velocity.dot(normal);
    const R mass_flux = state.density * un;
    const R energy_plus_pressure = conservative(state).total_energy_density + state.pressure;
    Matrix5<R> jacobian;

    // Each column varies one conserved density. The mass flux is momentum dotted with the fixed normal.
    for (std::size_t column = 0; column < 5; ++column) {
        const R dmass = column > 0 && column < 4 ? n[column - 1] : R{0};
        const R dp = primitive(0, column);
        R dun = 0;
        jacobian(0, column) = dmass;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const R du = primitive(axis + 1, column);
            dun += n[axis] * du;
            jacobian(axis + 1, column) = velocity[axis] * dmass + mass_flux * du + n[axis] * dp;
        }

        // Differentiate the energy flux (rho*E + p) * un.
        const R denergy = column == 4 ? R{1} : R{0};
        jacobian(4, column) = un * (denergy + dp) + energy_plus_pressure * dun;
    }
    return jacobian;
}

} // namespace krona::numerics::riemann
