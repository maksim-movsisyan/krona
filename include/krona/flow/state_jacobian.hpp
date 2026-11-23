#pragma once

#include <krona/flow/conservative.hpp>
#include <krona/math/matrix5.hpp>

namespace krona::flow {

/// @brief Differentiates (p, u, v, w, T) with respect to (rho, rho*u, rho*v, rho*w, rho*E).
/// Here E is the total energy per unit mass, so rho*E is the energy density.
/// @tparam R Scalar type of the state and matrix entries.
/// @tparam EOS Equation of state with rhoe_derivatives(density, specific_internal_energy).
/// @param state Admissible state at which to evaluate the derivatives.
/// @param eos Supplies the derivatives of pressure and temperature with respect to rho and e.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION math::Matrix5<R> primitive_from_conservative_jacobian(const InternalEnergyState<R>& state, const EOS& eos) {
    const auto thermo = eos.rhoe_derivatives(state.density, state.internal_energy);
    const auto& velocity = state.velocity;
    const R inv_density = R{1} / state.density;

    // Changing density at fixed momentum and energy changes both velocity and specific internal energy.
    const R de_drho = (R{0.5} * state.velocity.norm2() - state.internal_energy) * inv_density;
    math::Matrix5<R> jacobian;
    jacobian(0, 0) = thermo.dp_drho_e + thermo.dp_de_rho * de_drho;
    jacobian(4, 0) = thermo.dT_drho_e + thermo.dT_de_rho * de_drho;
    jacobian(0, 4) = thermo.dp_de_rho * inv_density;
    jacobian(4, 4) = thermo.dT_de_rho * inv_density;

    // Use the internal energy derivative for the pressure and temperature rows, and u = momentum/rho for velocity.
    for (std::size_t axis = 0; axis < 3; ++axis) {
        jacobian(0, axis + 1) = -jacobian(0, 4) * velocity[axis];
        jacobian(4, axis + 1) = -jacobian(4, 4) * velocity[axis];
        jacobian(axis + 1, 0) = -velocity[axis] * inv_density;
        jacobian(axis + 1, axis + 1) = inv_density;
    }
    return jacobian;
}

/// @brief Differentiates (rho, rho*u, rho*v, rho*w, rho*E) with respect to (p, u, v, w, T).
/// The caller must supply a state where the EOS map from (rho, e) to (p, T) has an invertible derivative.
/// @tparam R Scalar type of the state and matrix entries.
/// @tparam EOS Equation of state with rhoe_derivatives(density, specific_internal_energy).
/// @param state Admissible state at which to evaluate the derivatives.
/// @param eos Supplies the derivatives of pressure and temperature with respect to rho and e.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION math::Matrix5<R> conservative_from_primitive_jacobian(const InternalEnergyState<R>& state, const EOS& eos) {
    const auto thermo = eos.rhoe_derivatives(state.density, state.internal_energy);

    // Invert only the 2 by 2 thermodynamic part to recover the rho and e derivatives in p and T.
    const R determinant = thermo.dp_drho_e * thermo.dT_de_rho - thermo.dp_de_rho * thermo.dT_drho_e;
    const R inv_det = R{1} / determinant;

    const R drho_dp =  thermo.dT_de_rho * inv_det;
    const R drho_dT = -thermo.dp_de_rho * inv_det;
    const R de_dp = -thermo.dT_drho_e * inv_det;
    const R de_dT = thermo.dp_drho_e * inv_det;

    const R total_energy = state.internal_energy + R{0.5} * state.velocity.norm2();
    const auto& velocity = state.velocity;

    // Multiply by density to form momentum and energy, and differentiate those products.
    math::Matrix5<R> jacobian;
    jacobian(0, 0) = drho_dp;
    jacobian(0, 4) = drho_dT;
    jacobian(4, 0) = total_energy * drho_dp + state.density * de_dp;
    jacobian(4, 4) = total_energy * drho_dT + state.density * de_dT;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        jacobian(axis + 1, 0) = velocity[axis] * drho_dp;
        jacobian(axis + 1, 4) = velocity[axis] * drho_dT;
        jacobian(axis + 1, axis + 1) = state.density;
        jacobian(4, axis + 1) = state.density * velocity[axis];
    }
    return jacobian;
}

} // namespace krona::flow
