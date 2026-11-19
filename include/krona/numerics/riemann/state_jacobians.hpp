#pragma once

#include <krona/numerics/riemann/types.hpp>
#include <Kokkos_Array.hpp>
#include <cstddef>

namespace krona::numerics::riemann {

/// @brief Dense 5 by 5 matrix with rows for outputs and columns for input variables.
/// @tparam R Scalar type of the entries.
template<class R>
struct Matrix5 {
    Kokkos::Array<R, 25> values{}; ///< Zero-initialized entries stored row by row.

    /// @brief Returns a writable entry at the given zero-based row and column.
    KOKKOS_INLINE_FUNCTION R& operator()(std::size_t row, std::size_t column)
    { return values[5 * row + column]; }

    /// @brief Reads the entry at the given zero-based row and column.
    KOKKOS_INLINE_FUNCTION R operator()(std::size_t row, std::size_t column) const
    { return values[5 * row + column]; }
};

namespace detail {
    
template<class R>
KOKKOS_INLINE_FUNCTION auto velocity_components(Vector<R> velocity)
{ return Kokkos::Array<R, 3>{velocity.x, velocity.y, velocity.z}; }

}

/// @brief Differentiates (p, u, v, w, T) with respect to (rho, rho*u, rho*v, rho*w, rho*E).
/// Here E is the total energy per unit mass, so rho*E is the energy density.
/// @tparam R Scalar type of the state and matrix entries.
/// @tparam EOS Equation of state with rhoe_derivatives(density, specific_internal_energy).
/// @param state Admissible state at which to evaluate the derivatives.
/// @param eos Supplies the derivatives of pressure and temperature with respect to rho and e.
template<class R, class EOS>
KOKKOS_INLINE_FUNCTION Matrix5<R> primitive_from_conservative_jacobian(State<R> state, const EOS& eos) {
    const auto thermo = eos.rhoe_derivatives(state.density, state.internal_energy);
    const auto velocity = detail::velocity_components(state.velocity);
    const R inv_density = R{1} / state.density;

    // Changing density at fixed momentum and energy changes both velocity and specific internal energy.
    const R de_drho = (R{0.5} * state.velocity.norm2() - state.internal_energy) * inv_density;
    Matrix5<R> jacobian;
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
KOKKOS_INLINE_FUNCTION Matrix5<R> conservative_from_primitive_jacobian(State<R> state, const EOS& eos) {
    const auto thermo = eos.rhoe_derivatives(state.density, state.internal_energy);

    // Invert only the 2 by 2 thermodynamic part to recover the rho and e derivatives in p and T.
    const R determinant = thermo.dp_drho_e * thermo.dT_de_rho - thermo.dp_de_rho * thermo.dT_drho_e;
    const R inv_det = R{1} / determinant;

    const R drho_dp =  thermo.dT_de_rho * inv_det;
    const R drho_dT = -thermo.dp_de_rho * inv_det;
    const R de_dp = -thermo.dT_drho_e * inv_det;
    const R de_dT = thermo.dp_drho_e * inv_det;

    const R total_energy = state.internal_energy + R{0.5} * state.velocity.norm2();
    const auto velocity = detail::velocity_components(state.velocity);

    // Multiply by density to form momentum and energy, and differentiate those products.
    Matrix5<R> jacobian;
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
