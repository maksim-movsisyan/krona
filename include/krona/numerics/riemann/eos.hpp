#pragma once
 
#include <krona/numerics/riemann/types.hpp>
#include <krona/thermodynamics/ideal_gas.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::numerics::riemann {

/// @brief Copies an evaluated thermodynamic state and adds the Cartesian velocity.
/// @tparam ThermodynamicState State type with density, pressure, internal_energy and sound_speed members.
/// @tparam R Scalar type of the velocity and the returned state.
/// @param q Thermodynamic state with specific internal energy e (J/kg).
/// @param velocity Fluid velocity in mesh coordinates.
template<class ThermodynamicState,class R>
KOKKOS_INLINE_FUNCTION State<R> from_thermodynamics(const ThermodynamicState& q,Vector<R> velocity) {
    return {static_cast<R>(q.density),velocity,static_cast<R>(q.pressure),
        static_cast<R>(q.internal_energy),static_cast<R>(q.sound_speed)};
}
 
/// @brief Builds a Riemann state by asking the EOS for the state at a given pressure and temperature.
/// Invalid thermodynamic inputs are handled by the supplied EOS.
/// @tparam EOS Equation of state with from_pT(pressure, temperature).
/// @tparam R Scalar type of the inputs and returned state.
/// @param eos Equation of state for this fluid.
/// @param pressure Absolute static pressure.
/// @param temperature Absolute temperature.
/// @param velocity Fluid velocity in mesh coordinates.
template<class EOS,class R>
KOKKOS_INLINE_FUNCTION State<R> from_pT(const EOS& eos,R pressure,R temperature,Vector<R> velocity)
{ return from_thermodynamics(eos.from_pT(pressure,temperature),velocity); }

/// @brief Computes specific internal energy and sound speed from density and pressure for an ideal gas.
/// If gamma, density or pressure fails the input checks, the solver receives NaNs for e and a.
/// @tparam Parameter Scalar type of the gas constants.
/// @tparam R Scalar type of the face data and the returned state.
/// @param gas Ideal-gas EOS with a constant ratio of specific heats.
/// @param density Mass per unit volume.
/// @param velocity Fluid velocity in mesh coordinates.
/// @param pressure Absolute static pressure.
template<class Parameter,class R>
KOKKOS_INLINE_FUNCTION State<R> ideal_gas_from_rhop(const thermodynamics::IdealGas<Parameter>& gas,
                                                    R density,Vector<R> velocity,R pressure) {
    const R gamma=static_cast<R>(gas.gamma);
    if(!Kokkos::isfinite(gamma)||!(gamma>R{1})||!Kokkos::isfinite(density)||!(density>R{0})||
       !Kokkos::isfinite(pressure)||!(pressure>R{0}))
        return {density,velocity,pressure,Kokkos::quiet_NaN_v<R>,Kokkos::quiet_NaN_v<R>};
    return {density,velocity,pressure,pressure/((gamma-R{1})*density),Kokkos::sqrt(gamma*pressure/density)};
}

/// @brief Recovers density from the ideal gas law and prepares the quantities needed for the flux.
/// @tparam Parameter Scalar type of the gas constants.
/// @tparam R Scalar type of the face data and the returned state.
/// @param gas Ideal-gas EOS providing the gas constant and ratio of specific heats.
/// @param pressure Absolute static pressure.
/// @param temperature Absolute temperature.
/// @param velocity Fluid velocity in mesh coordinates.
template<class Parameter,class R>
KOKKOS_INLINE_FUNCTION State<R> from_pT(const thermodynamics::IdealGas<Parameter>& gas,
                                         R pressure,R temperature,Vector<R> velocity) {
    const R gas_constant=static_cast<R>(gas.gas_constant);
    if(!Kokkos::isfinite(temperature)||!(temperature>R{0})||
       !Kokkos::isfinite(gas_constant)||!(gas_constant>R{0}))
        return {R{0},velocity,pressure,Kokkos::quiet_NaN_v<R>,Kokkos::quiet_NaN_v<R>};
    return ideal_gas_from_rhop(gas,pressure/(gas_constant*temperature),velocity,pressure);
}

/// @brief Recovers velocity and specific internal energy, then asks the EOS for the remaining quantities.
/// Bad conservative data return a zero-initialized state that solvers reject. The EOS handles its own domain.
/// @tparam EOS Equation of state with from_rhoe(density, specific_internal_energy).
/// @tparam R Scalar type of the densities and the returned state.
/// @param eos Equation of state using the same energy reference as u.
/// @param u Density, momentum density and total energy density.
template<class EOS,class R>
KOKKOS_INLINE_FUNCTION State<R> from_conservative(const EOS& eos,flow::Conservative<R> u) {
    flow::InternalEnergyState<R> state;
    if (!flow::try_split_conservative(u, state)) return {};
    return from_thermodynamics(eos.from_rhoe(state.density, state.internal_energy), state.velocity);
}

/// @brief Recovers the ideal-gas face state from conservative densities without computing entropy.
/// If velocity or specific internal energy cannot be recovered, returns a zero-initialized state.
/// @tparam Parameter Scalar type of the gas constants.
/// @tparam R Scalar type of the densities and the returned state.
/// @param gas Ideal-gas EOS providing the ratio of specific heats.
/// @param u Density, momentum density and total energy density for this gas.
template<class Parameter,class R>
KOKKOS_INLINE_FUNCTION State<R> from_conservative(const thermodynamics::IdealGas<Parameter>& gas,flow::Conservative<R> u) {
    flow::InternalEnergyState<R> state;
    if (!flow::try_split_conservative(u, state)) return {};
    // Pressure depends on the internal part of the energy only.
    const R pressure=(static_cast<R>(gas.gamma)-R{1})*state.density*state.internal_energy;
    return ideal_gas_from_rhop(gas, state.density, state.velocity, pressure);
}

} // namespace krona::numerics::riemann
