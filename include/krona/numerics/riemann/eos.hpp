#pragma once
 
#include <krona/numerics/riemann/types.hpp>
#include <krona/thermodynamics/ideal_gas.hpp>

namespace krona::numerics::riemann {

/// @brief Borrow an already evaluated EOS result; no extra EOS evaluation or entropy calculation.
template<class ThermodynamicState,class R>
KOKKOS_INLINE_FUNCTION State<R> from_thermodynamics(const ThermodynamicState& q,Vector<R> velocity)
{
    return {static_cast<R>(q.density),velocity,static_cast<R>(q.pressure),
        static_cast<R>(q.internal_energy),static_cast<R>(q.sound_speed)};
}
 
/// @brief Generic AcousticEOS adapter. The supplied EOS keeps its own domain/error policy.
template<class EOS,class R>
KOKKOS_INLINE_FUNCTION State<R> from_pT(const EOS& eos,R pressure,R temperature,Vector<R> velocity)
{ return from_thermodynamics(eos.from_pT(pressure,temperature),velocity); }

/// @brief Fast calorically perfect gas path; no unused entropy/logarithms are evaluated.
template<class Parameter,class R>
KOKKOS_INLINE_FUNCTION State<R> ideal_gas_from_rhop(
    const thermodynamics::IdealGas<Parameter>& gas,R density,Vector<R> velocity,R pressure)
{
    const R gamma=static_cast<R>(gas.gamma);
    if(!Kokkos::isfinite(gamma)||!(gamma>R{1})||!Kokkos::isfinite(density)||!(density>R{0})||
       !Kokkos::isfinite(pressure)||!(pressure>R{0}))
        return {density,velocity,pressure,Kokkos::quiet_NaN_v<R>,Kokkos::quiet_NaN_v<R>};
    return {density,velocity,pressure,pressure/((gamma-R{1})*density),Kokkos::sqrt(gamma*pressure/density)};
}

template<class Parameter,class R>
KOKKOS_INLINE_FUNCTION State<R> from_pT(
    const thermodynamics::IdealGas<Parameter>& gas,R pressure,R temperature,Vector<R> velocity)
{
    const R gas_constant=static_cast<R>(gas.gas_constant);
    if(!Kokkos::isfinite(temperature)||!(temperature>R{0})||
       !Kokkos::isfinite(gas_constant)||!(gas_constant>R{0}))
        return {R{0},velocity,pressure,Kokkos::quiet_NaN_v<R>,Kokkos::quiet_NaN_v<R>};
    return ideal_gas_from_rhop(gas,pressure/(gas_constant*temperature),velocity,pressure);
}

/// @brief Invert conservative densities using an EOS; energy-reference conventions belong to the EOS.
template<class EOS,class R>
KOKKOS_INLINE_FUNCTION State<R> from_conservative(const EOS& eos,Conservative<R> u)
{
    if(!detail::finite(u)||!(u.mass>R{0}))return {};
    const auto velocity=u.momentum*(R{1}/u.mass);
    const R e=u.energy/u.mass-R{0.5}*velocity.norm2();
    if(!Kokkos::isfinite(e))return {};
    return from_thermodynamics(eos.from_rhoe(u.mass,e),velocity);
}

template<class Parameter,class R>
KOKKOS_INLINE_FUNCTION State<R> from_conservative(
    const thermodynamics::IdealGas<Parameter>& gas,Conservative<R> u)
{
    if(!detail::finite(u)||!(u.mass>R{0}))return {};
    const auto velocity=u.momentum*(R{1}/u.mass);
    const R e=u.energy/u.mass-R{0.5}*velocity.norm2();
    const R pressure=(static_cast<R>(gas.gamma)-R{1})*u.mass*e;
    return ideal_gas_from_rhop(gas,u.mass,velocity,pressure);
}

} // namespace krona::numerics::riemann
