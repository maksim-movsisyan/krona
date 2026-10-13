#pragma once

#include <krona/execution/types.hpp>

#include <cmath>
#include <type_traits>

namespace krona::thermodynamics {

/// @brief Calorically perfect gas; contains no boundary conditions or boundary state types.
template<class Real = ComputeReal>
struct IdealGas {
    static_assert(std::is_floating_point_v<Real>);
    using real_type = Real;
    static constexpr bool calorically_perfect = true;
    Real gamma = static_cast<Real>(1.4);
    Real gas_constant = static_cast<Real>(287.05);

    KOKKOS_INLINE_FUNCTION Real cv() const { return gas_constant / (gamma - Real{1}); }
    KOKKOS_INLINE_FUNCTION Real cp() const { return gamma * cv(); }
    KOKKOS_INLINE_FUNCTION Real density(Real pressure, Real temperature) const
    { return pressure / (gas_constant * temperature); }
    KOKKOS_INLINE_FUNCTION Real pressure(Real density, Real temperature) const
    { return density * gas_constant * temperature; }
    KOKKOS_INLINE_FUNCTION Real internal_energy(Real temperature) const { return cv() * temperature; }
    KOKKOS_INLINE_FUNCTION Real enthalpy(Real temperature) const { return cp() * temperature; }
    KOKKOS_INLINE_FUNCTION Real temperature_from_energy(Real energy) const { return energy / cv(); }
    KOKKOS_INLINE_FUNCTION Real temperature_from_enthalpy(Real enthalpy) const { return enthalpy / cp(); }
    KOKKOS_INLINE_FUNCTION Real sound_speed(Real, Real temperature) const
    { return Kokkos::sqrt(gamma * gas_constant * temperature); }
    KOKKOS_INLINE_FUNCTION bool valid_parameters() const
    {
        return Kokkos::isfinite(gamma) && gamma > Real{1} &&
               Kokkos::isfinite(gas_constant) && gas_constant > Real{0} &&
               Kokkos::isfinite(cp()) && cp() > Real{0};
    }
    KOKKOS_INLINE_FUNCTION bool valid(Real pressure, Real temperature) const
    {
        return Kokkos::isfinite(pressure) && pressure > Real{0} &&
               Kokkos::isfinite(temperature) && temperature > Real{0};
    }
};

} // namespace krona::thermodynamics
