#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>
#include <cmath>

namespace krona::thermodynamics {

/// Material closure for incompressible flow; intentionally has no acoustic EOS API.
template<class Real = ComputeReal>
struct ConstantDensity {
    Real rho = 1000;
    void validate() const
    { mpi::require(std::isfinite(rho) && rho > Real{0}, "ConstantDensity: density must be finite and positive"); }
    template<class S>
    KOKKOS_INLINE_FUNCTION S density(S) const { return static_cast<S>(rho); }
};

/// Constant density and specific heat; e(T_ref)=0, h=e+p/rho.
template<class Real = ComputeReal>
struct ThermalConstantDensity {
    Real rho = 1000, specific_heat = 4200, reference_temperature = 273;
    void validate() const
    {
        ConstantDensity<Real>{rho}.validate();
        mpi::require(std::isfinite(specific_heat) && specific_heat > Real{0} &&
            std::isfinite(reference_temperature) && reference_temperature > Real{0},
            "ThermalConstantDensity: require positive heat capacity and reference temperature");
    }
    template<class S>
    KOKKOS_INLINE_FUNCTION S density(S) const { return static_cast<S>(rho); }
    template<class S>
    KOKKOS_INLINE_FUNCTION S internal_energy(S T) const
    { return static_cast<S>(specific_heat)*(T - static_cast<S>(reference_temperature)); }
    template<class S>
    KOKKOS_INLINE_FUNCTION S enthalpy(S p, S T) const
    { return internal_energy(T) + p/static_cast<S>(rho); }
};

} // namespace krona::thermodynamics
