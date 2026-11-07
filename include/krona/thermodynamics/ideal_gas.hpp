#pragma once

#include <krona/thermodynamics/state.hpp>
#include <krona/mpi/error.hpp>
#include <cmath>

namespace krona::thermodynamics {

/// Calorically perfect gas; no transport coefficients or flow velocities.
template<class Real = ComputeReal>
struct IdealGas {
    Real gamma = static_cast<Real>(1.4), gas_constant = static_cast<Real>(287.05);
    Real reference_temperature = 1, reference_pressure = 1;

    void validate() const
    {
        mpi::require(std::isfinite(gamma) && gamma > Real{1} &&
            std::isfinite(gas_constant) && gas_constant > Real{0} &&
            std::isfinite(reference_temperature) && reference_temperature > Real{0} &&
            std::isfinite(reference_pressure) && reference_pressure > Real{0},
            "IdealGas: require finite gamma>1, R>0 and positive reference states");
    }

    KOKKOS_INLINE_FUNCTION Real cv() const { return gas_constant/(gamma - Real{1}); }
    KOKKOS_INLINE_FUNCTION Real cp() const { return gamma*cv(); }

    template<class S>
    KOKKOS_INLINE_FUNCTION State<S> from_pT(S p, S T) const
    {
        if (!(p > S{0}) || !(T > S{0}) || !Kokkos::isfinite(p) || !Kokkos::isfinite(T))
            Kokkos::abort("IdealGas: nonpositive or nonfinite pressure/temperature");
        const S R = static_cast<S>(gas_constant), c_v = static_cast<S>(cv()), c_p = static_cast<S>(cp());
        return {p, T, p/(R*T), c_v*T, c_p*T,
            c_p*Kokkos::log(T/static_cast<S>(reference_temperature)) -
                R*Kokkos::log(p/static_cast<S>(reference_pressure)),
            Kokkos::sqrt(static_cast<S>(gamma)*R*T)};
    }

    template<class S>
    KOKKOS_INLINE_FUNCTION State<S> from_rhoe(S rho, S e) const
    {
        if (!(rho > S{0}) || !(e > S{0}) || !Kokkos::isfinite(rho) || !Kokkos::isfinite(e))
            Kokkos::abort("IdealGas: nonpositive or nonfinite density/internal energy");
        const S T = e/static_cast<S>(cv());
        return from_pT(rho*static_cast<S>(gas_constant)*T, T);
    }

    template<class S>
    KOKKOS_INLINE_FUNCTION State<S> from_ps(S p, S entropy) const
    {
        const S T = static_cast<S>(reference_temperature)*Kokkos::exp(
            (entropy + static_cast<S>(gas_constant)*Kokkos::log(p/static_cast<S>(reference_pressure)))/
            static_cast<S>(cp()));
        return from_pT(p, T);
    }

    template<class S>
    KOKKOS_INLINE_FUNCTION State<S> from_hs(S h, S entropy) const
    {
        const S T = h/static_cast<S>(cp());
        const S p = static_cast<S>(reference_pressure)*Kokkos::exp(
            (static_cast<S>(cp())*Kokkos::log(T/static_cast<S>(reference_temperature)) - entropy)/
            static_cast<S>(gas_constant));
        return from_pT(p, T);
    }

    template<class S>
    KOKKOS_INLINE_FUNCTION RhoEDerivatives<S> rhoe_derivatives(S rho, S e) const
    {
        const S g = static_cast<S>(gamma) - S{1};
        return {g*e, g*rho, S{0}, S{1}/static_cast<S>(cv())};
    }
};

} // namespace krona::thermodynamics
