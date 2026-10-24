#pragma once

#include <krona/transport/properties/types.hpp>
#include <krona/transport/properties/detail/validation.hpp>

namespace krona::transport::properties {

/// @brief Dynamic viscosity, not kinematic viscosity. Does not inspect the state.
template<class Parameter = ComputeReal>
struct ConstantViscosity {
    Parameter value = 0;

    void validate() const
    {
        mpi::require(detail::nonnegative(value),
            "Transport properties: constant dynamic viscosity must be finite and nonnegative");
    }

    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State&) const
    {
        static_assert(std::is_floating_point_v<Real>);
        return static_cast<Real>(value);
    }
};
template<class P> ConstantViscosity(P) -> ConstantViscosity<P>;

/// @brief Sutherland gas viscosity. Absolute temperatures; no implicit clipping or validity range.
template<class Parameter = ComputeReal>
struct SutherlandViscosity {
    Parameter reference_viscosity = 0;
    Parameter reference_temperature = 0;
    Parameter sutherland_temperature = 0;

    void validate() const
    {
        mpi::require(detail::positive(reference_viscosity) &&
                     detail::positive(reference_temperature) &&
                     detail::nonnegative(sutherland_temperature),
            "Transport properties: Sutherland requires finite mu_ref>0, T_ref>0 and S>=0");
    }

    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state) const
    {
        static_assert(std::is_floating_point_v<Real>);
        const Real temperature = static_cast<Real>(state.temperature);
        if (!Kokkos::isfinite(temperature) || !(temperature > Real{0}))
            return Kokkos::quiet_NaN_v<Real>;
        const Real reference = static_cast<Real>(reference_temperature);
        const Real shift = static_cast<Real>(sutherland_temperature);
        // Equivalent to (T/T_ref)^(3/2)*(T_ref+S)/(T+S), without forming T^(3/2).
        return static_cast<Real>(reference_viscosity) * Kokkos::sqrt(temperature/reference) *
               ((Real{1} + shift/reference)/(Real{1} + shift/temperature));
    }
};
template<class P> SutherlandViscosity(P, P, P) -> SutherlandViscosity<P>;

} // namespace krona::transport::properties
