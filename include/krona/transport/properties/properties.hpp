#pragma once

#include <krona/transport/properties/viscosity.hpp>
#include <krona/transport/properties/thermal.hpp>

namespace krona::transport::properties {

/// @brief Value-owned device-copyable policies; no arrays, EOS ownership, allocation or synchronization.
template<class Viscosity, class Conductivity>
struct TransportProperties {
    Viscosity viscosity;
    Conductivity conductivity;

    void validate() const
    {
        detail::validate(viscosity);
        detail::validate(conductivity);
    }

    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Coefficients<Real> evaluate(const State& state) const
    {
        const Real mu = static_cast<Real>(viscosity.template evaluate<Real>(state));
        const Real lambda = static_cast<Real>(conductivity.template evaluate<Real>(state, mu));
        return {mu, lambda};
    }

    /// @brief Default arithmetic uses ComputeReal regardless of parameter/state storage precision.
    template<class State>
    KOKKOS_INLINE_FUNCTION Coefficients<ComputeReal> operator()(const State& state) const
    { return evaluate<ComputeReal>(state); }
};
template<class V, class K> TransportProperties(V, K) -> TransportProperties<V, K>;

} // namespace krona::transport::properties
