#pragma once

#include <krona/transport/properties/viscosity.hpp>
#include <krona/transport/properties/thermal.hpp>

namespace krona::transport::properties {

/// @brief Combines two property models and stores them by value for use in host or device code.
/// @tparam Viscosity Policy with evaluate<Real>(state) returning dynamic viscosity.
/// @tparam Conductivity Policy with evaluate<Real>(state, mu) returning thermal conductivity.
template<class Viscosity, class Conductivity>
struct TransportProperties {
    Viscosity viscosity;       ///< Model for dynamic viscosity.
    Conductivity conductivity; ///< Model for thermal conductivity.

    /// @brief Runs the available parameter checks on the host before the models are used in kernels.
    void validate() const noexcept {
        detail::validate(viscosity);
        detail::validate(conductivity);
    } 

    /// @brief Evaluates molecular coefficients at one state. Use Coefficients::valid() to check the result.
    /// @tparam Real Floating-point type used for the policy calls and their results.
    /// @tparam State Type providing the state fields needed by the two policies.
    /// @param state Local data for property evaluation, such as the absolute temperature.
    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Coefficients<Real> evaluate(const State& state) const {
        // Conductivity models such as ConstantPrandtl reuse this same viscosity.
        const Real mu = static_cast<Real>(viscosity.template evaluate<Real>(state));
        const Real lambda = static_cast<Real>(conductivity.template evaluate<Real>(state, mu));
        return {mu, lambda};
    }

    /// @brief Evaluates both coefficients in ComputeReal.
    /// @tparam State Type providing the state fields needed by the two policies.
    /// @param state Local data for property evaluation.
    template<class State>
    KOKKOS_INLINE_FUNCTION Coefficients<ComputeReal> operator()(const State& state) const
    { return evaluate<ComputeReal>(state); }
};
template<class V, class K> TransportProperties(V, K) -> TransportProperties<V, K>;

} // namespace krona::transport::properties
