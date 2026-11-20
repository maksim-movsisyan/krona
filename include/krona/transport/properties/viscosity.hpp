#pragma once

#include <krona/transport/properties/types.hpp>
#include <krona/transport/properties/detail/validation.hpp>

namespace krona::transport::properties {

/// @brief Returns a fixed dynamic viscosity at every state.
/// @tparam Parameter Floating-point type used to store the configured viscosity.
template<class Parameter = ComputeReal>
struct ConstantViscosity {
    Parameter value = 0; ///< Dynamic viscosity mu in Pa s in SI.

    /// @brief Requires the configured viscosity to be finite and nonnegative on the host.
    void validate() const noexcept {
        mpi::require(detail::nonnegative(value),
            "Transport properties: constant dynamic viscosity must be finite and nonnegative");
    }

    /// @brief Returns the configured viscosity converted to the requested precision.
    /// @tparam Real Floating-point type of the returned value.
    /// @tparam State Any state type. This law uses only the stored coefficient.
    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State&) const {
        static_assert(std::is_floating_point_v<Real>);
        return static_cast<Real>(value);
    }
};
template<class P> ConstantViscosity(P) -> ConstantViscosity<P>;

/// @brief Evaluates a gas's dynamic viscosity as a function of absolute temperature using Sutherland's law.
/// The caller selects a temperature range appropriate to the fitted parameters.
/// @tparam Parameter Floating-point type of the three stored parameters.
template<class Parameter = ComputeReal>
struct SutherlandViscosity {
    Parameter reference_viscosity = 0;     ///< Dynamic viscosity at the reference temperature.
    Parameter reference_temperature = 0;   ///< Absolute reference temperature in K in SI.
    Parameter sutherland_temperature = 0;  ///< Sutherland fit constant in the same units as temperature.

    /// @brief Checks the configured reference viscosity, reference temperature and fit constant on the host.
    void validate() const noexcept {
        mpi::require(detail::positive(reference_viscosity) &&
                     detail::positive(reference_temperature) &&
                     detail::nonnegative(sutherland_temperature),
            "Transport properties: Sutherland requires finite mu_ref>0, T_ref>0 and S>=0");
    }

    /// @brief Returns dynamic viscosity, or NaN for a nonfinite or nonpositive temperature.
    /// @tparam Real Floating-point type used for evaluation and the result.
    /// @tparam State Type with a temperature member convertible to Real.
    /// @param state Supplies the local absolute temperature.
    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state) const {
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
