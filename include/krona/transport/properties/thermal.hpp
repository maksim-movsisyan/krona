#pragma once

#include <krona/transport/properties/types.hpp>
#include <krona/transport/properties/detail/validation.hpp>

namespace krona::transport::properties {

/// @brief Supplies a fixed specific heat capacity at constant pressure.
/// @tparam Parameter Floating-point type used to store the configured heat capacity.
template<class Parameter = ComputeReal>
struct ConstantHeatCapacity {
    Parameter value = 0; ///< Heat capacity cp per unit mass, in J/(kg K) in SI.

    /// @brief Requires the configured heat capacity to be finite and strictly positive on the host.
    void validate() const noexcept {
        mpi::require(detail::positive(value),
            "Transport properties: constant cp must be finite and positive");
    }

    /// @brief Returns the stored heat capacity converted to the requested precision.
    /// @tparam Real Floating-point type of the returned value.
    /// @tparam State Any state type. This law uses only the stored coefficient.
    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State&) const {
        static_assert(std::is_floating_point_v<Real>);
        return static_cast<Real>(value);
    }
};
template<class P> ConstantHeatCapacity(P) -> ConstantHeatCapacity<P>;

/// @brief Thermal conductivity independent of viscosity and thermodynamic state.
/// @tparam Parameter Floating-point type used to store the configured conductivity.
template<class Parameter = ComputeReal>
struct ConstantConductivity {
    Parameter value = 0; ///< Conductivity lambda in W/(m K) in SI.

    /// @brief Requires the configured conductivity to be finite and nonnegative on the host.
    void validate() const noexcept {
        mpi::require(detail::nonnegative(value),
            "Transport properties: constant thermal conductivity must be finite and nonnegative");
    }

    /// @brief Returns the stored conductivity. Accepts a state and viscosity to match the policy interface.
    /// @tparam Real Floating-point type of the supplied viscosity and returned conductivity.
    /// @tparam State Any state type. This law uses only the stored coefficient.
    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State&, Real) const {
        static_assert(std::is_floating_point_v<Real>);
        return static_cast<Real>(value);
    }
};
template<class P> ConstantConductivity(P) -> ConstantConductivity<P>;

/// @brief Computes thermal conductivity as lambda = mu * cp / Pr for a fixed Prandtl number.
/// @tparam HeatCapacity Policy with evaluate<Real>(state) returning specific heat at constant pressure.
/// @tparam Parameter Floating-point type used to store the Prandtl number.
template<class HeatCapacity, class Parameter = ComputeReal>
struct ConstantPrandtl {
    Parameter prandtl = 0;        ///< Dimensionless ratio of momentum to thermal diffusivity.
    HeatCapacity heat_capacity{}; ///< Supplies cp from the same state used to compute viscosity.

    /// @brief Requires positive finite Pr and runs the heat-capacity policy's host check when available.
    void validate() const noexcept {
        mpi::require(detail::positive(prandtl),
            "Transport properties: Prandtl number must be finite and positive");
        detail::validate(heat_capacity);
    }

    /// @brief Evaluates cp once and returns NaN conductivity if cp is nonfinite or nonpositive.
    /// @tparam Real Floating-point type used to evaluate conductivity.
    /// @tparam State Type providing the fields read by the heat-capacity policy.
    /// @param state Local data at which to evaluate cp.
    /// @param viscosity Dynamic viscosity already evaluated at this state.
    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state, Real viscosity) const {
        static_assert(std::is_floating_point_v<Real>);
        const Real cp = static_cast<Real>(heat_capacity.template evaluate<Real>(state));
        if (!Kokkos::isfinite(cp) || !(cp > Real{0})) return Kokkos::quiet_NaN_v<Real>;
        return viscosity * (cp/static_cast<Real>(prandtl));
    }
};
template<class P, class C> ConstantPrandtl(P, C) -> ConstantPrandtl<C, P>;

} // namespace krona::transport::properties
