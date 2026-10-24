#pragma once

#include <krona/transport/properties/types.hpp>
#include <krona/transport/properties/detail/validation.hpp>

namespace krona::transport::properties {

/// @brief Specific heat capacity at constant pressure; a value, not an EOS reference.
template<class Parameter = ComputeReal>
struct ConstantHeatCapacity {
    Parameter value = 0;

    void validate() const
    {
        mpi::require(detail::positive(value),
            "Transport properties: constant cp must be finite and positive");
    }

    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State&) const
    {
        static_assert(std::is_floating_point_v<Real>);
        return static_cast<Real>(value);
    }
};
template<class P> ConstantHeatCapacity(P) -> ConstantHeatCapacity<P>;

/// @brief Thermal conductivity independent of viscosity and thermodynamic state.
template<class Parameter = ComputeReal>
struct ConstantConductivity {
    Parameter value = 0;

    void validate() const
    {
        mpi::require(detail::nonnegative(value),
            "Transport properties: constant thermal conductivity must be finite and nonnegative");
    }

    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State&, Real) const
    {
        static_assert(std::is_floating_point_v<Real>);
        return static_cast<Real>(value);
    }
};
template<class P> ConstantConductivity(P) -> ConstantConductivity<P>;

/// @brief lambda=mu*cp/Pr; reuses the supplied viscosity and evaluates cp once.
template<class HeatCapacity, class Parameter = ComputeReal>
struct ConstantPrandtl {
    Parameter prandtl = 0;
    HeatCapacity heat_capacity{};

    void validate() const
    {
        mpi::require(detail::positive(prandtl),
            "Transport properties: Prandtl number must be finite and positive");
        detail::validate(heat_capacity);
    }

    template<class Real = ComputeReal, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state, Real viscosity) const
    {
        static_assert(std::is_floating_point_v<Real>);
        const Real cp = static_cast<Real>(heat_capacity.template evaluate<Real>(state));
        if (!Kokkos::isfinite(cp) || !(cp > Real{0})) return Kokkos::quiet_NaN_v<Real>;
        return viscosity * (cp/static_cast<Real>(prandtl));
    }
};
template<class P, class C> ConstantPrandtl(P, C) -> ConstantPrandtl<C, P>;

} // namespace krona::transport::properties
