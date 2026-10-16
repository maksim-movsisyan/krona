#pragma once

#include <krona/execution/types.hpp>

#include <type_traits>

namespace krona::transport {

/// @brief A constant transport coefficient; allocates no storage.
template<class Real = ComputeReal>
struct UniformCoefficient {
    Real value = 1;
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context&) const { return value; }
};

/// @brief Explicit owner-cell extrapolation to an exterior face; View may be strided.
template<class View>
struct OwnerCellCoefficient {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return values(static_cast<std::size_t>(c.owner)); }
};

/// @brief Scalar isotropic constitutive law q_n = -coefficient(context) * d(phi)/dn.
/// Providers may read existing fields or evaluate a model; no face array is required.
template<class Coefficient>
struct IsotropicDiffusion {
    Coefficient coefficient;

    template<class Context, class Value>
    KOKKOS_INLINE_FUNCTION auto normal_gradient(const Context& c, Value outward_flux) const
    {
        using Real = std::common_type_t<decltype(coefficient(c)), Value>;
        static_assert(std::is_floating_point_v<Real>);
        const Real value = static_cast<Real>(coefficient(c));
        KOKKOS_ASSERT(Kokkos::isfinite(value) && value > Real{0});
        return -static_cast<Real>(outward_flux)/value;
    }

    template<class Context, class Value>
    KOKKOS_INLINE_FUNCTION auto outward_flux(const Context& c, Value normal_gradient_value) const
    {
        using Real = std::common_type_t<decltype(coefficient(c)), Value>;
        static_assert(std::is_floating_point_v<Real>);
        const Real value = static_cast<Real>(coefficient(c));
        KOKKOS_ASSERT(Kokkos::isfinite(value) && value >= Real{0});
        return -value*static_cast<Real>(normal_gradient_value);
    }
};

} // namespace krona::transport
