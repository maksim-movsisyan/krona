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
    {
        if constexpr (requires { c.neighbour; }) { KOKKOS_ASSERT(c.neighbour < 0); }
        return values(static_cast<std::size_t>(c.owner));
    }
};

/// @brief Explicit face data, when the application already stores a face coefficient.
template<class View>
struct FaceCoefficient {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return values(static_cast<std::size_t>(c.face)); }
};

struct ArithmeticCoefficient {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(Real owner, Real neighbour, Real owner_weight) const
    { return owner_weight*owner + (Real{1}-owner_weight)*neighbour; }
};

/// @brief Series resistance: owner_weight is the neighbour-side distance fraction.
struct HarmonicCoefficient {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(Real owner, Real neighbour, Real owner_weight) const
    {
        // Return invalid data to the operator's asynchronous validation instead of hiding it.
        if (!Kokkos::isfinite(owner) || !(owner > Real{0})) return owner;
        if (!Kokkos::isfinite(neighbour) || !(neighbour > Real{0})) return neighbour;
        return Real{1}/((Real{1}-owner_weight)/owner + owner_weight/neighbour);
    }
};

/// @brief Two-sided cell data on interior faces; owner extrapolation on exterior faces.
/// Cell ghosts must be current. Interior contexts supply neighbour and owner_weight.
template<class View, class Interpolation = HarmonicCoefficient>
struct InterpolatedCellCoefficient {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;
    Interpolation interpolation{};

    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    {
        using Real = std::common_type_t<typename View::non_const_value_type, decltype(c.distance)>;
        const Real owner = static_cast<Real>(values(static_cast<std::size_t>(c.owner)));
        if constexpr (requires { c.neighbour; c.owner_weight; }) {
            if (c.neighbour >= 0) {
                const Real neighbour = static_cast<Real>(values(static_cast<std::size_t>(c.neighbour)));
                if (!Kokkos::isfinite(owner) || !(owner > Real{0})) return owner;
                if (!Kokkos::isfinite(neighbour) || !(neighbour > Real{0})) return neighbour;
                return interpolation(owner, neighbour, static_cast<Real>(c.owner_weight));
            }
        }
        return owner;
    }
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