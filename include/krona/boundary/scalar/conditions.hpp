#pragma once

#include <krona/boundary/context.hpp>
#include <krona/boundary/profiles.hpp>
#include <krona/boundary/scalar/relation.hpp>

#include <concepts>

namespace krona::boundary::scalar {

/// @brief Prescribes phi_f. Provider is a number or a device callable provider(context).
template<class Real = ComputeReal, class Provider = Real>
struct FixedValue {
    Provider value{};
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const FaceContext<Real>& c, Real increment = 0) const {
        const Real prescribed = static_cast<Real>(profile::sample(value, c));
        const Real inverse = Real{1}/c.distance;
        return {Real{0}, prescribed, -inverse, (prescribed-increment)*inverse};
    }
};

/// @brief Prescribes the outward normal derivative, including spatial/time-dependent data.
template<class Real = ComputeReal, class Provider = Real>
struct NormalGradient {
    Provider gradient{};
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const FaceContext<Real>& c, Real increment = 0) const {
        const Real prescribed = static_cast<Real>(profile::sample(gradient, c));
        return {Real{1}, c.distance*prescribed+increment, Real{0}, prescribed};
    }
};

/// @brief Outward physical flux density; a supplied law converts it to a normal derivative.
template<class Real = ComputeReal, class Provider = Real>
struct FixedFlux { Provider flux{}; };

template<class Real = ComputeReal, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto fixed_value(Provider provider) { return FixedValue<Real, Provider>{provider}; }
template<class Real = ComputeReal, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto normal_gradient(Provider provider) { return NormalGradient<Real, Provider>{provider}; }
template<class Real = ComputeReal, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto fixed_flux(Provider provider) { return FixedFlux<Real, Provider>{provider}; }

// Kind traits apply equally to constants and profiles.
template<class C> inline constexpr bool is_fixed_value = false;
template<class R, class P> inline constexpr bool is_fixed_value<FixedValue<R, P>> = true;
template<class C> inline constexpr bool is_normal_gradient = false;
template<class R, class P> inline constexpr bool is_normal_gradient<NormalGradient<R, P>> = true;
template<class C> inline constexpr bool is_fixed_flux = false;
template<class R, class P> inline constexpr bool is_fixed_flux<FixedFlux<R, P>> = true;

// Custom relation conditions can specialize this without adopting a built-in parameter layout.
template<class C> inline constexpr bool requires_flux_law = is_fixed_flux<C>;

struct NoFluxLaw {};

template<class Law, class Real>
concept FluxLaw = requires(const Law& law, const FaceContext<Real>& c, Real flux) {
    { law.normal_gradient(c, flux) } -> std::convertible_to<Real>;
};

/// @brief Custom conditions return Relation<Real> from (context, explicit increment).
template<class Condition, class Real, class Law = NoFluxLaw>
KOKKOS_INLINE_FUNCTION Relation<Real> make_relation(const Condition& condition,
    const FaceContext<Real>& c, Real increment = 0, const Law& = {})
{ return condition(c, increment); }

template<class Real, class Provider, class Law = NoFluxLaw>
KOKKOS_INLINE_FUNCTION Relation<Real> make_relation(const FixedFlux<Real, Provider>& condition,
    const FaceContext<Real>& c, Real increment = 0, const Law& law = {}) {
    if constexpr (FluxLaw<Law, Real>) {
        const Real flux = static_cast<Real>(profile::sample(condition.flux, c));
        if (!Kokkos::isfinite(flux)) return {Real{0}, flux, Real{0}, flux};
        return NormalGradient<Real>{static_cast<Real>(law.normal_gradient(c, flux))}(c, increment);
    } else {
        // All variant alternatives compile, even when no flux condition is assigned.
        Kokkos::abort("Scalar boundary: FIXED_FLUX requires a flux law with normal_gradient(context, flux)");
        return {};
    }
}

} // namespace krona::boundary::scalar