#pragma once

#include <krona/boundary/context.hpp>
#include <krona/boundary/profiles.hpp>
#include <krona/boundary/scalar/relation.hpp>

#include <concepts>

namespace krona::boundary::scalar {

/// @brief Prescribes the scalar value at the face, as at a wall held at a given temperature.
/// @tparam Real Floating-point type used to compute the face relation.
/// @tparam Provider A number or a device-callable object that returns a value for the current face and time.
template<class Real = ComputeReal, class Provider = Real>
struct FixedValue {
    Provider value{}; ///< Prescribed face value or the provider that supplies it.

    /// @brief Builds the face value and its matching outward normal derivative.
    /// @param c Face context with a positive projected distance from the owner cell centre.
    /// @param increment Explicit tangential correction, typically the owner gradient dotted
    /// with the tangential displacement to the face.
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const FaceContext<Real>& c, Real increment = 0) const {
        const Real prescribed = static_cast<Real>(profile::sample(value, c));
        const Real inverse = Real{1}/c.distance;
        // Solve phi_face = phi_owner + increment + distance * normal_gradient for the derivative.
        return {Real{0}, prescribed, -inverse, (prescribed-increment)*inverse};
    }
};

/// @brief Prescribes the derivative along the outward normal. Use zero for a zero normal gradient.
/// @tparam Real Floating-point type used to compute the face relation.
/// @tparam Provider A number or a device-callable object that returns a derivative for the current face and time.
template<class Real = ComputeReal, class Provider = Real>
struct NormalGradient {
    Provider gradient{}; ///< Prescribed outward normal derivative or its provider.

    /// @brief Builds the face value from the owner value and the prescribed derivative.
    /// @param c Face context with the projected distance from the owner cell centre.
    /// @param increment Explicit tangential correction to the owner value.
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const FaceContext<Real>& c, Real increment = 0) const {
        const Real prescribed = static_cast<Real>(profile::sample(gradient, c));
        return {Real{1}, c.distance*prescribed+increment, Real{0}, prescribed};
    }
};

/// @brief Prescribes an outward flux per unit area. A transport law converts it to a normal derivative.
/// @tparam Real Floating-point type used to compute the face relation.
/// @tparam Provider A number or a device-callable object that returns a flux density for the current face and time.
template<class Real = ComputeReal, class Provider = Real>
struct FixedFlux {
    Provider flux{}; ///< Prescribed outward flux density or its provider.
};

/// @brief Makes a fixed-value condition and deduces the provider type.
/// @tparam Real Floating-point type used to compute the face relation.
/// @tparam Provider Type of the constant or device-callable profile.
/// @param provider Supplies the scalar value at the face.
template<class Real = ComputeReal, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto fixed_value(Provider provider) { return FixedValue<Real, Provider>{provider}; }

/// @brief Makes a normal-gradient condition and deduces the provider type.
/// @tparam Real Floating-point type used to compute the face relation.
/// @tparam Provider Type of the constant or device-callable profile.
/// @param provider Supplies the outward normal derivative at the face.
template<class Real = ComputeReal, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto normal_gradient(Provider provider) { return NormalGradient<Real, Provider>{provider}; }

/// @brief Makes a fixed-flux condition and deduces the provider type.
/// @tparam Real Floating-point type used to compute the face relation.
/// @tparam Provider Type of the constant or device-callable profile.
/// @param provider Supplies the outward flux density at the face.
template<class Real = ComputeReal, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto fixed_flux(Provider provider) { return FixedFlux<Real, Provider>{provider}; }

/// @brief Identifies a FixedValue with any provider type.
/// @tparam C Condition type with cv/ref qualifiers removed.
template<class C> inline constexpr bool is_fixed_value = false;
template<class R, class P> inline constexpr bool is_fixed_value<FixedValue<R, P>> = true;

/// @brief Identifies a NormalGradient with any provider type.
/// @tparam C Condition type with cv/ref qualifiers removed.
template<class C> inline constexpr bool is_normal_gradient = false;
template<class R, class P> inline constexpr bool is_normal_gradient<NormalGradient<R, P>> = true;

/// @brief Identifies a FixedFlux with any provider type.
/// @tparam C Condition type with cv/ref qualifiers removed.
template<class C> inline constexpr bool is_fixed_flux = false;
template<class R, class P> inline constexpr bool is_fixed_flux<FixedFlux<R, P>> = true;

/// @brief Tells Manager to require a flux law before submitting a relation kernel.
/// @tparam C Condition type with cv/ref qualifiers removed.
/// @details Defaults to true for FixedFlux. A custom condition may specialize the trait.
/// The default make_relation overload for custom conditions calls condition(context, increment).
template<class C> inline constexpr bool requires_flux_law = is_fixed_flux<C>;

/// @brief Default argument for operations whose assigned conditions can be evaluated without a flux law.
struct NoFluxLaw {};

/// @brief Checks for a law that converts outward flux density to an outward normal derivative.
/// @tparam Law Device-callable object that provides normal_gradient(context, flux).
/// @tparam Real Floating-point type used for the face context and flux.
template<class Law, class Real>
concept FluxLaw = requires(const Law& law, const FaceContext<Real>& c, Real flux) {
    { law.normal_gradient(c, flux) } -> std::convertible_to<Real>;
};

/// @brief Asks a built-in value/gradient condition or a custom condition for its face relation.
/// @tparam Condition Device-callable object that returns Relation<Real> from (context, increment).
/// @tparam Real Floating-point type used to compute the relation.
/// @tparam Law Flux law accepted by the common interface. This overload uses the condition directly.
/// @param condition Condition assigned to the current face's patch.
/// @param c Geometry and time of the current face.
/// @param increment Explicit tangential correction to the owner value.
template<class Condition, class Real, class Law = NoFluxLaw>
KOKKOS_INLINE_FUNCTION Relation<Real> make_relation(const Condition& condition,
    const FaceContext<Real>& c, Real increment = 0, const Law& = {})
{ return condition(c, increment); }

/// @brief Converts the prescribed outward flux density to a derivative and builds its face relation.
/// @tparam Real Floating-point type used to compute the relation.
/// @tparam Provider Constant or device-callable flux profile.
/// @tparam Law Device-callable flux-to-gradient law.
/// @param condition Prescribed outward flux density for this patch.
/// @param c Geometry and time of the current face.
/// @param increment Explicit tangential correction to the owner value.
/// @param law Converts the sampled flux using the material properties for this face.
template<class Real, class Provider, class Law = NoFluxLaw>
KOKKOS_INLINE_FUNCTION Relation<Real> make_relation(const FixedFlux<Real, Provider>& condition,
    const FaceContext<Real>& c, Real increment = 0, const Law& law = {}) {
    if constexpr (FluxLaw<Law, Real>) {
        const Real flux = static_cast<Real>(profile::sample(condition.flux, c));
        // Let the caller report an invalid profile value before invoking the material law.
        if (!Kokkos::isfinite(flux)) return {Real{0}, flux, Real{0}, flux};
        return NormalGradient<Real>{static_cast<Real>(law.normal_gradient(c, flux))}(c, increment);
    } else {
        // All variant alternatives compile, even when no flux condition is assigned.
        Kokkos::abort("Scalar boundary: FIXED_FLUX requires a flux law with normal_gradient(context, flux)");
        return {};
    }
}

} // namespace krona::boundary::scalar
