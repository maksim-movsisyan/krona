#pragma once

#include <krona/numerics/viscous/constitutive.hpp>
#include <type_traits>

namespace krona::numerics::viscous {

namespace detail {

/// @brief Constructs the caller's five-component flux in the constitutive arithmetic type.
template<class Flux, class R>
KOKKOS_INLINE_FUNCTION Flux make_flux(Traction<R> momentum, R energy, R mass = R{0}) {
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(Flux{}.mass)>, R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.momentum.x)>, R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.momentum.y)>, R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.momentum.z)>, R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.energy)>, R>,
                  "Viscous flux container and local arithmetic must use the same scalar type");
    return Flux{mass, {momentum.x, momentum.y, momentum.z}, energy};
}

template<class Flux, class R>
KOKKOS_INLINE_FUNCTION Result<Flux> fail_flux(Status status) {
    const R n = Kokkos::quiet_NaN_v<R>;
    return {make_flux<Flux>(Traction<R>{n, n, n}, n, n), status};
}

} // namespace detail

/// @brief Returns (0,-tau*n,q*n-u*(tau*n)) per unit area for the outward flux balance.
/// @tparam Flux Five-component aggregate chosen by the caller.
/// @param surface Traction and outward heat flux from the same evaluation point.
/// @param velocity Face velocity used in the stress work.
template<class Flux, class R, class V>
KOKKOS_INLINE_FUNCTION Result<Flux> outward_flux(Result<SurfaceTransport<R>> surface, const V& velocity) {
    if (!surface.valid()) return detail::fail_flux<Flux, R>(surface.status);
    const auto u = detail::components<R>(velocity);
    if (!math::isfinite(u)) return detail::fail_flux<Flux, R>(Status::InvalidVelocity);

    const auto traction = surface.value.traction;
    if (!math::isfinite(traction) || !Kokkos::isfinite(surface.value.outward_heat_flux))
        return detail::fail_flux<Flux, R>(Status::NonfiniteResult);
    const R energy = surface.value.outward_heat_flux - u.dot(traction);
    if (!Kokkos::isfinite(energy)) return detail::fail_flux<Flux, R>(Status::NonfiniteResult);
    return {detail::make_flux<Flux>(-traction, energy)};
}

/// @brief Returns the viscous flux for a right-hand-side divergence.
template<class Flux, class R, class V>
KOKKOS_INLINE_FUNCTION Result<Flux> rhs_flux(Result<SurfaceTransport<R>> surface, const V& velocity) {
    auto result = outward_flux<Flux>(surface, velocity);
    if (result.valid()) {
        result.value.momentum.x = -result.value.momentum.x;
        result.value.momentum.y = -result.value.momentum.y;
        result.value.momentum.z = -result.value.momentum.z;
        result.value.energy = -result.value.energy;
    }
    return result;
}

/// @brief Integrates the outward flux density using the supplied positive face area.
template<class Flux, class R, class V>
KOKKOS_INLINE_FUNCTION Result<Flux> integrated_outward_flux(Result<SurfaceTransport<R>> surface, const V& velocity,
                                                            std::type_identity_t<R> area) {
    if (area <= R{0} || !Kokkos::isfinite(area)) return detail::fail_flux<Flux, R>(Status::InvalidArea);
    auto result = outward_flux<Flux>(surface, velocity);
    if (result.valid()) {
        auto& flux = result.value;
        flux.momentum.x *= area;
        flux.momentum.y *= area;
        flux.momentum.z *= area;
        flux.energy *= area;
        if (!Kokkos::isfinite(flux.momentum.x) || !Kokkos::isfinite(flux.momentum.y) ||
            !Kokkos::isfinite(flux.momentum.z) || !Kokkos::isfinite(flux.energy))
            return detail::fail_flux<Flux, R>(Status::NonfiniteResult);
    }
    return result;
}

} // namespace krona::numerics::viscous
