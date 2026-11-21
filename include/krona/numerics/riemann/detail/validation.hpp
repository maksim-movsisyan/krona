#pragma once

#include <krona/numerics/riemann/types.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::numerics::riemann::detail {

template<class R>
KOKKOS_INLINE_FUNCTION bool finite(Vector<R> v) {
    return Kokkos::isfinite(v.x) && Kokkos::isfinite(v.y) && Kokkos::isfinite(v.z);
}

template<class R>
KOKKOS_INLINE_FUNCTION bool finite(Flux<R> flux) {
    return Kokkos::isfinite(flux.mass) && finite(flux.momentum) && Kokkos::isfinite(flux.energy);
}

template<class R>
KOKKOS_INLINE_FUNCTION bool finite(flow::Conservative<R> u) {
    return Kokkos::isfinite(u.density) && finite(u.momentum) && Kokkos::isfinite(u.total_energy_density);
}

/// @brief Checks the side data shared by all solvers. The sign of e depends on its reference.
template<class R>
KOKKOS_INLINE_FUNCTION bool valid(State<R> s) {
    return Kokkos::isfinite(s.density) && s.density > R{0} && finite(s.velocity) &&
           Kokkos::isfinite(s.pressure) && s.pressure > R{0} && Kokkos::isfinite(s.internal_energy) &&
           Kokkos::isfinite(s.sound_speed) && s.sound_speed > R{0};
}

/// @brief Reports a solver failure and marks unusable numerical outputs with NaNs.
template<class R>
KOKKOS_INLINE_FUNCTION Result<R> failure(Status status) {
    const R nan = Kokkos::quiet_NaN_v<R>;
    return {{nan, {nan, nan, nan}, nan}, nan, status};
}

/// @brief Checks a unit normal, allowing for rounding in the mesh geometry.
template<class R>
KOKKOS_INLINE_FUNCTION bool unit(Vector<R> n) {
    const R norm = n.norm2();
    const R tolerance = R{128} * (Kokkos::epsilon_v<R> + static_cast<R>(Kokkos::epsilon_v<GeometryReal>));
    return finite(n) && Kokkos::isfinite(norm) && Kokkos::abs(norm - R{1}) <= tolerance;
}

/// @brief Checks the face geometry and both sides, including the conversion to densities.
template<class R>
KOKKOS_INLINE_FUNCTION Status input_status(State<R> l, State<R> r, Vector<R> n) {
    if (!unit(n)) return Status::InvalidNormal;
    if (!valid(l) || !valid(r) || !finite(conservative(l)) || !finite(conservative(r)))
        return Status::InvalidState;
    return Status::Success;
}

/// @brief Packages the flux only when its entries and the CFL speed are usable.
template<class R>
KOKKOS_INLINE_FUNCTION Result<R> finish(Flux<R> flux, R speed) {
    if (!finite(flux) || !Kokkos::isfinite(speed) || !(speed > R{0}))
        return failure<R>(Status::NonfiniteFlux);
    return {flux, speed, Status::Success};
}

} // namespace krona::numerics::riemann::detail
