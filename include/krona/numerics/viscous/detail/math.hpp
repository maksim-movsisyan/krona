#pragma once

#include <krona/numerics/viscous/types.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::numerics::viscous::detail {

template<class R, class V>
KOKKOS_INLINE_FUNCTION math::Vector3<R> components(const V& v) {
    return {static_cast<R>(v.x), static_cast<R>(v.y), static_cast<R>(v.z)};
}

template<class R>
KOKKOS_INLINE_FUNCTION bool finite(VelocityGradient<R> g) {
    return Kokkos::isfinite(g.du_dx) && Kokkos::isfinite(g.du_dy) && Kokkos::isfinite(g.du_dz) &&
           Kokkos::isfinite(g.dv_dx) && Kokkos::isfinite(g.dv_dy) && Kokkos::isfinite(g.dv_dz) &&
           Kokkos::isfinite(g.dw_dx) && Kokkos::isfinite(g.dw_dy) && Kokkos::isfinite(g.dw_dz);
}

template<class R, class N>
KOKKOS_INLINE_FUNCTION bool unit(const N& normal) {
    using Stored = std::remove_cvref_t<decltype(normal.x)>;
    static_assert(std::is_floating_point_v<Stored>);
    const auto n = components<R>(normal);
    const R norm = n.norm2();
    const R tolerance = R{128} * (Kokkos::epsilon_v<R> + static_cast<R>(Kokkos::epsilon_v<Stored>));
    return math::isfinite(n) && Kokkos::isfinite(norm) && Kokkos::abs(norm - R{1}) <= tolerance;
}

template<class R>
KOKKOS_INLINE_FUNCTION bool coefficient_valid(R x) {
    static_assert(std::is_floating_point_v<R>);
    return Kokkos::isfinite(x) && x >= R{0};
}

template<class R>
KOKKOS_INLINE_FUNCTION Status input_status(VelocityGradient<R> g, R mu, R bulk) {
    if (!coefficient_valid(mu) || !coefficient_valid(bulk)) return Status::InvalidCoefficient;
    return finite(g) ? Status::Success : Status::InvalidGradient;
}

template<class R>
KOKKOS_INLINE_FUNCTION R divergence(VelocityGradient<R> g) {
    return g.du_dx + g.dv_dy + g.dw_dz;
}

template<class R>
KOKKOS_INLINE_FUNCTION SymmetricStress<R> stress(VelocityGradient<R> g, R mu, R bulk) {
    const R theta = divergence(g);
    const R third = theta / R{3};
    const R isotropic = bulk * theta;
    return {R{2} * mu * (g.du_dx - third) + isotropic,
            R{2} * mu * (g.dv_dy - third) + isotropic,
            R{2} * mu * (g.dw_dz - third) + isotropic,
            mu * (g.du_dy + g.dv_dx), mu * (g.du_dz + g.dw_dx), mu * (g.dv_dz + g.dw_dy)};
}

template<class R>
KOKKOS_INLINE_FUNCTION Result<R> fail_scalar(Status status) {
    return {Kokkos::quiet_NaN_v<R>, status};
}

template<class R>
KOKKOS_INLINE_FUNCTION Result<Traction<R>> fail_traction(Status status) {
    const R n = Kokkos::quiet_NaN_v<R>;
    return {{n, n, n}, status};
}

template<class R>
KOKKOS_INLINE_FUNCTION Result<SymmetricStress<R>> fail_stress(Status status) {
    const R n = Kokkos::quiet_NaN_v<R>;
    return {{n, n, n, n, n, n}, status};
}

template<class R>
KOKKOS_INLINE_FUNCTION Result<SurfaceTransport<R>> fail_surface(Status status) {
    const R n = Kokkos::quiet_NaN_v<R>;
    return {{{n, n, n}, n}, status};
}

} // namespace krona::numerics::viscous::detail
