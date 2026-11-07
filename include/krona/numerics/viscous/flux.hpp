#pragma once

#include <krona/numerics/viscous/constitutive.hpp>

namespace krona::numerics::viscous {
namespace detail {

/// @brief Caller-selected five-component value type; no dependency on a Riemann solver.
template<class Flux,class R>
KOKKOS_INLINE_FUNCTION Flux make_flux(Traction<R> momentum,R energy,R mass=R{0})
{
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(Flux{}.mass)>,R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.momentum.x)>,R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.momentum.y)>,R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.momentum.z)>,R> &&
                  std::is_same_v<std::remove_cvref_t<decltype(Flux{}.energy)>,R>,
                  "Viscous flux container and local arithmetic must use the same scalar type");
    return Flux{mass,{momentum.x,momentum.y,momentum.z},energy};
}

template<class Flux,class R>
KOKKOS_INLINE_FUNCTION Result<Flux> fail_flux(Status status)
{
    const R n=Kokkos::quiet_NaN_v<R>;
    return {make_flux<Flux>(Traction<R>{n,n,n},n,n),status};
}

} // namespace detail

/// @brief Outward diffusive flux: (0,-tau*n,q*n-u*(tau*n)). Add to outward inviscid flux.
template<class Flux,class R,class V>
KOKKOS_INLINE_FUNCTION Result<Flux> outward_flux(Result<SurfaceTransport<R>> surface,const V& velocity)
{
    if(!surface.valid())return detail::fail_flux<Flux,R>(surface.status);
    const auto u=detail::components<R>(velocity);
    if(!detail::finite(u))return detail::fail_flux<Flux,R>(Status::InvalidVelocity);
    const auto t=surface.value.traction;
    if(!detail::finite(t)||!Kokkos::isfinite(surface.value.outward_heat_flux))
        return detail::fail_flux<Flux,R>(Status::NonfiniteResult);
    const R energy=surface.value.outward_heat_flux-detail::dot(u,t);
    if(!Kokkos::isfinite(energy))return detail::fail_flux<Flux,R>(Status::NonfiniteResult);
    return {detail::make_flux<Flux>(Traction<R>{-t.x,-t.y,-t.z},energy)};
}

/// @brief Opposite convention: dU/dt+div(F_inviscid)=div(F_viscous_rhs).
template<class Flux,class R,class V>
KOKKOS_INLINE_FUNCTION Result<Flux> rhs_flux(Result<SurfaceTransport<R>> surface,const V& velocity)
{
    auto result=outward_flux<Flux>(surface,velocity);
    if(result.valid()){
        result.value.momentum.x=-result.value.momentum.x;
        result.value.momentum.y=-result.value.momentum.y;
        result.value.momentum.z=-result.value.momentum.z;
        result.value.energy=-result.value.energy;
    }
    return result;
}

/// @brief Uses the supplied unit normal's projection and a separate positive face area.
template<class Flux,class R,class V>
KOKKOS_INLINE_FUNCTION Result<Flux> integrated_outward_flux(Result<SurfaceTransport<R>> surface,
    const V& velocity,std::type_identity_t<R> area)
{
    if(!Kokkos::isfinite(area)||!(area>R{0}))return detail::fail_flux<Flux,R>(Status::InvalidArea);
    auto result=outward_flux<Flux>(surface,velocity);
    if(result.valid()){
        result.value.momentum.x*=area;
        result.value.momentum.y*=area;
        result.value.momentum.z*=area;
        result.value.energy*=area;
        if(!Kokkos::isfinite(result.value.momentum.x)||!Kokkos::isfinite(result.value.momentum.y)||
           !Kokkos::isfinite(result.value.momentum.z)||!Kokkos::isfinite(result.value.energy))
            return detail::fail_flux<Flux,R>(Status::NonfiniteResult);
    }
    return result;
}

} // namespace krona::numerics::viscous
