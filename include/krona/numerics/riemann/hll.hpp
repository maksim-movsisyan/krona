#pragma once
 
#include <krona/numerics/riemann/types.hpp>

namespace krona::numerics::riemann {

/// @brief Endpoint acoustic bounds (Davis); no EOS-specific Roe average is assumed.
struct DavisSpeeds {
    template<class R>
    KOKKOS_INLINE_FUNCTION WaveSpeeds<R> operator()(State<R> l,State<R> r,Vector<R> n) const
    {
        const R ul=l.velocity.dot(n),ur=r.velocity.dot(n);
        return {Kokkos::min(ul-l.sound_speed,ur-r.sound_speed),
                Kokkos::max(ul+l.sound_speed,ur+r.sound_speed)};
    }
};

struct Rusanov {
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l,State<R> r,Vector<R> n) const
    {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success)return detail::failure<R>(status);
        const R speed=detail::endpoint_speed(l,r,n);
        return detail::finish((physical_flux(l,n)+physical_flux(r,n))*R{0.5}-
            (conservative(r)-conservative(l))*(R{0.5}*speed),speed);
    }
};

template<class Speeds=DavisSpeeds>
struct Hll {
    Speeds speeds{};
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l,State<R> r,Vector<R> n) const
    {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success)return detail::failure<R>(status);
        const auto wave=speeds(l,r,n);
        const R sl=wave.left,sr=wave.right,speed=Kokkos::max(Kokkos::abs(sl),Kokkos::abs(sr));
        if(!Kokkos::isfinite(sl)||!Kokkos::isfinite(sr)||!(sl<sr))return detail::failure<R>(Status::InvalidFan);
        if(sl>=R{0})return detail::finish(physical_flux(l,n),speed);
        if(sr<=R{0})return detail::finish(physical_flux(r,n),speed);
        // Convex flux weights avoid forming sl*sr before dividing by the fan width.
        const R inverse=R{1}/(sr-sl),wl=sr*inverse,wr=-sl*inverse;
        const auto flux=physical_flux(l,n)*wl+physical_flux(r,n)*wr+
            (conservative(r)-conservative(l))*(sl*wl);
        return detail::finish(flux,speed);
    }
};

namespace detail {
template<class R,class Speeds>
KOKKOS_INLINE_FUNCTION Result<R> hll_fallback(State<R> l,State<R> r,Vector<R> n,const Speeds& speeds,bool enabled)
{
    if(!enabled)return failure<R>(Status::InvalidFan);
    auto result=Hll<Speeds>{speeds}(l,r,n);
    if(result.valid())result.status=Status::HllFallback;
    return result;
}
} // namespace detail

} // namespace krona::numerics::riemann
