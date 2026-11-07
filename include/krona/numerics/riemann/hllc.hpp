#pragma once
 
#include <krona/numerics/riemann/hll.hpp>

namespace krona::numerics::riemann {

/// @brief Three-wave HLLC for prepared single-phase Euler states. Fallback is reported, never hidden.
template<class Speeds=DavisSpeeds>
struct Hllc {
    Speeds speeds{};
    bool fallback=true;
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
        const R ul=l.velocity.dot(n),ur=r.velocity.dot(n);
        const R dl=l.density*(sl-ul),dr=r.density*(sr-ur),denominator=dl-dr;
        if(!Kokkos::isfinite(denominator)||denominator==R{0})
            return detail::hll_fallback(l,r,n,speeds,fallback);
        const R middle=(r.pressure-l.pressure+dl*ul-dr*ur)/denominator;
        const R pl=l.pressure+dl*(middle-ul),pr=r.pressure+dr*(middle-ur);
        if(!Kokkos::isfinite(middle)||!(sl<middle&&middle<sr)||
           !Kokkos::isfinite(pl)||!(pl>R{0})||!Kokkos::isfinite(pr)||!(pr>R{0}))
            return detail::hll_fallback(l,r,n,speeds,fallback);
        const auto ql=conservative(l),qr=conservative(r);
        const R rl=dl/(sl-middle),rr=dr/(sr-middle);
        const Conservative<R> star_l{rl,(l.velocity+n*(middle-ul))*rl,
            ((sl-ul)*ql.energy-l.pressure*ul+pl*middle)/(sl-middle)};
        const Conservative<R> star_r{rr,(r.velocity+n*(middle-ur))*rr,
            ((sr-ur)*qr.energy-r.pressure*ur+pr*middle)/(sr-middle)};
        if(!(rl>R{0})||!(rr>R{0})||!detail::finite(star_l)||!detail::finite(star_r))
            return detail::hll_fallback(l,r,n,speeds,fallback);
        const auto flux=middle>=R{0}?physical_flux(l,n)+(star_l-ql)*sl:
                                           physical_flux(r,n)+(star_r-qr)*sr;
        auto result=detail::finish(flux,speed);
        if(!result.valid())return detail::hll_fallback(l,r,n,speeds,fallback);
        return result;
    }
};

} // namespace krona::numerics::riemann
