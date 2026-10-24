#pragma once 
#include <krona/numerics/riemann/hll.hpp>
#include <cmath>
#include <stdexcept>

namespace krona::numerics::riemann {

/// @brief Classical five-wave Roe flux for one calorically perfect gas.
/// Acoustic entropy smoothing and an explicitly reported HLL fallback are enabled by default.
template<class Parameter=ComputeReal>
struct Roe {
    Parameter gamma;
    bool entropy_fix=true, fallback=true;
    explicit Roe(Parameter g,bool entropy=true,bool use_fallback=true)
        : gamma(g),entropy_fix(entropy),fallback(use_fallback)
    {
        if(!std::isfinite(gamma)||!(gamma>Parameter{1}))
            throw std::invalid_argument("Roe: gamma must be finite and greater than one");
    }
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l,State<R> r,Vector<R> n) const
    {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success)return detail::failure<R>(status);
        const R g=static_cast<R>(gamma);
        if(!compatible(l,g)||!compatible(r,g))return detail::failure<R>(Status::IncompatibleModel);
        const R rl=Kokkos::sqrt(l.density),rr=Kokkos::sqrt(r.density),sum=rl+rr;
        const R wl=rl/sum,wr=rr/sum,rho=rl*rr;
        const auto u=l.velocity*wl+r.velocity*wr;
        const R hl=l.internal_energy+R{0.5}*l.velocity.norm2()+l.pressure/l.density;
        const R hr=r.internal_energy+R{0.5}*r.velocity.norm2()+r.pressure/r.density;
        const R h=wl*hl+wr*hr,un=u.dot(n),kinetic=R{0.5}*u.norm2();
        const R a2=(g-R{1})*(h-kinetic);
        if(!Kokkos::isfinite(a2)||!(a2>R{0}))return fail(l,r,n);
        const R a=Kokkos::sqrt(a2),dp=r.pressure-l.pressure;
        const auto du=r.velocity-l.velocity;
        const R dun=du.dot(n),am=(dp-rho*a*dun)/(R{2}*a2),ap=(dp+rho*a*dun)/(R{2}*a2);
        const R ac=r.density-l.density-dp/a2;
        const Conservative<R> minus{am,(u-n*a)*am,(h-a*un)*am};
        const Conservative<R> plus{ap,(u+n*a)*ap,(h+a*un)*ap};
        // Reject nonphysical Roe intermediate states before accepting the linearized fan.
        if(!positive(conservative(l)+minus)||!positive(conservative(r)-plus))return fail(l,r,n);
        const R ul=l.velocity.dot(n),ur=r.velocity.dot(n);
        const R lm=acoustic(un-a,ul-l.sound_speed,ur-r.sound_speed);
        const R lp=acoustic(un+a,ul+l.sound_speed,ur+r.sound_speed),lc=Kokkos::abs(un);
        const Conservative<R> middle{ac,u*ac+(du-n*dun)*rho,
                                      kinetic*ac+rho*u.dot(du-n*dun)};
        const auto diss=minus*lm+middle*lc+plus*lp;
        auto result=detail::finish((physical_flux(l,n)+physical_flux(r,n)-diss)*R{0.5},
            Kokkos::max(detail::endpoint_speed(l,r,n),Kokkos::max(lm,lp)));
        return result.valid()?result:fail(l,r,n);
    }
private:
    template<class R> KOKKOS_INLINE_FUNCTION static bool close(R x,R y)
    {
        return Kokkos::isfinite(x)&&Kokkos::isfinite(y)&&Kokkos::abs(x-y)<=
            R{128}*(Kokkos::epsilon_v<R>+static_cast<R>(Kokkos::epsilon_v<Parameter>))*
            Kokkos::max(Kokkos::abs(x),Kokkos::abs(y));
    }
    template<class R> KOKKOS_INLINE_FUNCTION static bool compatible(State<R> s,R g)
    {
        return Kokkos::isfinite(g)&&g>R{1}&&close(s.pressure,(g-R{1})*s.density*s.internal_energy)&&
            close(s.sound_speed*s.sound_speed,g*s.pressure/s.density);
    }
    template<class R> KOKKOS_INLINE_FUNCTION static bool positive(Conservative<R> q)
    { return detail::finite(q)&&q.mass>R{0}&&q.energy-R{0.5}*q.momentum.norm2()/q.mass>R{0}; }
    template<class R> KOKKOS_INLINE_FUNCTION R acoustic(R lambda,R left,R right) const
    {
        const R value=Kokkos::abs(lambda);
        if(!entropy_fix)return value;
        const R width=Kokkos::max(R{0},Kokkos::max(lambda-left,right-lambda));
        return value<width?R{0.5}*(value*(value/width)+width):value;
    }
    template<class R> KOKKOS_INLINE_FUNCTION Result<R> fail(State<R> l,State<R> r,Vector<R> n) const
    { return detail::hll_fallback(l,r,n,DavisSpeeds{},fallback); }
};

} // namespace krona::numerics::riemann
