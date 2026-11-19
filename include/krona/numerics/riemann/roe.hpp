#pragma once 
#include <krona/numerics/riemann/hll.hpp>
#include <krona/mpi/error.hpp>
#include <cmath>

namespace krona::numerics::riemann {

/// @brief Five-wave Roe solver for a gas with constant gamma and p = (gamma - 1) rho e.
/// The waves are two acoustic waves, one contact wave and two tangential shear waves.
/// @tparam Parameter Scalar type used to store gamma. Flux arithmetic uses the state scalar type.
template<class Parameter=ComputeReal>
struct Roe {
    Parameter gamma;        ///< Ratio of specific heats shared by both states.
    bool entropy_fix=true,  ///< Smooth the acoustic dissipation near a sonic wave.
         fallback=true;     ///< Try HLL when Roe cannot construct a usable fan or flux.

    /// @brief Sets up the solver on the host and validates the ratio of specific heats.
    /// @param g Finite ratio of specific heats greater than one.
    /// @param entropy Whether to enable the acoustic entropy fix.
    /// @param use_fallback Whether to try HLL with Davis bounds if Roe's fan or flux fails.
    explicit Roe(Parameter g,bool entropy=true,bool use_fallback=true)
        : gamma(g),entropy_fix(entropy),fallback(use_fallback) {
        mpi::require(std::isfinite(gamma) && gamma>Parameter{1},
                     "Roe: gamma must be finite and greater than one");
    }

    /// @brief Computes the numerical flux per unit area and reports whether HLL was used.
    /// @tparam R Scalar type of the states and result.
    /// @param l Prepared Euler state on the left side, consistent with gamma.
    /// @param r Prepared Euler state on the right side, consistent with gamma.
    /// @param n Unit normal pointing from left to right.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l,State<R> r,Vector<R> n) const {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success) return detail::failure<R>(status);
        const R g=static_cast<R>(gamma);
        if(!compatible(l,g)||!compatible(r,g)) return detail::failure<R>(Status::IncompatibleModel);

        // Roe weights the velocity and total enthalpy by the square roots of the densities.
        const R rl=Kokkos::sqrt(l.density),rr=Kokkos::sqrt(r.density),sum=rl+rr;
        const R wl=rl/sum,wr=rr/sum,rho=rl*rr;
        const auto u=l.velocity*wl+r.velocity*wr;
        const R hl=l.internal_energy+R{0.5}*l.velocity.norm2()+l.pressure/l.density;
        const R hr=r.internal_energy+R{0.5}*r.velocity.norm2()+r.pressure/r.density;
        const R h=wl*hl+wr*hr,un=u.dot(n),kinetic=R{0.5}*u.norm2();
        const R a2=(g-R{1})*(h-kinetic);
        if(!Kokkos::isfinite(a2)||!(a2>R{0})) return fail(l,r,n);

        // Split the jump into acoustic contributions from pressure and normal velocity,
        // a contact contribution from density, and shear contributions from tangential velocity.
        const R a=Kokkos::sqrt(a2),dp=r.pressure-l.pressure;
        const auto du=r.velocity-l.velocity;
        const R dun=du.dot(n),am=(dp-rho*a*dun)/(R{2}*a2),ap=(dp+rho*a*dun)/(R{2}*a2);
        const R ac=r.density-l.density-dp/a2;
        const flow::Conservative<R> minus{am,(u-n*a)*am,(h-a*un)*am};
        const flow::Conservative<R> plus{ap,(u+n*a)*ap,(h+a*un)*ap};

        // Check density and internal energy in the two states just inside the acoustic waves.
        if(!positive(conservative(l)+minus)||!positive(conservative(r)-plus)) return fail(l,r,n);

        // The contact and both shear waves travel at the same speed, so group them in the flux.
        const R ul=l.velocity.dot(n),ur=r.velocity.dot(n);
        const R lm=acoustic(un-a,ul-l.sound_speed,ur-r.sound_speed);
        const R lp=acoustic(un+a,ul+l.sound_speed,ur+r.sound_speed),lc=Kokkos::abs(un);
        const flow::Conservative<R> middle{ac,u*ac+(du-n*dun)*rho,
                                           kinetic*ac+rho*u.dot(du-n*dun)};
        const auto diss=wave_flux(minus,lm)+wave_flux(middle,lc)+wave_flux(plus,lp);
        auto result=detail::finish((physical_flux(l,n)+physical_flux(r,n)-diss)*R{0.5},
            Kokkos::max(detail::endpoint_speed(l,r,n),Kokkos::max(lm,lp)));
        return result.valid()?result:fail(l,r,n);
    }
private:
    /// @brief Compares values using a relative tolerance that accounts for both scalar types.
    template<class R> KOKKOS_INLINE_FUNCTION static bool close(R x,R y) {
        return Kokkos::isfinite(x)&&Kokkos::isfinite(y)&&Kokkos::abs(x-y)<=
            R{128}*(Kokkos::epsilon_v<R>+static_cast<R>(Kokkos::epsilon_v<Parameter>))*
            Kokkos::max(Kokkos::abs(x),Kokkos::abs(y));
    }
    /// @brief Checks the pressure and sound speed relations needed by the Roe linearization.
    template<class R> KOKKOS_INLINE_FUNCTION static bool compatible(State<R> s,R g) {
        return Kokkos::isfinite(g)&&g>R{1}&&close(s.pressure,(g-R{1})*s.density*s.internal_energy)&&
            close(s.sound_speed*s.sound_speed,g*s.pressure/s.density);
    }
    /// @brief Checks finite densities, positive mass density and positive internal energy density.
    template<class R> KOKKOS_INLINE_FUNCTION static bool positive(flow::Conservative<R> q)
    { return detail::finite(q)&&q.density>R{0}&&q.total_energy_density-R{0.5}*q.momentum.norm2()/q.density>R{0}; }
    template<class R> KOKKOS_INLINE_FUNCTION R acoustic(R lambda,R left,R right) const {
        const R value=Kokkos::abs(lambda);
        if(!entropy_fix) return value;
        // Use a smooth quadratic near zero when the acoustic wave spreads across the fan.
        const R width=Kokkos::max(R{0},Kokkos::max(lambda-left,right-lambda));
        return value<width?R{0.5}*(value*(value/width)+width):value;
    }
    template<class R> KOKKOS_INLINE_FUNCTION Result<R> fail(State<R> l,State<R> r,Vector<R> n) const
    { return detail::hll_fallback(l,r,n,DavisSpeeds{},fallback); }
};

} // namespace krona::numerics::riemann
