#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/thermodynamics/ideal_gas.hpp>

namespace krona::boundary::physical {

struct NoCharacteristics {};

/// Calorically perfect-gas characteristic closure. A different EOS supplies its own policy.
/// The invariant is local 1D normal compatibility, not a multidimensional nonreflecting BC.
struct PerfectGasCharacteristics {
    template<class Real>
    void validate(const thermodynamics::IdealGas<Real>&) const {}

    template<class EOS,class Real>
    KOKKOS_INLINE_FUNCTION Primitive<Real,true> outlet(const EOS& eos,
        const FaceContext<Real>& c, Primitive<Real,true> inside, Real pressure) const
    {
        const auto qi = eos.from_pT(inside.pressure,inside.temperature);
        const Real gm = static_cast<Real>(eos.gamma) - Real{1};
        const auto n = normal(c);
        const Real un = inside.velocity.dot(n);
        const Real invariant = un + Real{2}*qi.sound_speed/gm;
        auto qb = eos.from_ps(pressure,qi.entropy);
        Real ub = invariant - Real{2}*qb.sound_speed/gm;
        if (ub > qb.sound_speed) {
            // A subsonic interior cannot communicate an arbitrarily low back pressure.
            const Real sound = invariant*gm/(gm + Real{2});
            const Real T = sound*sound/(static_cast<Real>(eos.gamma)*static_cast<Real>(eos.gas_constant));
            const Real p = inside.pressure*Kokkos::pow(T/inside.temperature,
                static_cast<Real>(eos.gamma)/gm);
            qb = eos.from_pT(p,T);
            ub = qb.sound_speed;
        }
        return {qb.pressure, inside.velocity + n*(ub-un), qb.temperature};
    }

    template<class EOS,class Real>
    KOKKOS_INLINE_FUNCTION Primitive<Real,true> inlet(const EOS& eos,
        const FaceContext<Real>& c, Primitive<Real,true> inside, TotalConditionsInlet<Real> specified) const
    {
        const auto n = normal(c);
        const Real mu = specified.direction.dot(n);
        if (!(mu < Real{0})) Kokkos::abort("Physical boundary: inlet direction must point into the domain");
        const auto qi = eos.from_pT(inside.pressure,inside.temperature);
        const Real un = inside.velocity.dot(n);
        if (!(un > -qi.sound_speed))
            Kokkos::abort("Physical boundary: total conditions do not specify a supersonic inflow state");
        const auto stagnation = eos.from_pT(specified.total_pressure,specified.total_temperature);
        const Real gm = static_cast<Real>(eos.gamma) - Real{1};
        const Real invariant = un + Real{2}*qi.sound_speed/gm;
        const Real resting_invariant = Real{2}*stagnation.sound_speed/gm;
        if (resting_invariant < invariant)
            Kokkos::abort("Physical boundary: total inlet data admit no inward characteristic solution");
        const Real sonic_speed = Kokkos::sqrt(stagnation.sound_speed*stagnation.sound_speed/(mu*mu+gm/Real{2}));
        const Real sonic_sound = -mu*sonic_speed;
        Real speed = 0;
        if (mu*sonic_speed+Real{2}*sonic_sound/gm > invariant) {
            if (!specified.allow_choking)
                Kokkos::abort("Physical boundary: total inlet choked; explicitly allow_choking or supply a supersonic state");
            speed = sonic_speed;
        } else {
            const Real a = mu*mu + Real{2}/gm;
            const Real b = -Real{2}*invariant*mu;
            const Real d = (invariant-resting_invariant)*(invariant+resting_invariant);
            const Real discriminant = Kokkos::fmax(Real{0},b*b-Real{4}*a*d);
            const Real root = Kokkos::sqrt(discriminant);
            speed = b > Real{0} ? -Real{2}*d/(b+root) : (-b+root)/(Real{2}*a);
            speed = Kokkos::fmin(sonic_speed,Kokkos::fmax(Real{0},speed));
        }
        const auto qb = eos.from_hs(stagnation.enthalpy-speed*speed/Real{2},stagnation.entropy);
        return {qb.pressure,specified.direction*speed,qb.temperature};
    }
};

} // namespace krona::boundary::physical
