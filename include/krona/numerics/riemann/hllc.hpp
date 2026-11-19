#pragma once
 
#include <krona/numerics/riemann/hll.hpp>

namespace krona::numerics::riemann {

/// @brief HLLC resolves a contact wave between the two outer waves of HLL.
/// Across the contact, pressure and normal velocity agree while density and tangential velocity can jump.
/// @tparam Speeds Callable that takes both states and the unit normal and returns signed outer wave bounds.
template<class Speeds=DavisSpeeds>
struct Hllc {
    Speeds speeds{};    ///< Estimates the lower and upper outer wave speeds.
    bool fallback=true; ///< Try HLL if the intermediate states or their flux are unusable.

    /// @brief Computes the numerical flux per unit area and reports whether HLL was used.
    /// @tparam R Scalar type of the states and result.
    /// @param l Prepared Euler state on the left side.
    /// @param r Prepared Euler state on the right side.
    /// @param n Unit normal pointing from left to right.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l,State<R> r,Vector<R> n) const {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success) return detail::failure<R>(status);
        const auto wave=speeds(l,r,n);
        const R sl=wave.left,sr=wave.right,speed=Kokkos::max(Kokkos::abs(sl),Kokkos::abs(sr));
        if(!Kokkos::isfinite(sl)||!Kokkos::isfinite(sr)||!(sl<sr))return detail::failure<R>(Status::InvalidFan);

        // A fan moving entirely to one side of the face leaves the upwind state unchanged there.
        if(sl>=R{0}) return detail::finish(physical_flux(l,n),speed);
        if(sr<=R{0}) return detail::finish(physical_flux(r,n),speed);

        // Balance normal momentum across the outer waves to find the contact speed and pressure.
        const R ul=l.velocity.dot(n),ur=r.velocity.dot(n);
        const R dl=l.density*(sl-ul),dr=r.density*(sr-ur),denominator=dl-dr;
        if(!Kokkos::isfinite(denominator)||denominator==R{0})
            return detail::hll_fallback(l,r,n,speeds,fallback);
        const R middle=(r.pressure-l.pressure+dl*ul-dr*ur)/denominator;
        const R pl=l.pressure+dl*(middle-ul),pr=r.pressure+dr*(middle-ur);
        if(!Kokkos::isfinite(middle)||!(sl<middle&&middle<sr)||
           !Kokkos::isfinite(pl)||!(pl>R{0})||!Kokkos::isfinite(pr)||!(pr>R{0}))
            return detail::hll_fallback(l,r,n,speeds,fallback);

        // Build the two states next to the contact using the mass and energy jumps.
        // Set their normal velocity to the contact speed and keep each side's tangential velocity.
        const auto ql=conservative(l),qr=conservative(r);
        const R rl=dl/(sl-middle),rr=dr/(sr-middle);
        const flow::Conservative<R> star_l{rl,(l.velocity+n*(middle-ul))*rl,
            ((sl-ul)*ql.total_energy_density-l.pressure*ul+pl*middle)/(sl-middle)};
        const flow::Conservative<R> star_r{rr,(r.velocity+n*(middle-ur))*rr,
            ((sr-ur)*qr.total_energy_density-r.pressure*ur+pr*middle)/(sr-middle)};
        if(!(rl>R{0})||!(rr>R{0})||!detail::finite(star_l)||!detail::finite(star_r))
            return detail::hll_fallback(l,r,n,speeds,fallback);

        // The contact speed tells which intermediate state reaches the face.
        // Its flux is the upwind flux plus the jump carried by the outer wave.
        const auto flux=middle>=R{0}?physical_flux(l,n)+wave_flux(star_l-ql,sl):
                                           physical_flux(r,n)+wave_flux(star_r-qr,sr);
        auto result=detail::finish(flux,speed);
        if(!result.valid()) return detail::hll_fallback(l,r,n,speeds,fallback);
        return result;
    }
};

} // namespace krona::numerics::riemann
