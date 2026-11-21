#pragma once
 
#include <krona/numerics/riemann/detail/validation.hpp>

namespace krona::numerics::riemann {

namespace detail {

/// @brief Takes the larger of |u dot n| + a from the two states.
template<class R> KOKKOS_INLINE_FUNCTION R endpoint_speed(State<R> l,State<R> r,Vector<R> n)
{ return Kokkos::max(Kokkos::abs(l.velocity.dot(n))+l.sound_speed,Kokkos::abs(r.velocity.dot(n))+r.sound_speed); }

} // namespace detail

/// @brief Estimates the outer wave speeds from the sound speeds and normal velocities on both sides.
struct DavisSpeeds {
    /// @brief Takes the smallest u_n - a and the largest u_n + a across both sides.
    /// @tparam R Scalar type of the states and speeds.
    /// @param l Prepared state on the left side.
    /// @param r Prepared state on the right side.
    /// @param n Unit normal pointing from left to right.
    template<class R>
    KOKKOS_INLINE_FUNCTION WaveSpeeds<R> operator()(State<R> l, State<R> r, Vector<R> n) const {
        const R ul=l.velocity.dot(n),ur=r.velocity.dot(n);
        return {Kokkos::min(ul-l.sound_speed,ur-r.sound_speed),
                Kokkos::max(ul+l.sound_speed,ur+r.sound_speed)};
    }
};

/// @brief Rusanov (local Lax–Friedrichs) flux with one dissipation speed for all five equations.
struct Rusanov {
    /// @brief Computes the numerical flux per unit area.
    /// @tparam R Scalar type of the states and result.
    /// @param l Prepared state on the left side.
    /// @param r Prepared state on the right side.
    /// @param n Unit normal pointing from left to right.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l, State<R> r, Vector<R> n) const {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success) return detail::failure<R>(status);
        const R speed=detail::endpoint_speed(l,r,n);
        // Average the physical fluxes and use the conservative jump to damp the discontinuity.
        return detail::finish((physical_flux(l,n)+physical_flux(r,n))*R{0.5}-
            wave_flux(conservative(r)-conservative(l),R{0.5}*speed),speed);
    }
};

/// @brief Two-wave HLL solver with one intermediate region between the outer waves.
/// @tparam Speeds Callable that takes both states and the unit normal and returns signed wave bounds.
template<class Speeds=DavisSpeeds>
struct Hll {
    Speeds speeds{}; ///< Estimates the lower and upper wave speeds in left-to-right order.

    /// @brief Computes the numerical flux per unit area.
    /// @tparam R Scalar type of the states and result.
    /// @param l Prepared state on the left side.
    /// @param r Prepared state on the right side.
    /// @param n Unit normal pointing from left to right.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(State<R> l,State<R> r,Vector<R> n) const {
        const auto status=detail::input_status(l,r,n);
        if(status!=Status::Success) return detail::failure<R>(status);
        const auto wave=speeds(l,r,n);
        const R sl=wave.left,sr=wave.right,speed=Kokkos::max(Kokkos::abs(sl),Kokkos::abs(sr));
        if(!Kokkos::isfinite(sl)||!Kokkos::isfinite(sr)||!(sl<sr))return detail::failure<R>(Status::InvalidFan);

        // If both waves move the same way, the face sees the unchanged upwind state.
        if(sl>=R{0}) return detail::finish(physical_flux(l,n),speed);
        if(sr<=R{0}) return detail::finish(physical_flux(r,n),speed);

        // With waves moving both ways, blend the two fluxes and add the conservative jump term.
        // Apply the fan-width division before multiplying the wave speeds.
        const R inverse=R{1}/(sr-sl),wl=sr*inverse,wr=-sl*inverse;
        const auto flux=physical_flux(l,n)*wl+physical_flux(r,n)*wr+
            wave_flux(conservative(r)-conservative(l),sl*wl);
        return detail::finish(flux,speed);
    }
};

namespace detail {
/// @brief Tries HLL with the supplied speed policy and marks a successful retry as fallback.
/// Returns InvalidFan when fallback is disabled, or the HLL failure status if the retry fails.
template<class R,class Speeds>
KOKKOS_INLINE_FUNCTION Result<R> hll_fallback(State<R> l,State<R> r,Vector<R> n,const Speeds& speeds,bool enabled) {
    if(!enabled) return failure<R>(Status::InvalidFan);
    auto result=Hll<Speeds>{speeds}(l,r,n);
    if(result.valid()) result.status=Status::HllFallback;
    return result;
}
} // namespace detail

} // namespace krona::numerics::riemann
