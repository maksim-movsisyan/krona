#pragma once
#include <krona/boundary/physical/state.hpp>
#include <krona/numerics/riemann/eos.hpp>

namespace krona::transport::euler {

namespace detail {

template<class R>
KOKKOS_INLINE_FUNCTION bool primitive_valid(boundary::physical::Primitive<R,true> q) {
    return Kokkos::isfinite(q.pressure)&&q.pressure>R{0}&&Kokkos::isfinite(q.temperature)&&q.temperature>R{0}&&
        Kokkos::isfinite(q.velocity.x)&&Kokkos::isfinite(q.velocity.y)&&Kokkos::isfinite(q.velocity.z);
}
template<class R>
KOKKOS_INLINE_FUNCTION bool state_valid(numerics::riemann::State<R> q) {
    const auto u=numerics::riemann::conservative(q);
    return Kokkos::isfinite(q.density)&&q.density>R{0}&&Kokkos::isfinite(q.pressure)&&q.pressure>R{0}&&
        Kokkos::isfinite(q.internal_energy)&&Kokkos::isfinite(q.sound_speed)&&q.sound_speed>R{0}&&
        Kokkos::isfinite(u.density)&&Kokkos::isfinite(u.momentum.x)&&Kokkos::isfinite(u.momentum.y)&&
        Kokkos::isfinite(u.momentum.z)&&Kokkos::isfinite(u.total_energy_density);
}
// Check pressure, temperature and velocity before calling a potentially expensive EOS.
template<class R, class StateLaw>
KOKKOS_INLINE_FUNCTION numerics::riemann::State<R> prepare_state(const StateLaw& law,
                                                                 boundary::physical::Primitive<R, true> primitive) {
    if (!primitive_valid(primitive)) return {};
    return law(primitive);
}

template<class R,class Pack>
KOKKOS_INLINE_FUNCTION auto primitive(const Pack& values,std::size_t cell) {
    return boundary::physical::Primitive<R,true>{static_cast<R>(values(cell,0)),
        {static_cast<R>(values(cell,1)),static_cast<R>(values(cell,2)),static_cast<R>(values(cell,3))},static_cast<R>(values(cell,4))};
}
template<class R>
KOKKOS_INLINE_FUNCTION auto primitive(const Kokkos::Array<R,5>& a)
{ return boundary::physical::Primitive<R,true>{a[0],{a[1],a[2],a[3]},a[4]}; }

} // namespace detail

/// @brief Default domain guard for an EOS that accepts any finite positive pressure and temperature.
/// The caller checks those basic conditions before using the guard.
struct PositivePT {
    template<class EOS,class Primitive>
    KOKKOS_INLINE_FUNCTION bool operator()(const EOS&,const Primitive&) const { return true; }
};

/// @brief Prepares a Riemann state from pressure, velocity and temperature (StateLaw above).
/// @tparam EOS Equation of state used to recover density, specific internal energy and sound speed.
/// @tparam Domain Callable that decides whether to pass a primitive state to this EOS.
/// Supply a guard if the EOS has a narrower domain than positive pressure and temperature.
template<class EOS,class Domain=PositivePT>
struct Thermodynamics {
    EOS eos;         ///< Equation of state copied into the kernel.
    Domain domain{}; ///< Optional extra check before calling the EOS.
    /// @brief Checks EOS and domain configuration on the host.
    void validate() const {
        eos.validate();
        if constexpr(requires{domain.validate();})domain.validate();
    }
    /// @brief Converts one primitive state, returning an invalid default state when the guard fails.
    /// @tparam R Scalar type of the input and resulting state.
    /// @param q Pressure, Cartesian velocity and absolute temperature.
    template<class R>
    KOKKOS_INLINE_FUNCTION numerics::riemann::State<R> operator()(boundary::physical::Primitive<R,true> q) const {
        if(!detail::primitive_valid(q) || !domain(eos,q)) return {};
        return numerics::riemann::from_pT(eos, q.pressure, q.temperature, q.velocity);
    }
};

template<class EOS>Thermodynamics(EOS)->Thermodynamics<EOS>;
template<class EOS,class D>Thermodynamics(EOS,D)->Thermodynamics<EOS,D>;
} // namespace krona::transport::euler
