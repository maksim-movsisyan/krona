#pragma once
#include <krona/boundary/physical/state.hpp>
#include <krona/numerics/riemann/eos.hpp>

namespace krona::transport::euler {
namespace detail {
template<class R>
KOKKOS_INLINE_FUNCTION bool primitive_valid(boundary::physical::Primitive<R,true> q)
{
    return Kokkos::isfinite(q.pressure)&&q.pressure>R{0}&&Kokkos::isfinite(q.temperature)&&q.temperature>R{0}&&
        Kokkos::isfinite(q.velocity.x)&&Kokkos::isfinite(q.velocity.y)&&Kokkos::isfinite(q.velocity.z);
}
template<class R>
KOKKOS_INLINE_FUNCTION bool state_valid(numerics::riemann::State<R> q)
{
    const auto u=numerics::riemann::conservative(q);
    return Kokkos::isfinite(q.density)&&q.density>R{0}&&Kokkos::isfinite(q.pressure)&&q.pressure>R{0}&&
        Kokkos::isfinite(q.internal_energy)&&Kokkos::isfinite(q.sound_speed)&&q.sound_speed>R{0}&&
        Kokkos::isfinite(u.mass)&&Kokkos::isfinite(u.momentum.x)&&Kokkos::isfinite(u.momentum.y)&&
        Kokkos::isfinite(u.momentum.z)&&Kokkos::isfinite(u.energy);
}
template<class R,class Pack>
KOKKOS_INLINE_FUNCTION auto primitive(const Pack& values,std::size_t cell)
{
    return boundary::physical::Primitive<R,true>{static_cast<R>(values(cell,0)),
        {static_cast<R>(values(cell,1)),static_cast<R>(values(cell,2)),static_cast<R>(values(cell,3))},static_cast<R>(values(cell,4))};
}
template<class R>
KOKKOS_INLINE_FUNCTION auto primitive(const Kokkos::Array<R,5>& a)
{ return boundary::physical::Primitive<R,true>{a[0],{a[1],a[2],a[3]},a[4]}; }
} // namespace detail

/// @brief Additional model-domain guard; finite positive p,T are always checked before this guard.
struct PositivePT {
    template<class EOS,class Primitive>
    KOKKOS_INLINE_FUNCTION bool operator()(const EOS&,const Primitive&) const { return true; }
};

/// @brief EOS preparation policy. Generic EOS keeps its error policy; Domain must cover its admissible p,T region.
template<class EOS,class Domain=PositivePT>
struct Thermodynamics {
    EOS eos;
    Domain domain{};
    void validate() const
    {
        eos.validate();
        if constexpr(requires{domain.validate();})domain.validate();
    }
    template<class R>
    KOKKOS_INLINE_FUNCTION numerics::riemann::State<R> operator()(boundary::physical::Primitive<R,true> q) const
    {
        if(!detail::primitive_valid(q)||!domain(eos,q))return {};
        return numerics::riemann::from_pT(eos,q.pressure,q.temperature,
            numerics::riemann::Vector<R>{q.velocity.x,q.velocity.y,q.velocity.z});
    }
};
template<class EOS>Thermodynamics(EOS)->Thermodynamics<EOS>;
template<class EOS,class D>Thermodynamics(EOS,D)->Thermodynamics<EOS,D>;
} // namespace krona::transport::euler
