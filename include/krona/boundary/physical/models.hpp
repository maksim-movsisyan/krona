#pragma once

#include <krona/boundary/physical/characteristics.hpp>
#include <krona/thermodynamics/constant_density.hpp>
#include <type_traits>

namespace krona::boundary::physical {

template<class EOS,class Characteristics = NoCharacteristics>
struct Compressible {
    static constexpr bool compressible = true, thermal = true;
    static constexpr bool has_characteristics = !std::is_same_v<Characteristics,NoCharacteristics>;
    EOS eos;
    Characteristics characteristics;

    void validate() const
    {
        eos.validate();
        if constexpr (has_characteristics) characteristics.validate(eos);
    }
    template<class Real>
    KOKKOS_INLINE_FUNCTION auto thermodynamic(Primitive<Real,true> s) const
    { return eos.from_pT(s.pressure,s.temperature); }

    template<class Real>
    KOKKOS_INLINE_FUNCTION Conservative<Real> conservative(Primitive<Real,true> s) const
    {
        const auto q = thermodynamic(s);
        return {q.density,s.velocity*q.density,q.density*(q.internal_energy+s.velocity.norm2()/Real{2})};
    }
    template<class Real>
    KOKKOS_INLINE_FUNCTION Primitive<Real,true> primitive(Conservative<Real> s) const
    {
        if (!(s.density > Real{0})) Kokkos::abort("Physical boundary: nonpositive conservative density");
        const auto u = s.momentum*(Real{1}/s.density);
        const auto q = eos.from_rhoe(s.density,s.total_energy/s.density-u.norm2()/Real{2});
        return {q.pressure,u,q.temperature};
    }
};
template<class EOS,class C> Compressible(EOS,C)->Compressible<EOS,C>;
template<class EOS> Compressible(EOS)->Compressible<EOS,NoCharacteristics>;

template<class Material,bool Thermal = false>
struct Incompressible {
    static constexpr bool compressible = false, thermal = Thermal, has_characteristics = false;
    Material material;
    void validate() const { material.validate(); }
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real density(Primitive<Real,Thermal> s) const
    {
        if constexpr (Thermal) return material.density(s.temperature);
        else return material.density(Real{0});
    }
};

} // namespace krona::boundary::physical
