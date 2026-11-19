#pragma once

#include <krona/boundary/physical/characteristics.hpp>
#include <krona/flow/conservative.hpp>
#include <krona/thermodynamics/constant_density.hpp>
#include <type_traits>

namespace krona::boundary::physical {

/// @brief Supplies EOS results and the characteristic closure for compressible boundaries.
/// @tparam EOS Equation of state used to convert pressure and temperature or density and internal energy.
/// @tparam Characteristics Policy for characteristic inlets and pressure outlets.
template<class EOS,class Characteristics = NoCharacteristics>
struct Compressible {
    static constexpr bool compressible = true, thermal = true;
    static constexpr bool has_characteristics = !std::is_same_v<Characteristics,NoCharacteristics>;
    EOS eos;
    Characteristics characteristics;

    /// @brief Checks the EOS parameters and its compatibility with the characteristic policy on the host.
    void validate() const {
        eos.validate();
        if constexpr (has_characteristics) characteristics.validate(eos);
    }
    /// @brief Evaluates the thermodynamic state from pressure and temperature.
    /// @tparam Real Arithmetic type of the state.
    /// @param s Primitive state whose pressure and temperature are passed to the EOS.
    template<class Real>
    KOKKOS_INLINE_FUNCTION auto thermodynamic(Primitive<Real,true> s) const
    { return eos.from_pT(s.pressure,s.temperature); }

    /// @brief Forms mass, momentum and total-energy densities from a primitive state.
    /// @tparam Real Arithmetic and storage type of one state.
    /// @param s Primitive state to convert.
    template<class Real>
    KOKKOS_INLINE_FUNCTION flow::Conservative<Real> conservative(Primitive<Real,true> s) const {
        const auto q = thermodynamic(s);
        return flow::to_conservative<Real>(static_cast<Real>(q.density), s.velocity, static_cast<Real>(q.internal_energy));
    }

    /// @brief Recovers pressure and temperature using the specific internal energy and density.
    /// @tparam Real Arithmetic and storage type of one state.
    /// @param s Conservative densities to convert.
    template<class Real>
    KOKKOS_INLINE_FUNCTION Primitive<Real,true> primitive(flow::Conservative<Real> s) const {
        flow::InternalEnergyState<Real> recovered;
        if (!flow::try_split_conservative(s, recovered))
            Kokkos::abort("Physical boundary: invalid conservative state");
        const auto q = eos.from_rhoe(recovered.density, recovered.internal_energy);
        return {q.pressure, recovered.velocity, q.temperature};
    }
};
template<class EOS,class C> Compressible(EOS,C)->Compressible<EOS,C>;
template<class EOS> Compressible(EOS)->Compressible<EOS,NoCharacteristics>;

/// @brief Supplies the density of an incompressible fluid with an optional thermal equation.
/// @tparam Material Density law with a host validation method.
/// @tparam Thermal True to supply the actual temperature to the density law.
template<class Material,bool Thermal = false>
struct Incompressible {
    static constexpr bool compressible = false, thermal = Thermal, has_characteristics = false;
    Material material;
    /// @brief Checks the parameters of the material law on the host.
    void validate() const { material.validate(); }

    /// @brief Evaluates the material density for one primitive state.
    /// @tparam Real Arithmetic type of the state and returned density.
    /// @param s State with the temperature when the model is thermal.
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real density(Primitive<Real,Thermal> s) const {
        if constexpr (Thermal) return material.density(s.temperature);
        else return material.density(Real{0});
    }
};

} // namespace krona::boundary::physical
