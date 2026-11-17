#pragma once

#include <krona/execution/types.hpp>

#include <concepts>

namespace krona::thermodynamics {

/// @brief Local thermodynamic properties of a material.
/// @tparam Scalar Floating-point type used for the evaluated properties.
/// Use a single unit system throughout. Energies and entropy are per unit mass.
template <class Scalar> 
struct State {
    Scalar pressure{};        ///< Static thermodynamic pressure.
    Scalar temperature{};     ///< Absolute temperature.
    Scalar density{};         ///< Mass per unit volume.
    Scalar internal_energy{}; ///< Specific internal energy with the model's energy reference.
    Scalar enthalpy{};        ///< Specific static enthalpy e + p/rho.
    Scalar entropy{};         ///< Specific entropy with the model's entropy reference.
    Scalar sound_speed{};     ///< Adiabatic sound speed.
};

/// @brief Partial derivatives of pressure and temperature in (rho, e) coordinates.
/// @tparam Scalar Type of each derivative.
/// Here rho is the mass density and e is the specific internal energy.
template <class Scalar> 
struct RhoEDerivatives {
    Scalar dp_drho_e{}; ///< Pressure derivative with respect to density at fixed internal energy.
    Scalar dp_de_rho{}; ///< Pressure derivative with respect to internal energy at fixed density.
    Scalar dT_drho_e{}; ///< Temperature derivative with respect to density at fixed internal energy.
    Scalar dT_de_rho{}; ///< Temperature derivative with respect to internal energy at fixed density.
};

/// @brief Checks the two conversions used to query an acoustic equation of state.
/// @tparam EOS Model type with const evaluation methods.ё
/// @tparam Scalar Scalar type of both inputs and all output properties.
/// The conversions must return State<Scalar>. Entropy inversions and derivatives are separate capabilities.
template <class EOS, class Scalar>
concept AcousticEOS = requires(const EOS& eos, Scalar p, Scalar T, Scalar rho, Scalar e) {
    { eos.from_pT(p, T) } -> std::same_as<State<Scalar>>;
    { eos.from_rhoe(rho, e) } -> std::same_as<State<Scalar>>;
};

} // namespace krona::thermodynamics