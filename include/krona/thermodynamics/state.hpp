#pragma once

#include <krona/execution/types.hpp>
#include <concepts>

namespace krona::thermodynamics {

/// Specific internal energy/enthalpy; entropy has the model's explicit reference.
template<class Scalar>
struct State {
    Scalar pressure{}, temperature{}, density{}, internal_energy{}, enthalpy{}, entropy{}, sound_speed{};
};

template<class Scalar>
struct RhoEDerivatives {
    Scalar dp_drho_e{}, dp_de_rho{}, dT_drho_e{}, dT_de_rho{};
};

/// A single-phase acoustic EOS. Additional inversions are separate capabilities.
template<class EOS, class Scalar>
concept AcousticEOS = requires(const EOS& eos, Scalar p, Scalar T, Scalar rho, Scalar e) {
    { eos.from_pT(p, T) } -> std::same_as<State<Scalar>>;
    { eos.from_rhoe(rho, e) } -> std::same_as<State<Scalar>>;
};

} // namespace krona::thermodynamics
