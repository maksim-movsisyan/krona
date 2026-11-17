#pragma once

#include <krona/mpi/error.hpp>
#include <krona/thermodynamics/state.hpp>

#include <cmath>

namespace krona::thermodynamics {

/// @brief Calorically perfect gas with constant heat capacities.
/// @tparam Real Type used to store parameters and compute cv() and cp().
/// Set the parameters, then call validate() on the host before using the model.
/// Evaluation methods take the scalar type S from their arguments and return State<S>.
template <class Real = ComputeReal> 
struct IdealGas {
    Real gamma = static_cast<Real>(1.4);           ///< Ratio cp/cv.
    Real gas_constant = static_cast<Real>(287.05); ///< Specific gas constant R.
    Real reference_temperature = 1;                ///< Temperature at the entropy reference state.
    Real reference_pressure = 1;                   ///< Pressure at the entropy reference state.

    /// @brief Checks the model parameters on the host.
    void validate() const {
        mpi::require(std::isfinite(gamma) && gamma > Real{1} && std::isfinite(gas_constant) &&
                     gas_constant > Real{0} && std::isfinite(reference_temperature) &&
                     reference_temperature > Real{0} && std::isfinite(reference_pressure) &&
                     reference_pressure > Real{0},
                     "IdealGas: require finite gamma>1, R>0 and positive reference states");
    }

    /// @brief Returns the specific heat capacity at constant volume.
    KOKKOS_INLINE_FUNCTION Real cv() const { return gas_constant / (gamma - Real{1}); }

    /// @brief Returns the specific heat capacity at constant pressure.
    KOKKOS_INLINE_FUNCTION Real cp() const { return gamma * cv(); }

    /// @brief Evaluates the full state from pressure and temperature.
    /// @tparam S Scalar type of the inputs and the returned state.
    /// @param p Finite positive absolute pressure.
    /// @param T Finite positive absolute temperature.
    template <class S> 
    KOKKOS_INLINE_FUNCTION State<S> from_pT(S p, S T) const {
        if (!(p > S{0}) || !(T > S{0}) || !Kokkos::isfinite(p) || !Kokkos::isfinite(T))
            Kokkos::abort("IdealGas: nonpositive or nonfinite pressure/temperature");

        const auto acoustic = evaluate_acoustic_pT(p, T);
        const S R = static_cast<S>(gas_constant);
        const S c_p = static_cast<S>(cp());

        // Only entropy uses the reference state. The energy relations use absolute T.
        return {.pressure = p,
                .temperature = T,
                .density = acoustic.density,
                .internal_energy = acoustic.internal_energy,
                .enthalpy = c_p * T,
                .entropy = c_p * Kokkos::log(T / static_cast<S>(reference_temperature)) -
                           R * Kokkos::log(p / static_cast<S>(reference_pressure)),
                .sound_speed = acoustic.sound_speed};
    }

    /// @brief Evaluates density, internal energy and sound speed from p and T.
    /// @tparam S Scalar type of the inputs and the acoustic state.
    /// @param p Absolute pressure.
    /// @param T Absolute temperature.
    /// @param state Output, written only when all state members are finite and positive.
    /// Use a validated gas. The acoustic evaluation avoids the entropy logarithms.
    template <class S>
    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool try_acoustic_from_pT(S p, S T, AcousticState<S>& state) const {
        if (!positive_finite(p) || !positive_finite(T))
            return false;
        return store_acoustic(evaluate_acoustic_pT(p, T), state);
    }

    /// @brief Recovers the acoustic state from mass density and specific internal energy.
    /// @tparam S Scalar type of the inputs and the acoustic state.
    /// @param rho Mass density.
    /// @param e Internal energy per unit mass, after removing kinetic energy.
    /// @param state Output, written only when all state members are finite and positive.
    /// Use a validated gas. The returned state uses the supplied rho and e.
    template <class S>
    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool try_acoustic_from_rhoe(S rho, S e, AcousticState<S>& state) const {
        if (!positive_finite(rho) || !positive_finite(e))
            return false;
        const S T = e / static_cast<S>(cv());
        const S p = rho * static_cast<S>(gas_constant) * T;
        return store_acoustic({p, T, rho, e, sound_speed(T)}, state);
    }

    /// @brief Recovers the acoustic state from mass density and pressure.
    /// @tparam S Scalar type of the inputs and the acoustic state.
    /// @param rho Mass density.
    /// @param p Absolute pressure.
    /// @param state Output, written only when all state members are finite and positive.
    /// Use a validated gas. The returned state uses the supplied rho and p.
    template <class S>
    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool try_acoustic_from_rhop(S rho, S p, AcousticState<S>& state) const {
        if (!positive_finite(rho) || !positive_finite(p))
            return false;
        const S T = p / (rho * static_cast<S>(gas_constant));
        return store_acoustic({p, T, rho, static_cast<S>(cv()) * T, sound_speed(T)}, state);
    }

    /// @brief Recovers the state from mass density and specific internal energy.
    /// @tparam S Scalar type of the inputs and the returned state.
    /// @param rho Finite positive mass density.
    /// @param e Finite positive internal energy per unit mass, after removing kinetic energy.
    template <class S> 
    KOKKOS_INLINE_FUNCTION State<S> from_rhoe(S rho, S e) const {
        if (!(rho > S{0}) || !(e > S{0}) || !Kokkos::isfinite(rho) || !Kokkos::isfinite(e))
            Kokkos::abort("IdealGas: nonpositive or nonfinite density/internal energy");

        // Recover T from e = cv*T, then use p = rho*R*T.
        const S T = e / static_cast<S>(cv());
        return from_pT(rho * static_cast<S>(gas_constant) * T, T);
    }

    /// @brief Recovers the state at a specified pressure and entropy.
    /// @tparam S Scalar type of the inputs and the returned state.
    /// @param p Positive absolute pressure.
    /// @param entropy Specific entropy with this gas model's reference.
    template <class S>
    KOKKOS_INLINE_FUNCTION State<S> from_ps(S p, S entropy) const {
        // Rearrange the entropy relation to isolate log(T/T_ref).
        const S T = static_cast<S>(reference_temperature) *
                    Kokkos::exp((entropy + static_cast<S>(gas_constant) *
                                               Kokkos::log(p / static_cast<S>(reference_pressure))) /
                                static_cast<S>(cp()));
        return from_pT(p, T);
    }

    /// @brief Recovers the state from specific static enthalpy and entropy.
    /// @tparam S Scalar type of the inputs and the returned state.
    /// @param h Specific static enthalpy, with the kinetic contribution removed.
    /// @param entropy Specific entropy with this gas model's reference.
    template <class S> 
    KOKKOS_INLINE_FUNCTION State<S> from_hs(S h, S entropy) const {
        // Enthalpy fixes T, so the entropy relation gives log(p/p_ref).
        const S T = h / static_cast<S>(cp());
        const S p =
            static_cast<S>(reference_pressure) *
            Kokkos::exp(
                (static_cast<S>(cp()) * Kokkos::log(T / static_cast<S>(reference_temperature)) - entropy) /
                static_cast<S>(gas_constant));
        return from_pT(p, T);
    }

    /// @brief Evaluates the derivatives of p = (gamma - 1)*rho*e and T = e/cv.
    /// @tparam S Scalar type of the inputs and the returned derivatives.
    /// @param rho Mass density at the evaluation point.
    /// @param e Specific internal energy at the evaluation point.
    template <class S> 
    KOKKOS_INLINE_FUNCTION RhoEDerivatives<S> rhoe_derivatives(S rho, S e) const {
        const S gamma_minus_one = static_cast<S>(gamma) - S{1};
        return {gamma_minus_one * e, gamma_minus_one * rho, S{0}, S{1} / static_cast<S>(cv())};
    }

  private:
    template <class S> 
    KOKKOS_INLINE_FUNCTION S sound_speed(S T) const {
        return Kokkos::sqrt(static_cast<S>(gamma) * static_cast<S>(gas_constant) * T);
    }

    // The full and acoustic evaluations use the same formulas for their shared properties.
    template <class S> 
    KOKKOS_INLINE_FUNCTION AcousticState<S> evaluate_acoustic_pT(S p, S T) const {
        const S R = static_cast<S>(gas_constant);
        return {p, T, p / (R * T), static_cast<S>(cv()) * T, sound_speed(T)};
    }

    template <class S> KOKKOS_INLINE_FUNCTION static bool positive_finite(S value) {
        return value > S{0} && Kokkos::isfinite(value);
    }

    template <class S>
    KOKKOS_INLINE_FUNCTION static bool store_acoustic(AcousticState<S> candidate, AcousticState<S>& state) {
        if (!positive_finite(candidate.pressure) || !positive_finite(candidate.temperature) ||
            !positive_finite(candidate.density) || !positive_finite(candidate.internal_energy) ||
            !positive_finite(candidate.sound_speed))
            return false;
        state = candidate;
        return true;
    }
};

} // namespace krona::thermodynamics