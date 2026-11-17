#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>

namespace krona::thermodynamics {

/// @brief Material with prescribed density for incompressible flow.
/// @tparam Real Type used to store the material parameter.
template <class Real = ComputeReal> 
struct ConstantDensity {
    Real rho = 1000; ///< Mass per unit volume.

    /// @brief Checks that the prescribed density is finite and positive on the host.
    void validate() const {
        mpi::require(std::isfinite(rho) && rho > Real{0},
                     "ConstantDensity: density must be finite and positive");
    }

    /// @brief Returns the prescribed density. Accepts temperature to match the material interface.
    /// @tparam S Scalar type of the caller's temperature and the returned density.
    template <class S> KOKKOS_INLINE_FUNCTION S density(S) const { return static_cast<S>(rho); }
};

/// @brief Incompressible thermal material with constant density and specific heat.
/// @tparam Real Type used to store the material parameters.
template <class Real = ComputeReal> 
struct ThermalConstantDensity {
    Real rho = 1000;                  ///< Mass per unit volume.
    Real specific_heat = 4200;        ///< Energy per unit mass per unit temperature.
    Real reference_temperature = 273; ///< Temperature at which specific internal energy is zero.

    /// @brief Checks the material parameters on the host.
    void validate() const {
        ConstantDensity<Real>{rho}.validate();
        mpi::require(std::isfinite(specific_heat) && specific_heat > Real{0} &&
                     std::isfinite(reference_temperature) && reference_temperature > Real{0},
                     "ThermalConstantDensity: require positive heat capacity and reference temperature");
    }

    /// @brief Returns the same density at every temperature.
    /// @tparam S Scalar type of the caller's temperature and the returned density.
    template <class S> 
    KOKKOS_INLINE_FUNCTION S density(S) const { return static_cast<S>(rho); }

    /// @brief Returns specific internal energy relative to reference_temperature.
    /// @tparam S Scalar type used for temperature and energy.
    /// @param T Temperature of the state.
    template <class S> 
    KOKKOS_INLINE_FUNCTION S internal_energy(S T) const {
        return static_cast<S>(specific_heat) * (T - static_cast<S>(reference_temperature));
    }

    /// @brief Returns specific static enthalpy by adding the pressure term p/rho to e.
    /// @tparam S Scalar type of the inputs and the returned enthalpy.
    /// @param p Pressure used for the pressure term.
    /// @param T Temperature used to evaluate internal energy.
    template <class S> 
    KOKKOS_INLINE_FUNCTION S enthalpy(S p, S T) const {
        return internal_energy(T) + p / static_cast<S>(rho);
    }
};

} // namespace krona::thermodynamics