#pragma once

#include <krona/execution/types.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <type_traits>

namespace krona::transport::properties {

/// @brief Molecular transport coefficients in a consistent system of units.
/// @tparam Real Floating-point type used to store the evaluated coefficients.
template<class Real = ComputeReal>
struct Coefficients {
    static_assert(std::is_floating_point_v<Real>);
    Real dynamic_viscosity = 0;     ///< Dynamic viscosity mu in Pa s in SI.
    Real thermal_conductivity = 0;  ///< Thermal conductivity lambda in W/(m K) in SI.

    /// @brief Returns true if both coefficients are finite and nonnegative.
    KOKKOS_INLINE_FUNCTION bool valid() const {
        return Kokkos::isfinite(dynamic_viscosity) && dynamic_viscosity >= Real{0} &&
               Kokkos::isfinite(thermal_conductivity) && thermal_conductivity >= Real{0};
    }
};

} // namespace krona::transport::properties
