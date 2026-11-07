#pragma once

#include <krona/execution/types.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <type_traits>

namespace krona::transport::properties {

/// @brief Molecular transport coefficients in a consistent system of units.
template<class Real = ComputeReal>
struct Coefficients {
    static_assert(std::is_floating_point_v<Real>);
    Real dynamic_viscosity = 0;
    Real thermal_conductivity = 0;

    /// @brief Optional pointwise check; zero coefficients are allowed.
    KOKKOS_INLINE_FUNCTION bool valid() const
    {
        return Kokkos::isfinite(dynamic_viscosity) && dynamic_viscosity >= Real{0} &&
               Kokkos::isfinite(thermal_conductivity) && thermal_conductivity >= Real{0};
    }
};

} // namespace krona::transport::properties
