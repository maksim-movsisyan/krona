#pragma once

#include <krona/execution/types.hpp>
#include <krona/math/vector3.hpp>

namespace krona::flow {

/// @brief Mass, momentum and total energy flux through a face.
/// The face law returns flux per unit area. The assembler integrates it over the face.
/// @tparam Real Scalar type of all five flux components.
template<class Real = ComputeReal>
struct Flux {
    Real mass=0;                    ///< Mass flux.
    math::Vector3<Real> momentum{}; ///< Cartesian momentum flux.
    Real energy=0;                  ///< Total energy flux.

    /// @brief Adds corresponding entries.
    KOKKOS_INLINE_FUNCTION Flux operator+(Flux b) const
    { return {mass+b.mass,momentum+b.momentum,energy+b.energy}; }

    /// @brief Subtracts corresponding entries.
    KOKKOS_INLINE_FUNCTION Flux operator-(Flux b) const
    { return {mass-b.mass,momentum-b.momentum,energy-b.energy}; }

    /// @brief Scales all five entries by the same factor.
    KOKKOS_INLINE_FUNCTION Flux operator*(Real a) const
    { return {a*mass,momentum*a,a*energy}; }
};

} // namespace krona::flow
