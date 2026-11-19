#pragma once

#include <Kokkos_Core.hpp>

namespace krona::boundary::scalar {

/// @brief Expresses both the face value and outward normal derivative as a * owner_value + b.
/// @tparam Real Floating-point type of the four coefficients.
/// @details A solver can use the coefficients directly in an implicit assembly. Apply material
/// coefficients and face area separately when turning the derivative into a flux.
template<class Real>
struct Relation {
    Real value_owner = 0;       ///< Multiplier of the owner value in the face value.
    Real value_constant = 0;    ///< Part of the face value independent of the owner value.
    Real gradient_owner = 0;    ///< Multiplier of the owner value in the normal derivative.
    Real gradient_constant = 0; ///< Part of the normal derivative independent of the owner value.

    /// @brief Evaluates the scalar at the face. A zero multiplier returns the constant directly.
    /// @param owner_value Scalar value at the centre of the adjacent cell.
    KOKKOS_INLINE_FUNCTION Real face_value(Real owner_value) const
    { return value_owner == Real{0} ? value_constant : value_owner*owner_value+value_constant; }

    /// @brief Evaluates the outward normal derivative. A zero multiplier returns the constant directly.
    /// @param owner_value Scalar value at the centre of the adjacent cell.
    KOKKOS_INLINE_FUNCTION Real normal_gradient(Real owner_value) const
    { return gradient_owner == Real{0} ? gradient_constant : gradient_owner*owner_value+gradient_constant; }
};

/// @brief Checks that all four coefficients are finite before a face uses the relation.
/// @tparam Real Floating-point type of the coefficients.
/// @param r Relation produced by the boundary condition.
template<class Real>
KOKKOS_INLINE_FUNCTION bool finite(const Relation<Real>& r) {
    return Kokkos::isfinite(r.value_owner) && Kokkos::isfinite(r.value_constant) &&
           Kokkos::isfinite(r.gradient_owner) && Kokkos::isfinite(r.gradient_constant);
}

} // namespace krona::boundary::scalar
