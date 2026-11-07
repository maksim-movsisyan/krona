#pragma once

#include <Kokkos_Core.hpp>

namespace krona::boundary::scalar {

/// @brief Affine face value and outward normal derivative; no coefficient or face area.
template<class Real>
struct Relation {
    Real value_owner = 0, value_constant = 0;
    Real gradient_owner = 0, gradient_constant = 0;

    KOKKOS_INLINE_FUNCTION Real face_value(Real owner_value) const
    { return value_owner == Real{0} ? value_constant : value_owner*owner_value+value_constant; }
    KOKKOS_INLINE_FUNCTION Real normal_gradient(Real owner_value) const
    { return gradient_owner == Real{0} ? gradient_constant : gradient_owner*owner_value+gradient_constant; }
};

template<class Real>
KOKKOS_INLINE_FUNCTION bool finite(const Relation<Real>& r)
{
    return Kokkos::isfinite(r.value_owner) && Kokkos::isfinite(r.value_constant) &&
           Kokkos::isfinite(r.gradient_owner) && Kokkos::isfinite(r.gradient_constant);
}

} // namespace krona::boundary::scalar
