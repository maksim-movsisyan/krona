#pragma once

#include <krona/math/vector3.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Owner weight from the two cell centres and the face plane.
/// The neighbour's projected distance is the total projection minus the owner's.
template<class Real>
KOKKOS_INLINE_FUNCTION Real geometric_owner_weight(
    math::Vector3<Real> owner_to_face, math::Vector3<Real> normal, Real height) {
    return (height - owner_to_face.dot(normal)) / height;
}

/// @brief Part of the outward area vector left after taking the centre-to-centre term.
/// Keep the normal norm in the formula to match the mesh cache's treatment of rounding.
template<class Real>
KOKKOS_INLINE_FUNCTION math::Vector3<Real> nonorthogonal_area(
    math::Vector3<Real> displacement, math::Vector3<Real> normal, Real area, Real height) {
    return area * (normal - displacement * (normal.norm2() / height));
}

} // namespace krona::transport::scalar::diffusion::detail
