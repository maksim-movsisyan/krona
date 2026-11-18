#pragma once

#include <krona/numerics/reconstruction/geometry.hpp>

#include <type_traits>
#include <utility>

namespace krona::numerics::reconstruction {
namespace detail {

/// @brief Reconstructs each component by adding the scheme's correction to the base cell value.
/// @tparam Real Floating-point type used for the calculation and result.
/// @tparam Formula Reconstruction scheme with its inputs already attached, such as a linear scheme with gradients.
/// @tparam Side Side descriptor with the base cell index and any geometry or sample indices the scheme needs.
/// @tparam Values Field pack or another reader that provides values by cell or sample index and component.
/// @tparam Component Compile-time component indices, from 0 to Values::components - 1.
/// @param formula Scheme that computes the correction for each component.
/// @param side Side being reconstructed. Its `cell` member selects the base value.
/// @param values Values for the base cell and any additional samples read by the scheme.
/// @details The index sequence lets the compiler expand the calculation once for each component.
template <class Real, class Formula, class Side, class Values, std::size_t... Component>
KOKKOS_INLINE_FUNCTION auto evaluate_components(const Formula& formula, const Side& side, const Values& values,
                                                std::index_sequence<Component...>) {
    return Kokkos::Array<Real, Values::components>{
        (static_cast<Real>(values(side.cell, Component)) +
         formula.template increment<Real>(side, values, Component))...};
}

} // namespace detail

/// @brief Evaluates one reconstructed state in the caller's kernel.
/// @tparam Real Arithmetic type of the result.
/// @tparam Formula Bound formula that returns the change from the base cell value.
/// @tparam Side Geometry or explicit samples for the requested side.
/// @tparam Values Provider of base values in component order.
template <class Real = ComputeReal, class Formula, class Side, class Values>
KOKKOS_INLINE_FUNCTION auto evaluate_side(const Formula& formula, const Side& side, const Values& values) {
    static_assert(std::is_floating_point_v<Real> && Values::components > 0);
    return detail::evaluate_components<Real>(formula, side, values,
                                             std::make_index_sequence<Values::components>{});
}

/// @brief Evaluates owner and neighbour states at the same face point.
/// @tparam Real Arithmetic type of both results.
/// @tparam Formula Bound formula to use on both sides.
/// @tparam Face Pair with left and right side descriptors.
/// @tparam Values Provider of values for all referenced cells.
template <class Real = ComputeReal, class Formula, class Face, class Values>
KOKKOS_INLINE_FUNCTION auto evaluate(const Formula& formula, const Face& face, const Values& values) {
    return FaceValues<Real, Values::components>{evaluate_side<Real>(formula, face.left, values),
                                                evaluate_side<Real>(formula, face.right, values)};
}

} // namespace krona::numerics::reconstruction
