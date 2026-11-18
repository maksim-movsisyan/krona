#pragma once

#include <krona/execution/types.hpp>
#include <krona/math/vector3.hpp>
#include <krona/math/symmetric_tensor3.hpp>
#include <cstddef>

namespace krona::numerics::reconstruction {

template <class Real = ComputeReal>
using Vector = math::Vector3<Real>;

template <class Real = ComputeReal>
using SymmetricTensor = math::SymmetricTensor3<Real>;

using math::dot;
using math::dot_as;

/// @brief Selects the side data a formula needs to evaluate one trace.
enum class Geometry {
    Cell,    ///< Base cell index.
    Point,   ///< Cell index and displacement to the point.
    Line,    ///< Point data plus the opposite sample, span and line fraction.
    Explicit ///< Side supplied by the caller, for example a three-sample QUICK stencil.
};

/// @brief One side of a face, oriented from the cell that supplies the reconstructed value.
/// @tparam Real Scalar type of the displacements and line fraction.
template <class Real = ComputeReal>
struct CellSide {
    std::size_t cell = 0;  ///< Cell whose value and derivatives define the profile.
    std::size_t other = 0; ///< Opposite sample index for a line-based formula.
    Vector<Real> offset{}; ///< From this cell centroid to the evaluation point.
    Vector<Real> span{};   ///< From this cell centroid to the opposite sample.
    Real fraction = -1;    ///< Position along span. A negative value requests orthogonal projection.
};

/// @brief Views the same face point from the owner and neighbour.
/// @tparam Real Scalar type of both cell-side geometries.
template <class Real = ComputeReal>
struct FaceGeometry {
    CellSide<Real> left, right;
};

/// @brief Forms oriented sides from the two offsets to the same point.
/// @tparam Real Scalar type of the offsets and optional multiplier.
/// @param owner Base cell for the left side.
/// @param neighbour Base cell for the right side.
/// @param owner_to_face Displacement from the owner centroid to the point.
/// @param neighbour_to_face Displacement from the neighbour centroid to the point.
/// @param owner_fraction Position on the owner-to-neighbour line, or a negative projection marker.
template <class Real>
KOKKOS_INLINE_FUNCTION auto face_geometry(std::size_t owner, std::size_t neighbour, Vector<Real> owner_to_face,
                                          Vector<Real> neighbour_to_face, Real owner_fraction = Real{-1}) {
    const auto span = owner_to_face - neighbour_to_face;
    return FaceGeometry<Real>{{owner, neighbour, owner_to_face, span, owner_fraction},
                              {neighbour, owner, neighbour_to_face, -span,
                               owner_fraction < Real{0} ? Real{-1} : Real{1} - owner_fraction}};
}

/// @brief Reconstructed owner and neighbour values in input component order.
/// @tparam Real Scalar type of the evaluated values.
/// @tparam N Number of scalar components in each state.
template <class Real, std::size_t N>
struct FaceValues {
    Kokkos::Array<Real, N> left{}, right{};
};

/// @brief Holds two explicitly prepared side stencils.
/// @tparam Left Stencil for the owner side.
/// @tparam Right Stencil for the neighbour side.
template <class Left, class Right = Left>
struct FaceStencil {
    Left left;
    Right right;
};

} // namespace krona::numerics::reconstruction
