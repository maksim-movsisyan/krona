#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>

namespace krona::boundary {

/// @brief Geometry, field indices and time for one face on the boundary of the computational domain.
/// @tparam Real Floating-point type used by the boundary calculation.
/// @details GeometryView::context() builds this object for a condition, profile or transport kernel.
template<class Real>
struct FaceContext {
    LocalIndex face = 0;            ///< Face ID in the local mesh.
    LocalIndex owner = 0;           ///< ID of the adjacent owned cell.
    std::size_t boundary_index = 0; ///< Index in a boundary-only array, starting at the first boundary face.
    std::size_t extended_index = 0; ///< Index in an extended field, after owned and MPI ghost cells.

    // Face centroid and outward unit normal in Cartesian coordinates.
    Real x = 0, y = 0, z = 0;
    Real nx = 0, ny = 0, nz = 0;

    Real distance = 0; ///< Owner-centre-to-face displacement projected onto the outward normal.
    Real area = 0;     ///< Face area for converting a flux density to a flux through the face.
    Real time = 0;     ///< Time supplied by the caller for this evaluation.

    // Tangential part of the owner-centre-to-face displacement, used for gradient correction.
    Real tx = 0, ty = 0, tz = 0;
};

} // namespace krona::boundary
