#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <array>
#include <limits>

namespace krona::mesh::host {

/// @brief Geometry statistics; cell volumes count owned cells, face extrema include local copies.
struct GeometryStats {
    double total_volume = 0;
    double min_cell_volume = std::numeric_limits<double>::infinity();
    double max_cell_volume = 0;
    double min_face_area = std::numeric_limits<double>::infinity();
    double max_face_area = 0;
    std::array<double, 3> bbox_lo{
        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> bbox_hi{
        -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
};

/// @brief Recomputes local cell/face metrics and checks outward winding without collectives.
/// Returns local statistics; empty local minima/bounds retain their infinity sentinels.
[[nodiscard]] GeometryStats compute_mesh_geometry(MeshPart& mesh);

/// @brief Recomputes metrics collectively and returns global statistics with refreshed bounds.
/// Performs two reductions; the global empty-mesh result has zero extrema and bounds.
[[nodiscard]] GeometryStats compute_mesh_geometry(MeshPart& mesh, mpi::CommView comm);

// Requires valid local topology and synchronized owned/ghost coordinates on the MPI main thread.
// Supports linear TET/PYRA/PRISM/HEXA and outward TRI/QUAD loops; quads use a vertex-mean fan.
// face_area = |summed triangle area vector|; its centroid uses actual triangle-area weights.
// Integration uses double and storage uses GeometryReal; positive finite measures must fit storage.
// Topology/grouping/halo maps are unchanged. See the pipeline fragment for the full contract.

} // namespace krona::mesh::host
