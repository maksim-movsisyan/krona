#pragma once

#include <krona/mpi/communicator.hpp>
#include <krona/mesh/raw_mesh.hpp>
#include <krona/mesh/element.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace krona::mesh {

inline constexpr GlobalIndex invalid_cell_id = -1;
inline constexpr std::uint8_t invalid_local_face = 255;

/// @brief One unique face, stored on the rank owning its smaller cell ID.
struct FaceRecord {
    FaceKey key; // Matching key; oriented nodes come from cell_a and local_face_a.
    GlobalIndex cell_a = invalid_cell_id;
    GlobalIndex cell_b = invalid_cell_id; // No neighbour on a boundary face.
    PatchId patch = invalid_patch_id;
    std::uint8_t local_face_a = invalid_local_face;
    std::uint8_t local_face_b = invalid_local_face;
};

/// @brief Symmetric cell adjacency; rows follow the local cell order.
struct DualGraph {
    std::vector<std::size_t> offsets; // Host-storage offsets, including the final sentinel.
    std::vector<GlobalIndex> adjacency; // Sorted, unique global neighbour IDs per row.
};

/// @brief Global topology counts; graph_arcs counts directed CSR entries.
struct FaceStats {
    std::uint64_t faces = 0;
    std::uint64_t boundary_faces = 0;
    std::uint64_t interior_faces = 0;
    std::uint64_t graph_arcs = 0;
};

/// @brief Controls the average half-face payload per rank and round.
struct BuildFacesOptions {
    std::size_t batch_bytes = 64 * 1024 * 1024;
};

/// @brief Owns unique faces, local CSR rows and global statistics.
struct BuildFacesResult {
    std::vector<FaceRecord> faces;
    DualGraph graph;
    FaceStats stats;
};
 
/// @brief Collectively matches cell faces and boundary labels, then builds the dual graph.
/// Uses connectivity only; coordinates and the mesh source do not affect matching.
[[nodiscard]] BuildFacesResult build_faces(
    const RawMesh& mesh, mpi::CommView comm, const BuildFacesOptions& options = {});

// All ranks participate with identical options, including ranks without cells or labels.
// RawMesh contains owned cells only, in its current block-distribution order.
// Connectivity uses globally consistent node IDs and canonical CGNS node ordering.
// Supported cells: TET, PYRA, PRISM, HEXA; boundary elements: TRI, QUAD.
// Faces shared by more than two cells are rejected; hanging faces need prior conformity.
// Boundary labels are optional; unmatched exterior faces retain invalid_patch_id.
// Batch size is a target, not a hard receive-memory limit under hash imbalance.
// Face order is unspecified; local face indices preserve oriented connectivity.

} // namespace krona::mesh
