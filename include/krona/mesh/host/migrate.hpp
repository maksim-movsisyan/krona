#pragma once

#include <krona/mesh/host/faces.hpp>
#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <span>
#include <vector>

namespace krona::mesh::host {

/// @brief Owns migration payloads with arbitrary contiguous cell and coordinate ownership.
struct MigrationSource {
    ElementTable cells;
    Coordinates coordinates;
    std::vector<Patch> patches;
    std::vector<GlobalIndex> cell_distribution, node_distribution; // P+1 endpoints each.
};

/// @brief Consumes a 3D import, migrates cells/faces and constructs one face-neighbour halo.
[[nodiscard]] MeshPart migrate_local_mesh(
    RawMesh&& source, std::vector<FaceRecord>&& faces,
    std::span<const int> cell_targets, mpi::CommView comm);

/// @brief Consumes source-independent payloads with explicit contiguous ownership ranges.
[[nodiscard]] MeshPart migrate_local_mesh(
    MigrationSource&& source, std::vector<FaceRecord>&& faces,
    std::span<const int> cell_targets, mpi::CommView comm);

// Collective: every rank participates, including empty ranks, on the MPI main thread.
// Source cells/coordinates follow their rank ranges; node and cell IDs are globally dense.
// Unequal ranges are allowed. After adaptation/migration, renumber and remap consistently.
// Faces are unique globally, held on the source owner of cell_a, with valid local facets.
// Cells are linear TET/PYRA/PRISM/HEXA in canonical CGNS node order.
// cell_targets has one valid communicator rank per source cell and stays readable until return.
// Payloads and face storage are consumed; the target-rank span is borrowed and unchanged.
// Geometry and RCM/SFC/face grouping are separate stages; no partitioner dependency exists.
// MPI 3.1 packed counts/displacements must fit int; this implementation is not batched.

} // namespace krona::mesh::host
