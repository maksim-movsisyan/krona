#pragma once

#include <krona/mesh/element.hpp>
#include <krona/mesh/host/patch.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace krona::mesh::host {

/// @brief Solver mesh with owned cells, one face-neighbour halo and local connectivity.
struct MeshPart {
    int rank = 0;
    int nprocs = 1;

    // Global metadata (replicated on all ranks) 
    std::uint64_t n_cells_g = 0, n_nodes_g = 0, n_faces_g = 0;
    std::uint64_t n_bfaces_g = 0;                               // All exterior faces, including unlabelled ones.
    std::uint64_t n_labeled_bfaces_g = 0, n_cut_faces_g = 0;
    std::uint64_t n_ghost_copies_g = 0;                         // Sum of local ghosts, not unique cells.
    std::array<GeometryReal, 3> bbox_lo{}, bbox_hi{};


    // Cells 
    // [0, n_own) are owned cells (SFC or RCM ordered for cache locality).
    // [n_own, n_cells) are ghost cells received from neighbours.
    LocalIndex n_own = 0, n_cells = 0;
    std::vector<CellType> cell_type;
    std::vector<GlobalIndex> cell_gid;
    std::vector<int> cell_donor;                                // -1 for owned cells; otherwise the new owner rank.
    std::vector<std::size_t> cell_nodes_offsets{0};
    std::vector<LocalIndex> cell_nodes;


    // Nodes
    // [0, n_nodes_own) are nodes used by at least one owned cell.
    // [n_nodes_own, n_nodes) are nodes used by ghost cells.
    LocalIndex n_nodes_own = 0, n_nodes = 0;
    std::vector<GlobalIndex> node_gid;
    std::vector<GeometryReal> node_x, node_y, node_z;

    
    // Faces
    // Every face owner is owned; the neighbour is owned, ghost, or -1 at a boundary.
    LocalIndex n_faces = 0;
    std::vector<LocalIndex> face_owner, face_neigh;
    std::vector<CellType> face_type;
    std::vector<PatchId> face_patch;
    std::vector<std::uint8_t> face_local_index;             // Facet of the current owner cell.
    std::vector<std::size_t> face_nodes_offsets{0};
    std::vector<LocalIndex> face_nodes;                     // Oriented CGNS facet order, relative to owner.

    // Valid ranges only after the separate face-grouping/reordering stage.
    // [0,n_internal_faces): owned-owned; [n_internal_faces,n_inner_faces): owned-ghost;
    // [n_inner_faces,n_faces): boundary. Migration leaves faces_grouped false.
    bool faces_grouped = false;
    LocalIndex n_internal_faces = 0, n_inner_faces = 0;

    // Sorted peer ranks; matching send/receive order for future field exchanges.
    std::vector<int> nb_ranks;
    std::vector<std::size_t> recv_offsets{0}, send_offsets{0};
    std::vector<LocalIndex> recv_ghost_local, send_owned_local;

    // Patch labels describe the source; physical solver BCs are assigned separately.
    std::vector<Patch> patches;
    std::vector<std::size_t> patch_face_offsets{0};
    std::vector<LocalIndex> patch_faces;

    // Filled by a separate geometry stage; migration allocates none of these arrays.
    std::vector<GeometryReal> cell_centroid_x, cell_centroid_y, cell_centroid_z, cell_volume;
    std::vector<GeometryReal> face_centroid_x, face_centroid_y, face_centroid_z;
    std::vector<GeometryReal> face_normal_x, face_normal_y, face_normal_z, face_area;
    std::uint64_t n_flipped = 0;

    /// @brief Returns the number of halo peers in the current communicator.
    [[nodiscard]] int n_neighbors() const noexcept
    {
        // Peer count never exceeds the communicator size.
        return static_cast<int>(nb_ranks.size());
    }
};

// IDs are zero-based; local indices fit LocalIndex, CSR storage offsets use size_t.
// Cut faces occur once on each adjacent rank; boundary faces occur once globally.
// Facet orientation assumes valid, positively oriented input volume cells.
// Reordering must update connectivity, halo indices, patch indices and geometry together.
// A binary writer must encode offset widths explicitly rather than dumping size_t bytes.

} // namespace krona::mesh::host
