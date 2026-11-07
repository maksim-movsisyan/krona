#pragma once

#include <krona/mesh/aux_types.hpp>

#include "krona/config.hpp"
#include <krona/mesh/types.hpp>
#include <span>
#include <vector>
#include <krona/mpi/communicator.hpp>

namespace krona::mesh::host {

struct MeshPart;
struct MeshAuxConnectivity;

enum class WallDistStencil : std::uint8_t {
    FaceNeighbors,
    NodeNeighbors // Uses only vertex neighbours present in the current local halo.
};

/// @brief Controls approximate shortest-path relaxation from wall-adjacent cells.
struct WallDistanceOptions {
    WallDistStencil stencil = WallDistStencil::FaceNeighbors;
    int max_sweeps = 100;
    double relative_tolerance = 1.0e-4;
};

/// @brief Replicated wall-distance convergence information; unreachable cells retain infinity.
struct WallDistanceStats {
    int sweeps = 0;
    double relative_change = 0;
    bool converged = false;
    std::uint64_t unreachable_owned = 0; // Global owned-cell count, excluding ghost copies.
};

/// @brief Optional host metrics stored in GeometryReal and computed with double intermediates.
struct MeshAuxGeometry {
    AuxGeomType active_mask = AuxGeomType::None;
    LocalIndex n_faces = 0, n_cells = 0; // Snapshot counts, including ghost cells.

    // Interior d=C_neighbour-C_owner; boundary d=(F-C_owner) distance to face center.
    std::vector<GeometryReal> face_cell_dist;
    std::vector<GeometryReal> face_cell_dist_x, face_cell_dist_y, face_cell_dist_z;
    std::vector<GeometryReal> face_cell_dist_inv;

    // Interior dn=dot(C_neighbour-C_owner, normal); boundary dn=dot(F-C_owner, normal).
    std::vector<GeometryReal> face_normal_dist;
    std::vector<GeometryReal> face_normal_dist_inv;
    std::vector<GeometryReal> face_area_over_normal_dist;

    // Interior d=C_neighbour-C_owner; boundary d=F-C_owner.
    // phi_face=w*phi_owner+(1-w)*phi_other; boundary weight is 0.
    std::vector<GeometryReal> face_interp_weight;
    // Over-relaxed correction T: S=Delta+T, with Delta parallel to d and T dot S=0.
    std::vector<GeometryReal> face_non_ortho_x, face_non_ortho_y, face_non_ortho_z;

    // Owned+ghost cells; approximate graph distance.
    std::vector<GeometryReal> cell_wall_dist;

    /// @brief Computes missing face metrics locally in one pass; only requested arrays are allocated.
    /// WallDistance is handled only by the separate collective call and is never marked ready here.
    void add_geometry(const MeshPart& mesh, AuxGeomType requested);

    /// @brief Computes or recomputes graph wall distance collectively and refreshes final ghost values.
    /// wall_patch is a replicated byte mask; nonzero entries identify physical wall patches.
    [[nodiscard]] WallDistanceStats add_wall_distance(const MeshPart& mesh,
                                                      MeshAuxConnectivity& connectivity,
                                                      mpi::CommView comm,
                                                      std::span<const std::uint8_t> wall_patch,
                                                      WallDistanceOptions options = {});

    [[nodiscard]] bool has(AuxGeomType flag) const noexcept { return has_flag(active_mask, flag); }
    [[nodiscard]] bool has_face_cell_dist() const noexcept { return has(AuxGeomType::FaceCellDistance); }
    [[nodiscard]] bool has_face_cell_dist_vector() const noexcept { return has(AuxGeomType::FaceCellDistanceVector); }
    [[nodiscard]] bool has_face_cell_dist_inv() const noexcept { return has(AuxGeomType::FaceCellDistanceInv); }
    [[nodiscard]] bool has_face_interp_weights() const noexcept { return has(AuxGeomType::FaceInterpolationWeights); }
    [[nodiscard]] bool has_face_non_ortho() const noexcept { return has(AuxGeomType::FaceNonOrthogonalityVector); }
    [[nodiscard]] bool has_face_normal_dist() const noexcept { return has(AuxGeomType::FaceNormalDistance); }
    [[nodiscard]] bool has_face_normal_dist_inv() const noexcept { return has(AuxGeomType::FaceNormalDistanceInv); }
    [[nodiscard]] bool has_face_area_over_normal_dist() const noexcept { return has(AuxGeomType::FaceAreaOverNormalDistance); }
    [[nodiscard]] bool has_wall_dist() const noexcept { return has(AuxGeomType::WallDistance); }

    /// @brief Releases cached metrics before recomputation after motion or topology changes.
    void clear() noexcept
    {
        // Replacing with an empty object releases retained vector capacities.
        *this = MeshAuxGeometry{};
    }
};

/// @brief Builds requested face metrics locally; wall distance needs its explicit collective call.
[[nodiscard]] MeshAuxGeometry build_aux_geometry(const MeshPart& mesh, AuxGeomType mask);

// Requires valid topology, grouped faces and complete, synchronized owned/ghost geometry.
// Add calls cache an unchanged snapshot: clear after motion, geometry changes or reordering.
// Wall distance is a graph-path approximation seeded by owner-to-wall-face-centroid distance,
// not an exact nearest-surface query. NodeNeighbors is incomplete without an expanded node halo.
// All ranks call add_wall_distance, including empty or disconnected ranks, with identical options.
// A converged relaxation may still have unreachable cells; inspect the returned global count.

} // namespace krona::mesh::host
