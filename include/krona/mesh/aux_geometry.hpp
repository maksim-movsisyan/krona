#pragma once

#include "krona/config.hpp"
#include <krona/mesh/aux_connectivity.hpp>
#include <krona/mpi/communicator.hpp>

namespace krona::mesh {

enum class AuxGeomType : std::uint64_t {
    None                        = std::uint64_t{0},
    FaceCellDistance            = std::uint64_t{1} << 0,
    FaceCellDistanceVector      = std::uint64_t{1} << 1,
    FaceCellDistanceInv         = std::uint64_t{1} << 2,
    FaceInterpolationWeights    = std::uint64_t{1} << 3,
    FaceNonOrthogonalityVector  = std::uint64_t{1} << 4,
    FaceNormalDistance          = std::uint64_t{1} << 5,
    FaceNormalDistanceInv       = std::uint64_t{1} << 6,
    FaceAreaOverNormalDistance  = std::uint64_t{1} << 7,
    WallDistance                = std::uint64_t{1} << 8,
    AllFaces                    = (std::uint64_t{1} << 8) - std::uint64_t{1},
    All                         = (std::uint64_t{1} << 9) - std::uint64_t{1},
};

/// @brief Combines requested auxiliary geometry flags.
[[nodiscard]] constexpr AuxGeomType operator|(AuxGeomType a, AuxGeomType b) noexcept
{
    return static_cast<AuxGeomType>(static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b));
}
/// @brief Selects flags shared by two geometry masks.
[[nodiscard]] constexpr AuxGeomType operator&(AuxGeomType a, AuxGeomType b) noexcept
{
    return static_cast<AuxGeomType>(static_cast<std::uint64_t>(a) & static_cast<std::uint64_t>(b));
}
/// @brief Returns whether every requested flag is present; None is always present.
[[nodiscard]] constexpr bool has_flag(AuxGeomType mask, AuxGeomType flag) noexcept
{
    return (mask & flag) == flag;
}

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
    LocalIndex n_faces = 0;

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

} // namespace krona::mesh
