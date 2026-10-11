#pragma once

#include <krona/mesh/localmesh.hpp>

#include <cassert>
#include <span>

namespace krona::mesh {

enum class AuxConnType : std::uint64_t {
    None                        = 0,
    CellFaces                   = std::uint64_t{1} << 0,
    CellCellsByFace             = std::uint64_t{1} << 1,
    NodeCells                   = std::uint64_t{1} << 2,
    NodeFaces                   = std::uint64_t{1} << 3,
    CellCellsByNode             = std::uint64_t{1} << 4,
    All                         = (std::uint64_t{1} << 5) - 1
};

/// @brief Combines requested connectivity flags.
[[nodiscard]] constexpr AuxConnType operator|(AuxConnType a, AuxConnType b) noexcept
{
    return static_cast<AuxConnType>(static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b));
}

/// @brief Selects flags shared by two connectivity masks.
[[nodiscard]] constexpr AuxConnType operator&(AuxConnType a, AuxConnType b) noexcept
{
    return static_cast<AuxConnType>(static_cast<std::uint64_t>(a) & static_cast<std::uint64_t>(b));
}

/// @brief Returns whether every requested flag is present; None is always present.
[[nodiscard]] constexpr bool has_flag(AuxConnType mask, AuxConnType flag) noexcept
{
    return (mask & flag) == flag;
}

/// @brief Local optional CSR tables; values use LocalIndex and storage offsets use size_t.
struct MeshAuxConnectivity {
    AuxConnType active_mask = AuxConnType::None;
    LocalIndex n_faces = 0, n_cells_own = 0, n_nodes_own = 0;

    // Cell rows belong to owned cells; neighbour values may refer to ghosts.
    std::vector<std::size_t> cell_faces_offsets, cell_cells_face_offsets, cell_cells_node_offsets;
    std::vector<LocalIndex> cell_faces, cell_cells_face, cell_cells_node;

    // Node rows belong to the owned-used prefix, not to exclusive global node owners.
    std::vector<std::size_t> node_cells_offsets, node_faces_offsets;
    std::vector<LocalIndex> node_cells, node_faces;

    /// @brief Builds missing requested tables locally, retaining tables already built.
    void add_connectivity(const MeshPart& mesh, AuxConnType requested);

    [[nodiscard]] bool has(AuxConnType flag) const noexcept { return has_flag(active_mask, flag); }
    [[nodiscard]] bool has_cell_faces() const noexcept { return has(AuxConnType::CellFaces); }
    [[nodiscard]] bool has_cell_cells_face() const noexcept { return has(AuxConnType::CellCellsByFace); }
    [[nodiscard]] bool has_cell_cells_node() const noexcept { return has(AuxConnType::CellCellsByNode); }
    [[nodiscard]] bool has_node_cells() const noexcept { return has(AuxConnType::NodeCells); }
    [[nodiscard]] bool has_node_faces() const noexcept { return has(AuxConnType::NodeFaces); }

    /// @brief Releases all tables; call before rebuilding after topology or numbering changes.
    void clear() noexcept
    {
        // Move an empty object over the old one to release vector capacities as well.
        *this = MeshAuxConnectivity{};
    }
};

/// @brief Builds only the requested tables from valid local MeshPart topology; no MPI is used.
[[nodiscard]] MeshAuxConnectivity build_aux_connectivity(const MeshPart& mesh, AuxConnType mask);

/// @brief Builds owned-cell-to-face CSR; internal faces appear in both owned rows.
void build_cell_faces_conn(const MeshPart& mesh, std::vector<std::size_t>& offsets,
                           std::vector<LocalIndex>& faces);
/// @brief Builds sorted owned-cell face neighbours, retaining one entry per connecting face.
void build_cell_cells_face_conn(const MeshPart& mesh, std::vector<std::size_t>& offsets,
                                std::vector<LocalIndex>& cells);
/// @brief Builds node-to-cell CSR for owned-used nodes, scanning owned and ghost cells.
void build_node_cells_conn(const MeshPart& mesh, std::vector<std::size_t>& offsets,
                           std::vector<LocalIndex>& cells);
/// @brief Builds node-to-face CSR for owned-used nodes in increasing local face order.
void build_node_faces_conn(const MeshPart& mesh, std::vector<std::size_t>& offsets,
                           std::vector<LocalIndex>& faces);
/// @brief Builds unique local vertex neighbours, reusing an existing node-to-cell table.
void build_cell_cells_node_conn(const MeshPart& mesh, std::span<const std::size_t> node_offsets,
                                std::span<const LocalIndex> node_cells,
                                std::vector<std::size_t>& offsets, std::vector<LocalIndex>& cells);
/// @brief Builds unique local vertex neighbours using a temporary node-to-cell table.
void build_cell_cells_node_conn(const MeshPart& mesh, std::vector<std::size_t>& offsets,
                                std::vector<LocalIndex>& cells);

// Input topology is valid, with every local face owned by an owned cell.
// CellCellsByNode/NodeCells describe only cells present in this MeshPart: a face halo
// does not supply all remote vertex-only neighbours. Expand the halo separately if needed.
// Clear/rebuild after reordering, adaptation or migration, even when counts are unchanged.
// Coordinate-only motion does not invalidate connectivity. Geometry caches are separate.

} // namespace krona::mesh
