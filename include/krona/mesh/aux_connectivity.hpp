#pragma once

#include <krona/mesh/localmesh.hpp>

#include <cassert>
#include <span>

namespace krona::mesh {

enum class AuxConnType : std::uint32_t {
    None = 0,
    CellFaces = 1U << 0,
    CellCellsByFace = 1U << 1,
    NodeCells = 1U << 2,
    NodeFaces = 1U << 3,
    CellCellsByNode = 1U << 4,
    All = (1U << 5) - 1
};

/// @brief Combines requested connectivity flags.
[[nodiscard]] constexpr AuxConnType operator|(AuxConnType a, AuxConnType b) noexcept
{
    return static_cast<AuxConnType>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

/// @brief Selects flags shared by two connectivity masks.
[[nodiscard]] constexpr AuxConnType operator&(AuxConnType a, AuxConnType b) noexcept
{
    return static_cast<AuxConnType>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
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

    /// @brief Returns whether all requested tables have been built, including empty tables.
    [[nodiscard]] bool has(AuxConnType flag) const noexcept { return has_flag(active_mask, flag); }
    /// @brief Returns whether cell-to-face CSR is available.
    [[nodiscard]] bool has_cell_faces() const noexcept { return has(AuxConnType::CellFaces); }
    /// @brief Returns whether face-neighbour CSR is available.
    [[nodiscard]] bool has_cell_cells_face() const noexcept { return has(AuxConnType::CellCellsByFace); }
    /// @brief Returns whether locally available vertex-neighbour CSR is available.
    [[nodiscard]] bool has_cell_cells_node() const noexcept { return has(AuxConnType::CellCellsByNode); }
    /// @brief Returns whether node-to-cell CSR is available.
    [[nodiscard]] bool has_node_cells() const noexcept { return has(AuxConnType::NodeCells); }
    /// @brief Returns whether node-to-face CSR is available.
    [[nodiscard]] bool has_node_faces() const noexcept { return has(AuxConnType::NodeFaces); }

    /// @brief Borrows incident face IDs for one owned cell in increasing face order.
    [[nodiscard]] std::span<const LocalIndex> cell_faces_of(LocalIndex cell) const noexcept
    {
        assert(has_cell_faces() && cell >= 0 && cell < n_cells_own);
        return row(cell_faces_offsets, cell_faces, cell);
    }
    /// @brief Borrows sorted face neighbours of one owned cell, including ghosts.
    [[nodiscard]] std::span<const LocalIndex> cell_neighbors_face_of(LocalIndex cell) const noexcept
    {
        assert(has_cell_cells_face() && cell >= 0 && cell < n_cells_own);
        return row(cell_cells_face_offsets, cell_cells_face, cell);
    }
    /// @brief Borrows sorted unique vertex neighbours available in the current local halo.
    [[nodiscard]] std::span<const LocalIndex> cell_neighbors_node_of(LocalIndex cell) const noexcept
    {
        assert(has_cell_cells_node() && cell >= 0 && cell < n_cells_own);
        return row(cell_cells_node_offsets, cell_cells_node, cell);
    }
    /// @brief Borrows locally stored cells using a node from the owned-used prefix.
    [[nodiscard]] std::span<const LocalIndex> node_cells_of(LocalIndex node) const noexcept
    {
        assert(has_node_cells() && node >= 0 && node < n_nodes_own);
        return row(node_cells_offsets, node_cells, node);
    }
    /// @brief Borrows local face IDs using a node from the owned-used prefix.
    [[nodiscard]] std::span<const LocalIndex> node_faces_of(LocalIndex node) const noexcept
    {
        assert(has_node_faces() && node >= 0 && node < n_nodes_own);
        return row(node_faces_offsets, node_faces, node);
    }
    /// @brief Releases all tables; call before rebuilding after topology or numbering changes.
    void clear() noexcept
    {
        // Move an empty object over the old one to release vector capacities as well.
        *this = MeshAuxConnectivity{};
    }

private:
    /// @brief Borrows a CSR row without pointer arithmetic on an empty vector.
    [[nodiscard]] static std::span<const LocalIndex> row(std::span<const std::size_t> offsets,
                                                         std::span<const LocalIndex> values,
                                                         LocalIndex index) noexcept
    {
        const auto i = static_cast<std::size_t>(index);
        // subspan keeps an empty row valid even when the values vector is empty.
        return values.subspan(offsets[i], offsets[i + 1] - offsets[i]);
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
