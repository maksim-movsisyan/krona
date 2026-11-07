#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/mesh/host/mesh_part.hpp>
#include <cassert>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <limits>

namespace krona::mesh::host {
namespace {

/// @brief Converts row counts to starting cursors and allocates the exact CSR payload.
void allocate_rows(std::vector<std::size_t>& offsets, std::vector<LocalIndex>& values)
{
    std::size_t total = 0;
    for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
        const auto count = offsets[row];
        if (count > std::numeric_limits<std::size_t>::max() - total)
            mpi::fatal("Auxiliary connectivity size overflow");
        offsets[row] = total;
        total += count;
    }
    // LocalIndex limits individual IDs, not the total number of CSR entries.
    offsets.back() = total;
    values.resize(total);
}

/// @brief Restores starts after the offsets array has served as insertion cursors.
void restore_rows(std::vector<std::size_t>& offsets)
{
    // Each cursor now points to its row end; shift those ends into CSR positions.
    for (std::size_t row = offsets.size() - 1; row != 0; --row)
        offsets[row] = offsets[row - 1];
    offsets.front() = 0;
}

/// @brief Sorts each row without creating a second payload buffer.
void sort_rows(std::span<const std::size_t> offsets, std::vector<LocalIndex>& values)
{
    auto entries = std::span{values};
    for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
        // span iterators also support zero-length rows.
        auto slice = entries.subspan(offsets[row], offsets[row + 1] - offsets[row]);
        std::sort(slice.begin(), slice.end());
    }
}

} // namespace

/// @brief Inverts face-to-cell incidence into owned-cell rows using in-place cursors.
void build_cell_faces_conn(const MeshPart& m, std::vector<std::size_t>& offsets,
                           std::vector<LocalIndex>& faces)
{
    offsets.assign(static_cast<std::size_t>(m.n_own) + 1, 0);
    for (std::size_t f = 0; f < m.face_owner.size(); ++f) {
        ++offsets[static_cast<std::size_t>(m.face_owner[f])];
        const auto neighbour = m.face_neigh[f];
        if (neighbour >= 0 && neighbour < m.n_own) ++offsets[static_cast<std::size_t>(neighbour)];
    }
    allocate_rows(offsets, faces);
    // Face traversal already gives increasing IDs within every row.
    for (std::size_t f = 0; f < m.face_owner.size(); ++f) {
        faces[offsets[static_cast<std::size_t>(m.face_owner[f])]++] = static_cast<LocalIndex>(f);
        const auto neighbour = m.face_neigh[f];
        if (neighbour >= 0 && neighbour < m.n_own)
            faces[offsets[static_cast<std::size_t>(neighbour)]++] = static_cast<LocalIndex>(f);
    }
    restore_rows(offsets);
}

/// @brief Collects face neighbours for owned cells, skipping physical boundary faces.
void build_cell_cells_face_conn(const MeshPart& m, std::vector<std::size_t>& offsets,
                                std::vector<LocalIndex>& cells)
{
    offsets.assign(static_cast<std::size_t>(m.n_own) + 1, 0);
    for (std::size_t f = 0; f < m.face_owner.size(); ++f) {
        const auto neighbour = m.face_neigh[f];
        if (neighbour < 0) continue;
        ++offsets[static_cast<std::size_t>(m.face_owner[f])];
        if (neighbour < m.n_own) ++offsets[static_cast<std::size_t>(neighbour)];
    }
    allocate_rows(offsets, cells);
    // A cut face contributes to its local owner; its reverse belongs to the remote rank.
    for (std::size_t f = 0; f < m.face_owner.size(); ++f) {
        const auto owner = m.face_owner[f], neighbour = m.face_neigh[f];
        if (neighbour < 0) continue;
        cells[offsets[static_cast<std::size_t>(owner)]++] = neighbour;
        if (neighbour < m.n_own) cells[offsets[static_cast<std::size_t>(neighbour)]++] = owner;
    }
    restore_rows(offsets);
    sort_rows(offsets, cells);
}

/// @brief Inverts available cell connectivity for the owned-used node prefix.
void build_node_cells_conn(const MeshPart& m, std::vector<std::size_t>& offsets,
                           std::vector<LocalIndex>& cells)
{
    offsets.assign(static_cast<std::size_t>(m.n_nodes_own) + 1, 0);
    for (const auto node : m.cell_nodes)
        if (node < m.n_nodes_own) ++offsets[static_cast<std::size_t>(node)];
    allocate_rows(offsets, cells);
    // Include ghost cells, but omit rows for ghost-only nodes.
    for (std::size_t c = 0; c < m.cell_type.size(); ++c) {
        for (auto k = m.cell_nodes_offsets[c]; k < m.cell_nodes_offsets[c + 1]; ++k) {
            const auto node = m.cell_nodes[k];
            if (node < m.n_nodes_own)
                cells[offsets[static_cast<std::size_t>(node)]++] = static_cast<LocalIndex>(c);
        }
    }
    restore_rows(offsets);
}

/// @brief Inverts local face connectivity for the owned-used node prefix.
void build_node_faces_conn(const MeshPart& m, std::vector<std::size_t>& offsets,
                           std::vector<LocalIndex>& faces)
{
    offsets.assign(static_cast<std::size_t>(m.n_nodes_own) + 1, 0);
    for (const auto node : m.face_nodes)
        if (node < m.n_nodes_own) ++offsets[static_cast<std::size_t>(node)];
    allocate_rows(offsets, faces);
    // Each valid TRI/QUAD loop contains each vertex once, so no deduplication is needed.
    for (std::size_t f = 0; f < m.face_owner.size(); ++f) {
        for (auto k = m.face_nodes_offsets[f]; k < m.face_nodes_offsets[f + 1]; ++k) {
            const auto node = m.face_nodes[k];
            if (node < m.n_nodes_own)
                faces[offsets[static_cast<std::size_t>(node)]++] = static_cast<LocalIndex>(f);
        }
    }
    restore_rows(offsets);
}

/// @brief Counts and fills unique vertex neighbours using one LocalIndex stamp per local cell.
void build_cell_cells_node_conn(const MeshPart& m, std::span<const std::size_t> node_offsets,
                                std::span<const LocalIndex> node_cells,
                                std::vector<std::size_t>& offsets, std::vector<LocalIndex>& cells)
{
    assert(node_offsets.size() == static_cast<std::size_t>(m.n_nodes_own) + 1);
    const auto owned = static_cast<std::size_t>(m.n_own);
    offsets.assign(owned + 1, 0);
    std::vector<LocalIndex> stamp(static_cast<std::size_t>(m.n_cells), invalid_local_index);

    // Count first for an exact allocation, then repeat the same traversal to fill.
    for (int pass = 0; pass < 2; ++pass) {
        std::fill(stamp.begin(), stamp.end(), invalid_local_index);
        for (std::size_t c = 0; c < owned; ++c) {
            const auto tag = static_cast<LocalIndex>(c);
            auto cursor = pass == 0 ? std::size_t{0} : offsets[c];
            for (auto k = m.cell_nodes_offsets[c]; k < m.cell_nodes_offsets[c + 1]; ++k) {
                const auto node = static_cast<std::size_t>(m.cell_nodes[k]);
                for (auto j = node_offsets[node]; j < node_offsets[node + 1]; ++j) {
                    const auto other = node_cells[j];
                    auto& seen = stamp[static_cast<std::size_t>(other)];
                    if (other == tag || seen == tag) continue;
                    seen = tag;
                    if (pass != 0) cells[cursor] = other;
                    ++cursor;
                }
            }
            if (pass == 0) offsets[c] = cursor;
        }
        if (pass == 0) allocate_rows(offsets, cells);
    }
    sort_rows(offsets, cells);
}

/// @brief Builds the node transpose temporarily when it is not retained by the caller.
void build_cell_cells_node_conn(const MeshPart& m, std::vector<std::size_t>& offsets,
                                std::vector<LocalIndex>& cells)
{
    std::vector<std::size_t> node_offsets;
    std::vector<LocalIndex> node_cells;
    // Keep dependency storage local instead of activating an unrequested table.
    build_node_cells_conn(m, node_offsets, node_cells);
    build_cell_cells_node_conn(m, node_offsets, node_cells, offsets, cells);
}

/// @brief Adds requested missing tables without copying already computed CSR arrays.
void MeshAuxConnectivity::add_connectivity(const MeshPart& m, AuxConnType requested)
{
    if ((static_cast<std::uint64_t>(requested) & ~static_cast<std::uint64_t>(AuxConnType::All)) != 0)
        mpi::fatal("Unknown auxiliary connectivity flag");
    assert(active_mask == AuxConnType::None ||
           (n_cells_own == m.n_own && n_nodes_own == m.n_nodes_own && n_faces == m.n_faces));
    n_cells_own = m.n_own; n_nodes_own = m.n_nodes_own; n_faces = m.n_faces;

    // Build the node transpose first so a requested vertex stencil can reuse it.
    if (has_flag(requested, AuxConnType::NodeCells) && !has_node_cells()) {
        build_node_cells_conn(m, node_cells_offsets, node_cells);
        active_mask = active_mask | AuxConnType::NodeCells;
    }
    if (has_flag(requested, AuxConnType::CellFaces) && !has_cell_faces()) {
        build_cell_faces_conn(m, cell_faces_offsets, cell_faces);
        active_mask = active_mask | AuxConnType::CellFaces;
    }
    if (has_flag(requested, AuxConnType::CellCellsByFace) && !has_cell_cells_face()) {
        build_cell_cells_face_conn(m, cell_cells_face_offsets, cell_cells_face);
        active_mask = active_mask | AuxConnType::CellCellsByFace;
    }
    if (has_flag(requested, AuxConnType::NodeFaces) && !has_node_faces()) {
        build_node_faces_conn(m, node_faces_offsets, node_faces);
        active_mask = active_mask | AuxConnType::NodeFaces;
    }
    if (has_flag(requested, AuxConnType::CellCellsByNode) && !has_cell_cells_node()) {
        if (has_node_cells())
            build_cell_cells_node_conn(m, node_cells_offsets, node_cells, cell_cells_node_offsets, cell_cells_node);
        else
            build_cell_cells_node_conn(m, cell_cells_node_offsets, cell_cells_node);
        active_mask = active_mask | AuxConnType::CellCellsByNode;
    }
}

/// @brief Creates an empty cache and builds the requested local relations.
MeshAuxConnectivity build_aux_connectivity(const MeshPart& m, AuxConnType mask)
{
    MeshAuxConnectivity result;
    result.add_connectivity(m, mask);
    return result;
}

} // namespace krona::mesh::host
