#pragma once

#include <krona/mesh/device/detail/validation/common.hpp>
#include <krona/mesh/device/aux_view.hpp>

namespace krona::mesh::device::validation {

namespace detail {

/// @brief Looks for a local ID in one already checked adjacency row.
/// @tparam Offsets Kokkos view storing row starts and the final offset.
/// @tparam Values Kokkos view storing the local IDs.
/// @tparam Id Type of the ID being looked up, usually LocalIndex.
template <class Offsets, class Values, class Id>
KOKKOS_INLINE_FUNCTION bool contains(Offsets offsets, Values values, std::size_t row, Id id) {
    for (std::size_t j = offsets(row); j < offsets(row + 1); ++j)
        if (values(j) == id)
            return true;
    return false;
}

} // namespace detail

/// @brief Checks that the selected tables agree on the faces, nodes and neighbours of a cell.
/// @tparam Data Base mesh groups describing cell nodes or the cells on a face.
/// @tparam Connectivity Auxiliary adjacency groups to compare with those base arrays.
/// @tparam Space Kokkos memory space of the mesh and connectivity views.
/// Run over owned cells after validating the selected CSR tables and indices.
template <MeshData Data, AuxConnType Connectivity, class Space>
struct ValidateCellIncidence {
    MeshView<Space> mesh;
    AuxConnectivityView<Space> conn;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell, std::size_t& bad) const {
        const auto& a = mesh.connectivity;
        bool valid = true;

        // Nodes of an owned cell belong to the prefix used by owned cells.
        if constexpr (has(Data, MeshData::CellNodes))
            for (std::size_t j = a.cell_nodes_offsets(cell); j < a.cell_nodes_offsets(cell + 1); ++j)
                valid &= a.cell_nodes(j) < mesh.sizes.n_nodes_own;

        // Every listed face must touch this cell. If present, the neighbour row must agree.
        if constexpr (has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)) {
            for (std::size_t j = conn.cell_faces_offsets(cell); j < conn.cell_faces_offsets(cell + 1); ++j) {
                const auto face = static_cast<std::size_t>(conn.cell_faces(j));
                const auto owner = a.face_owner(face), neighbour = a.face_neigh(face);
                const auto id = static_cast<LocalIndex>(cell);
                valid &= owner == id || neighbour == id;
                if constexpr (has(Connectivity, AuxConnType::CellCellsByFace)) {
                    const auto other = owner == id ? neighbour : owner;
                    if (other >= 0)
                        valid &=
                            detail::contains(conn.cell_cells_face_offsets, conn.cell_cells_face, cell, other);
                }
            }

            if constexpr (has(Connectivity, AuxConnType::CellCellsByFace)) {
                // A cell pair can share several faces. Compare the number of faces with the number of entries.
                for (std::size_t j = conn.cell_cells_face_offsets(cell);
                     j < conn.cell_cells_face_offsets(cell + 1); ++j) {
                    const auto other = conn.cell_cells_face(j);
                    std::size_t expected = 0, actual = 0;
                    for (std::size_t k = conn.cell_faces_offsets(cell); k < conn.cell_faces_offsets(cell + 1);
                         ++k) {
                        const auto f = static_cast<std::size_t>(conn.cell_faces(k));
                        const auto id = static_cast<LocalIndex>(cell);
                        if ((a.face_owner(f) == id && a.face_neigh(f) == other) ||
                            (a.face_neigh(f) == id && a.face_owner(f) == other))
                            ++expected;
                    }
                    for (std::size_t k = conn.cell_cells_face_offsets(cell);
                         k < conn.cell_cells_face_offsets(cell + 1); ++k)
                        if (conn.cell_cells_face(k) == other)
                            ++actual;
                    valid &= expected == actual;
                }
            }
        }

        // Check the reverse link from each of the cell's nodes back to the cell.
        if constexpr (has(Data, MeshData::CellNodes) && has(Connectivity, AuxConnType::NodeCells)) {
            for (std::size_t j = a.cell_nodes_offsets(cell); j < a.cell_nodes_offsets(cell + 1); ++j) {
                const auto node = a.cell_nodes(j);
                if (node < mesh.sizes.n_nodes_own)
                    valid &= detail::contains(conn.node_cells_offsets, conn.node_cells,
                                              static_cast<std::size_t>(node), static_cast<LocalIndex>(cell));
            }
        }

        if constexpr (has(Data, MeshData::CellNodes) && has(Connectivity, AuxConnType::CellCellsByNode)) {
            // Each listed neighbour must share at least one node with this cell.
            for (std::size_t j = conn.cell_cells_node_offsets(cell);
                 j < conn.cell_cells_node_offsets(cell + 1); ++j) {
                const auto other = static_cast<std::size_t>(conn.cell_cells_node(j));
                bool shared = false;
                for (std::size_t k = a.cell_nodes_offsets(cell); k < a.cell_nodes_offsets(cell + 1); ++k)
                    shared |= detail::contains(a.cell_nodes_offsets, a.cell_nodes, other, a.cell_nodes(k));
                valid &= shared;
            }

            // When the reverse table is available, check that no neighbour is missing.
            if constexpr (has(Connectivity, AuxConnType::NodeCells))
                for (std::size_t j = a.cell_nodes_offsets(cell); j < a.cell_nodes_offsets(cell + 1); ++j) {
                    const auto node = static_cast<std::size_t>(a.cell_nodes(j));
                    if (node >= static_cast<std::size_t>(mesh.sizes.n_nodes_own))
                        continue;
                    for (std::size_t k = conn.node_cells_offsets(node); k < conn.node_cells_offsets(node + 1);
                         ++k)
                        if (conn.node_cells(k) != static_cast<LocalIndex>(cell))
                            valid &= detail::contains(conn.cell_cells_node_offsets, conn.cell_cells_node,
                                                      cell, conn.node_cells(k));
                }
        }

        if (!valid)
            ++bad;
    }
};

/// @brief Checks that the selected tables agree on which cells and nodes belong to a face.
/// @tparam Data Base mesh groups giving the cells and nodes on a face and the nodes of each cell.
/// @tparam Connectivity Auxiliary cell-to-face and node-to-face groups to compare, if selected.
/// @tparam Space Kokkos memory space of the mesh and connectivity views.
/// Run over all local faces after validating the selected CSR tables and indices.
template <MeshData Data, AuxConnType Connectivity, class Space>
struct ValidateFaceIncidence {
    MeshView<Space> mesh;
    AuxConnectivityView<Space> conn;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t face, std::size_t& bad) const {
        const auto& a = mesh.connectivity;
        bool valid = true;

        // The owner and any owned neighbour must list the face in their rows.
        if constexpr (has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)) {
            valid &=
                detail::contains(conn.cell_faces_offsets, conn.cell_faces,
                                 static_cast<std::size_t>(a.face_owner(face)), static_cast<LocalIndex>(face));
            const auto n = a.face_neigh(face);
            if (n >= 0 && n < mesh.sizes.n_own)
                valid &= detail::contains(conn.cell_faces_offsets, conn.cell_faces,
                                          static_cast<std::size_t>(n), static_cast<LocalIndex>(face));
        }

        // Each face node must occur in both connected cells when there is a neighbour.
        if constexpr ((has(Data, MeshData::FaceCells) && has(Data, MeshData::FaceNodes) &&
                       has(Data, MeshData::CellNodes))) {
            const auto o = static_cast<std::size_t>(a.face_owner(face));
            const auto n = a.face_neigh(face);
            for (std::size_t j = a.face_nodes_offsets(face); j < a.face_nodes_offsets(face + 1); ++j) {
                const auto node = a.face_nodes(j);
                valid &= detail::contains(a.cell_nodes_offsets, a.cell_nodes, o, node);
                if (n >= 0)
                    valid &= detail::contains(a.cell_nodes_offsets, a.cell_nodes, static_cast<std::size_t>(n),
                                              node);
            }
        }

        // Check the reverse link from each face node with a NodeFaces row.
        if constexpr (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces)) {
            for (std::size_t j = a.face_nodes_offsets(face); j < a.face_nodes_offsets(face + 1); ++j) {
                const auto node = a.face_nodes(j);
                if (node < mesh.sizes.n_nodes_own)
                    valid &= detail::contains(conn.node_faces_offsets, conn.node_faces,
                                              static_cast<std::size_t>(node), static_cast<LocalIndex>(face));
            }
        }

        if (!valid)
            ++bad;
    }
};

/// @brief Checks that the cells and faces listed for a node actually use it.
/// @tparam Data Base mesh groups giving the nodes of each cell or face.
/// @tparam Connectivity Auxiliary node-to-cell and node-to-face groups to check.
/// @tparam Space Kokkos memory space of the mesh and connectivity views.
/// Run over nodes used by owned cells after validating the selected CSR tables and indices.
template <MeshData Data, AuxConnType Connectivity, class Space>
struct ValidateNodeIncidence {
    MeshView<Space> mesh;
    AuxConnectivityView<Space> conn;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t node, std::size_t& bad) const {
        const auto& a = mesh.connectivity;
        bool valid = true;

        // The row belongs to the owned-used prefix, so at least one listed cell must be owned.
        if constexpr (has(Connectivity, AuxConnType::NodeCells)) {
            bool owned_user = false;
            for (std::size_t j = conn.node_cells_offsets(node); j < conn.node_cells_offsets(node + 1); ++j)
                owned_user |= conn.node_cells(j) < mesh.sizes.n_own;
            valid &= owned_user;
        }

        // Follow each listed cell or face back to its node row.
        if constexpr (has(Data, MeshData::CellNodes) && has(Connectivity, AuxConnType::NodeCells))
            for (std::size_t j = conn.node_cells_offsets(node); j < conn.node_cells_offsets(node + 1); ++j)
                valid &= detail::contains(a.cell_nodes_offsets, a.cell_nodes,
                                          static_cast<std::size_t>(conn.node_cells(j)),
                                          static_cast<LocalIndex>(node));
        if constexpr (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces))
            for (std::size_t j = conn.node_faces_offsets(node); j < conn.node_faces_offsets(node + 1); ++j)
                valid &= detail::contains(a.face_nodes_offsets, a.face_nodes,
                                          static_cast<std::size_t>(conn.node_faces(j)),
                                          static_cast<LocalIndex>(node));

        if (!valid)
            ++bad;
    }
};

} // namespace krona::mesh::device::validation
