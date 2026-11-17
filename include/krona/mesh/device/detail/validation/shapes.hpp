#pragma once

#include <krona/mesh/device/detail/validation/common.hpp>
#include <krona/mesh/device/aux_view.hpp>

namespace krona::mesh::device::validation {

namespace detail {

/// @brief Checks the array length and whether a nonempty view has a data pointer.
/// @tparam View Type of the Kokkos array view being checked.
template <class View>
bool shape(View v, std::size_t count) {
    return v.extent(0) == count && (count == 0 || v.data() != nullptr);
}

/// @brief Checks that a CSR table has one offset per row plus a final offset.
/// @tparam Offsets Kokkos view containing the row offsets.
/// @tparam Values Kokkos view containing the local IDs in those rows.
template <class Offsets, class Values>
bool csr_shape(Offsets offsets, Values values, std::size_t rows) {
    return rows != std::numeric_limits<std::size_t>::max() && shape(offsets, rows + 1) &&
           (values.extent(0) == 0 || values.data() != nullptr);
}

/// @brief Checks on the host that selected arrays are present and have the expected sizes.
/// @tparam Data Base mesh groups to check.
/// @tparam Connectivity Auxiliary adjacency groups to check.
/// @tparam Geometry Auxiliary metric groups to check.
/// @tparam Space Kokkos memory space of all three views.
template <MeshData Data, AuxConnType Connectivity, AuxGeomType Geometry, class Space>
std::size_t check_shapes(MeshView<Space> mesh, AuxConnectivityView<Space> conn, AuxGeometryView<Space> geom) {
    static_assert((static_cast<std::uint64_t>(Data) & ~static_cast<std::uint64_t>(MeshData::All)) == 0);
    static_assert((static_cast<std::uint64_t>(Connectivity) & ~static_cast<std::uint64_t>(AuxConnType::All)) == 0);
    static_assert((static_cast<std::uint64_t>(Geometry) & ~static_cast<std::uint64_t>(AuxGeomType::All)) == 0);

    if (!valid_sizes(mesh.sizes))
        return 1;
    std::size_t bad = 0;
    auto check = [&](bool valid) {
        if (!valid)
            ++bad;
    };
    check(mesh.has(Data));
    check(conn.has(Connectivity));
    check(geom.has(Geometry));

    const std::size_t cells = static_cast<std::size_t>(mesh.sizes.n_cells);
    const std::size_t own = static_cast<std::size_t>(mesh.sizes.n_own);
    const std::size_t faces = static_cast<std::size_t>(mesh.sizes.n_faces);
    const std::size_t nodes = static_cast<std::size_t>(mesh.sizes.n_nodes);
    const std::size_t own_nodes = static_cast<std::size_t>(mesh.sizes.n_nodes_own);

    // Check that the auxiliary caches describe the same counts as the base mesh.
    if constexpr (Connectivity != AuxConnType::None)
        check(conn.n_faces == mesh.sizes.n_faces && conn.n_cells_own == mesh.sizes.n_own &&
              conn.n_nodes_own == mesh.sizes.n_nodes_own);
    if constexpr ((static_cast<std::uint64_t>(Geometry) &
                   static_cast<std::uint64_t>(AuxGeomType::AllFaces)) != 0)
        check(geom.n_faces == mesh.sizes.n_faces && mesh.sizes.faces_grouped);
    if constexpr (has(Geometry, AuxGeomType::WallDistance))
        check(geom.n_cells == mesh.sizes.n_cells);

    // Base arrays cover all local entities, including ghosts when present.
    const auto& a = mesh.connectivity;
    const auto& g = mesh.geometry;
    if constexpr (has(Data, MeshData::FaceCells)) {
        check(shape(a.face_owner, faces));
        check(shape(a.face_neigh, faces));
    }
    if constexpr (has(Data, MeshData::CellNodes)) {
        check(shape(a.cell_type, cells));
        check(csr_shape(a.cell_nodes_offsets, a.cell_nodes, cells));
    }
    if constexpr (has(Data, MeshData::FaceNodes)) {
        check(shape(a.face_type, faces));
        check(shape(a.face_local_index, faces));
        check(csr_shape(a.face_nodes_offsets, a.face_nodes, faces));
    }
    if constexpr (has(Data, MeshData::NodeCoordinates)) {
        check(shape(g.node_x, nodes));
        check(shape(g.node_y, nodes));
        check(shape(g.node_z, nodes));
    }
    if constexpr (has(Data, MeshData::CellGeometry)) {
        check(shape(g.cell_centroid_x, cells));
        check(shape(g.cell_centroid_y, cells));
        check(shape(g.cell_centroid_z, cells));
        check(shape(g.cell_volume, cells));
    }
    if constexpr (has(Data, MeshData::FaceGeometry)) {
        check(shape(g.face_centroid_x, faces));
        check(shape(g.face_centroid_y, faces));
        check(shape(g.face_centroid_z, faces));
        check(shape(g.face_normal_x, faces));
        check(shape(g.face_normal_y, faces));
        check(shape(g.face_normal_z, faces));
        check(shape(g.face_area, faces));
    }
    if constexpr (has(Data, MeshData::Halo)) {
        check(mesh.halo.send_owned_local.extent(0) == 0 || mesh.halo.send_owned_local.data() != nullptr);
        check(mesh.halo.recv_ghost_local.extent(0) == 0 || mesh.halo.recv_ghost_local.data() != nullptr);
    }

    // Adjacency caches have a row per owned cell or per node used by an owned cell.
    if constexpr (has(Connectivity, AuxConnType::CellFaces))
        check(csr_shape(conn.cell_faces_offsets, conn.cell_faces, own));
    if constexpr (has(Connectivity, AuxConnType::CellCellsByFace))
        check(csr_shape(conn.cell_cells_face_offsets, conn.cell_cells_face, own));
    if constexpr (has(Connectivity, AuxConnType::CellCellsByNode))
        check(csr_shape(conn.cell_cells_node_offsets, conn.cell_cells_node, own));
    if constexpr (has(Connectivity, AuxConnType::NodeCells))
        check(csr_shape(conn.node_cells_offsets, conn.node_cells, own_nodes));
    if constexpr (has(Connectivity, AuxConnType::NodeFaces))
        check(csr_shape(conn.node_faces_offsets, conn.node_faces, own_nodes));

    // Face metrics cover every local face. Wall distances cover every local cell.
    if constexpr (has(Geometry, AuxGeomType::FaceCellDistance))
        check(shape(geom.face_cell_dist, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceInv))
        check(shape(geom.face_cell_dist_inv, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance))
        check(shape(geom.face_normal_dist, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceNormalDistanceInv))
        check(shape(geom.face_normal_dist_inv, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceAreaOverNormalDistance))
        check(shape(geom.face_area_over_normal_dist, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceInterpolationWeights))
        check(shape(geom.face_interp_weight, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceVector)) {
        check(shape(geom.face_cell_dist_x, faces));
        check(shape(geom.face_cell_dist_y, faces));
        check(shape(geom.face_cell_dist_z, faces));
    }
    if constexpr (has(Geometry, AuxGeomType::FaceNonOrthogonalityVector)) {
        check(shape(geom.face_non_ortho_x, faces));
        check(shape(geom.face_non_ortho_y, faces));
        check(shape(geom.face_non_ortho_z, faces));
    }
    if constexpr (has(Geometry, AuxGeomType::WallDistance))
        check(shape(geom.cell_wall_dist, cells));

    return bad;
}

} // namespace detail

} // namespace krona::mesh::device::validation
