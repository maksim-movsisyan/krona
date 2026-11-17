#pragma once

#include <krona/mesh/device/detail/validation/cells_and_nodes.hpp>
#include <krona/mesh/device/detail/validation/csr.hpp>
#include <krona/mesh/device/detail/validation/element_types.hpp>
#include <krona/mesh/device/detail/validation/faces.hpp>
#include <krona/mesh/device/detail/validation/halo.hpp>
#include <krona/mesh/device/detail/validation/incidence.hpp>
#include <krona/mesh/device/detail/validation/shapes.hpp>

namespace krona::mesh::device::validation {

/// @brief Counts failures by category on this rank.
/// Each entity or CSR row is counted once per check, even if several conditions fail.
/// The shapes counter records missing groups, invalid sizes or descriptors, and bad tolerances.
struct Result {
    std::size_t shapes = 0, cells = 0, faces = 0, nodes = 0;
    std::size_t cell_nodes = 0, face_nodes = 0;
    std::size_t cell_faces = 0, cell_cells_face = 0, cell_cells_node = 0;
    std::size_t node_cells = 0, node_faces = 0;
    std::size_t halo_send = 0, halo_recv = 0;
    std::size_t cell_incidence = 0, face_incidence = 0, node_incidence = 0;

    /// @brief Reports whether every validation counter is zero.
    [[nodiscard]] bool ok() const noexcept {
        return shapes == 0 && cells == 0 && faces == 0 && nodes == 0 && cell_nodes == 0 && face_nodes == 0 &&
               cell_faces == 0 && cell_cells_face == 0 && cell_cells_node == 0 && node_cells == 0 &&
               node_faces == 0 && halo_send == 0 && halo_recv == 0 && cell_incidence == 0 &&
               face_incidence == 0 && node_incidence == 0;
    }
};

/// @brief Checks the selected arrays on this rank and waits for the error counts (so the call is NOT async).
/// @tparam Data Base mesh groups to check, for example MeshData::FiniteVolume.
/// @tparam Connectivity Auxiliary adjacency groups to check, for example AuxConnType::CellFaces.
/// @tparam Geometry Auxiliary metric groups to check, for example AuxGeomType::WallDistance.
/// @tparam Compare Also check that related mesh data agree. For example, if
/// both CellFaces and FaceCells are selected, check that each face listed
/// for a cell actually touches that cell.
/// @tparam Exec Kokkos execution space used to run the checks.
/// @tparam Space Kokkos memory space where the views' arrays live. Exec must be able to access it.
/// @param exec Execution-space instance on which to run the checks. Finish any writes
/// to the input arrays submitted to a different instance before calling.
/// @param mesh Base mesh view. Keep its backing arrays alive for the call.
/// @param conn Connectivity view for the same mesh. Leave empty when Connectivity is None.
/// @param geom Geometry view for the same mesh. Leave empty when Geometry is None.
/// @param tolerance Thresholds for face normal and metric comparisons.
/// @return Local counts of failures. If shapes is nonzero, no kernels were launched.
///
/// Example for a mesh attached with the default MeshOptions:
/// @code
/// const auto result = validation::validate<MeshData::FiniteVolume>(exec, storage.view());
/// @endcode
///
/// This call performs no MPI communication. Halo peer metadata, repeated receive
/// destinations and element-specific topology have separate checks.
template <MeshData Data = MeshData::All, AuxConnType Connectivity = AuxConnType::None,
          AuxGeomType Geometry = AuxGeomType::None, bool Compare = false, class Exec, class Space>
[[nodiscard]] Result validate(const Exec& exec, MeshView<Space> mesh, AuxConnectivityView<Space> conn = {},
                              AuxGeometryView<Space> geom = {}, Tolerance tolerance = {}) {
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    Result result;

    // Check the descriptors on the host before a kernel can index their arrays.
    result.shapes = detail::check_shapes<Data, Connectivity, Geometry>(mesh, conn, geom);
    if (!detail::valid_tolerance(tolerance))
        ++result.shapes;
    if (result.shapes != 0)
        return result;

    const std::size_t cells = static_cast<std::size_t>(mesh.sizes.n_cells);
    const std::size_t own = static_cast<std::size_t>(mesh.sizes.n_own);
    const std::size_t faces = static_cast<std::size_t>(mesh.sizes.n_faces);
    const std::size_t nodes = static_cast<std::size_t>(mesh.sizes.n_nodes);
    const std::size_t own_nodes = static_cast<std::size_t>(mesh.sizes.n_nodes_own);

    // These reductions check individual entities and the face links used by later checks.
    if constexpr (has(Data, MeshData::CellGeometry) || has(Geometry, AuxGeomType::WallDistance))
        result.cells = detail::reduce("mesh/validate_cells", exec, cells,
                                      ValidateCells<Data, Geometry, Space>{mesh, geom});
    if constexpr (has(Data, MeshData::FaceCells) || has(Data, MeshData::FaceGeometry) ||
                  (static_cast<std::uint64_t>(Geometry) &
                   static_cast<std::uint64_t>(AuxGeomType::AllFaces)) != 0)
        result.faces = detail::reduce("mesh/validate_faces", exec, faces,
                                      ValidateFaces<Data, Geometry, Compare, Space>{mesh, geom, tolerance});
    if constexpr (has(Data, MeshData::NodeCoordinates))
        result.nodes = detail::reduce("mesh/validate_nodes", exec, nodes, ValidateNodes<Data, Space>{mesh});

    // Each CSR table has its own row count, ID range and ordering rules.
    auto csr = [&]<RowOrder Order, bool Self, bool Unique>(const char* label, auto offsets, auto values,
                                                           std::size_t rows, std::size_t bound) {
        return detail::reduce(label, exec, rows + 1,
                              ValidateCsr<Order, Self, Unique, decltype(offsets), decltype(values)>{
                                  offsets, values, rows, bound});
    };
    const auto& a = mesh.connectivity;
    if constexpr (has(Data, MeshData::CellNodes))
        result.cell_nodes = csr.template operator()<RowOrder::Unordered, false, true>(
            "mesh/validate_cell_nodes", a.cell_nodes_offsets, a.cell_nodes, cells, nodes);
    if constexpr (has(Data, MeshData::FaceNodes))
        result.face_nodes = csr.template operator()<RowOrder::Unordered, false, true>(
            "mesh/validate_face_nodes", a.face_nodes_offsets, a.face_nodes, faces, nodes);
    if constexpr (has(Connectivity, AuxConnType::CellFaces))
        result.cell_faces = csr.template operator()<RowOrder::Strict, false, false>(
            "mesh/validate_cell_faces", conn.cell_faces_offsets, conn.cell_faces, own, faces);
    if constexpr (has(Connectivity, AuxConnType::CellCellsByFace))
        result.cell_cells_face = csr.template operator()<RowOrder::Nondecreasing, true, false>(
            "mesh/validate_cell_cells_face", conn.cell_cells_face_offsets, conn.cell_cells_face, own, cells);
    if constexpr (has(Connectivity, AuxConnType::CellCellsByNode))
        result.cell_cells_node = csr.template operator()<RowOrder::Strict, true, false>(
            "mesh/validate_cell_cells_node", conn.cell_cells_node_offsets, conn.cell_cells_node, own, cells);
    if constexpr (has(Connectivity, AuxConnType::NodeCells))
        result.node_cells = csr.template operator()<RowOrder::Strict, false, false>(
            "mesh/validate_node_cells", conn.node_cells_offsets, conn.node_cells, own_nodes, cells);
    if constexpr (has(Connectivity, AuxConnType::NodeFaces))
        result.node_faces = csr.template operator()<RowOrder::Strict, false, false>(
            "mesh/validate_node_faces", conn.node_faces_offsets, conn.node_faces, own_nodes, faces);

    // Send indices refer to owned cells and receive indices refer to ghosts.
    if constexpr (has(Data, MeshData::Halo)) {
        result.halo_send =
            detail::reduce("mesh/validate_halo_send", exec, mesh.halo.send_owned_local.extent(0),
                           ValidateIndices<decltype(mesh.halo.send_owned_local)>{mesh.halo.send_owned_local,
                                                                                 0, mesh.sizes.n_own});
        result.halo_recv =
            detail::reduce("mesh/validate_halo_recv", exec, mesh.halo.recv_ghost_local.extent(0),
                           ValidateIndices<decltype(mesh.halo.recv_ghost_local)>{
                               mesh.halo.recv_ghost_local, mesh.sizes.n_own, mesh.sizes.n_cells});
    }

    // These checks follow links between tables, so wait until all earlier checks pass.
    if constexpr (Compare) {
        if (result.ok()) {
            if constexpr (has(Data, MeshData::CellNodes) ||
                          (has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)))
                result.cell_incidence =
                    detail::reduce("mesh/validate_cell_incidence", exec, own,
                                   ValidateCellIncidence<Data, Connectivity, Space>{mesh, conn});
            if constexpr ((has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)) ||
                          (has(Data, MeshData::FaceCells) && has(Data, MeshData::FaceNodes) &&
                           has(Data, MeshData::CellNodes)) ||
                          (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces)))
                result.face_incidence =
                    detail::reduce("mesh/validate_face_incidence", exec, faces,
                                   ValidateFaceIncidence<Data, Connectivity, Space>{mesh, conn});
            if constexpr (has(Connectivity, AuxConnType::NodeCells) ||
                          (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces)))
                result.node_incidence =
                    detail::reduce("mesh/validate_node_incidence", exec, own_nodes,
                                   ValidateNodeIncidence<Data, Connectivity, Space>{mesh, conn});
        }
    }
    return result;
}

} // namespace krona::mesh::device::validation
