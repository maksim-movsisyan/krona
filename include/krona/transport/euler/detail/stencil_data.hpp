#pragma once

#include <krona/mesh/device/aux_view.hpp>

namespace krona::transport::euler::detail {

/// @brief Borrows cell-to-face adjacency, adjacent cell indices and surface metrics.
/// @tparam Space Kokkos memory space holding the mesh arrays.
template<class Space>
struct StencilData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;

    Index owner, neighbour; ///< Cell indices in the direction of the stored face normal.
    Index faces;            ///< Local face IDs in the owned-cell adjacency list.
    Offset offsets;         ///< Start of each owned cell's face list, plus its final end.
    Metric nx, ny, nz;      ///< Unit normal pointing from owner to neighbour.
    Metric area;            ///< Area used to integrate flux density over the face.
};

/// @brief Captures the mesh views needed for an inner-face row kernel.
/// @tparam Space Memory space of the borrowed mesh and connectivity.
/// @param mesh Device mesh with grouped face connectivity, normals and areas.
/// @param connectivity CellFaces adjacency for the same owned-cell numbering.
template<class Space>
auto stencil_data(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> connectivity) {
    return StencilData<Space>{mesh.connectivity.face_owner, mesh.connectivity.face_neigh,
        connectivity.cell_faces, connectivity.cell_faces_offsets, mesh.geometry.face_normal_x,
        mesh.geometry.face_normal_y, mesh.geometry.face_normal_z, mesh.geometry.face_area};
}

} // namespace krona::transport::euler::detail
