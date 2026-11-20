#pragma once

#include <krona/mesh/device/aux_view.hpp>
#include <krona/transport/scalar/convection/types.hpp>

namespace krona::transport::scalar::convection::detail {

/// @brief Borrows the cell-to-face lists and face data used by the inner kernels.
/// @tparam Space Kokkos memory space where the mesh arrays live.
template<class Space>
struct StencilData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;

    Index owner, neighbour; ///< Cells on either side of an inner face.
    Index faces;            ///< Face IDs in the flattened cell-to-face lists.
    Offset offsets;         ///< Start of each owned cell's face list, plus the final end offset.
    Metric x, y, z;          ///< Face centre coordinates.
    Metric nx, ny, nz;      ///< Unit normals pointing out of the face owners.
    Metric area;            ///< Area of each face.
};

/// @brief Takes the mesh views needed for convection without copying the arrays.
/// @tparam Space Memory space shared by the mesh and connectivity.
/// @param mesh Mesh holding the face geometry and adjacent cell indices.
/// @param connectivity Cell-to-face lists for the same mesh.
template<class Space>
auto stencil_data(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> connectivity) {
    const auto& g = mesh.geometry;
    return StencilData<Space>{mesh.connectivity.face_owner, mesh.connectivity.face_neigh,
        connectivity.cell_faces, connectivity.cell_faces_offsets,
        g.face_centroid_x, g.face_centroid_y, g.face_centroid_z,
        g.face_normal_x, g.face_normal_y, g.face_normal_z, g.face_area};
}

/// @brief Reads the geometry and adjacent cells for a face between two cells.
/// @tparam Real Type used to evaluate the face flux.
/// @tparam Space Memory space of the stencil arrays.
/// @param data Borrowed mesh arrays.
/// @param face Local face index in the inner-face range.
/// @param time Current time passed to face providers.
template<class Real, class Space>
KOKKOS_INLINE_FUNCTION auto context(const StencilData<Space>& data, LocalIndex face, Real time) {
    const std::size_t f = static_cast<std::size_t>(face);
    return InteriorFaceContext<Real>{face, data.owner(f), data.neighbour(f),
        static_cast<Real>(data.x(f)), static_cast<Real>(data.y(f)), static_cast<Real>(data.z(f)),
        static_cast<Real>(data.nx(f)), static_cast<Real>(data.ny(f)), static_cast<Real>(data.nz(f)),
        static_cast<Real>(data.area(f)), time};
}

} // namespace krona::transport::scalar::convection::detail
