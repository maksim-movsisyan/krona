#pragma once

#include <krona/mesh/device/aux_view.hpp>
#include <krona/transport/scalar/convection/types.hpp>

namespace krona::transport::scalar::convection::detail {

/// @brief Only arrays used by the convection row kernels; all descriptors borrow storage.
template<class Space>
struct RowData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;
    LocalIndex n_inner_faces = 0;
    Index owner, neighbour, faces;
    Offset offsets;
    Metric x, y, z, nx, ny, nz, area;
};

template<class Space>
auto row_data(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> connectivity) {
    const auto& g = mesh.geometry;
    return RowData<Space>{mesh.sizes.n_inner_faces, mesh.connectivity.face_owner, mesh.connectivity.face_neigh,
        connectivity.cell_faces, connectivity.cell_faces_offsets,
        g.face_centroid_x, g.face_centroid_y, g.face_centroid_z,
        g.face_normal_x, g.face_normal_y, g.face_normal_z, g.face_area};
}

template<class Real, class Space>
KOKKOS_INLINE_FUNCTION auto context(const RowData<Space>& d, LocalIndex face, Real time) {
    const std::size_t f = static_cast<std::size_t>(face);
    return InteriorFaceContext<Real>{face, d.owner(f), d.neighbour(f),
        static_cast<Real>(d.x(f)), static_cast<Real>(d.y(f)), static_cast<Real>(d.z(f)),
        static_cast<Real>(d.nx(f)), static_cast<Real>(d.ny(f)), static_cast<Real>(d.nz(f)),
        static_cast<Real>(d.area(f)), time};
}

} // namespace krona::transport::scalar::convection::detail
