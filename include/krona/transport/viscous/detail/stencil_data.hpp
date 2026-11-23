#pragma once

#include <krona/mesh/device/aux_view.hpp>
#include <krona/transport/detail/component_pack.hpp>
#include <krona/transport/viscous/detail/face_data.hpp>

namespace krona::transport::viscous::detail {

/// @brief Geometry of an inner face oriented from owner to neighbour.
template<class R>
struct InnerGeometry {
    Vector<R> normal;     ///< Unit normal pointing from owner to neighbour.
    Vector<R> connection; ///< Displacement from the owner centroid to the neighbour centroid.
    Vector<R> skew;       ///< Displacement from the clamped interpolation point to the face centroid.
    R height = 0;         ///< Centre-to-centre distance projected onto the normal (connection.dot(normal)).
    R owner_weight = 0;   ///< Linear interpolation weight of the owner (the neighbour gets 1 - owner_weight).
};

/// @brief Borrows the device geometry and adjacency used by the owned-row kernels.
template<class Space>
struct StencilData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;

    Index owner, neighbour, faces;
    Offset offsets;
    Metric area, nx, ny, nz;
    Metric cell_x, cell_y, cell_z, face_x, face_y, face_z;

    template<class R>
    KOKKOS_INLINE_FUNCTION InnerGeometry<R> inner_geometry(std::size_t face, std::size_t left,
                                                           std::size_t right) const {
        const Vector<GeometryReal> n{nx(face), ny(face), nz(face)};
        const Vector<GeometryReal> d{cell_x(right) - cell_x(left), cell_y(right) - cell_y(left),
                                     cell_z(right) - cell_z(left)};
        const Vector<GeometryReal> xf{face_x(face) - cell_x(left), face_y(face) - cell_y(left),
                                      face_z(face) - cell_z(left)};
        // Clamp the projected base interpolation to the segment. The skew correction
        // then spans the full displacement from this point to the actual face centre.
        const auto height = d.dot(n);
        if (!positive(height)) return {};
        const auto fraction = Kokkos::max(GeometryReal{0},
            Kokkos::min(GeometryReal{1}, xf.dot(n) / height));
        return {math::cast<R>(n), math::cast<R>(d), math::cast<R>(xf - d * fraction),
                static_cast<R>(height), static_cast<R>(1 - fraction)};
    }
};

template<class Space>
auto stencil_data(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> connectivity) {
    const auto& g = mesh.geometry;
    return StencilData<Space>{mesh.connectivity.face_owner, mesh.connectivity.face_neigh,
        connectivity.cell_faces, connectivity.cell_faces_offsets, g.face_area,
        g.face_normal_x, g.face_normal_y, g.face_normal_z, g.cell_centroid_x, g.cell_centroid_y,
        g.cell_centroid_z, g.face_centroid_x, g.face_centroid_y, g.face_centroid_z};
}

/// @brief Reads the p,u,v,w,T pack and gradients laid out as 3*primitive + axis.
template<class Values, class Derivatives>
struct Fields {
    Values values;
    Derivatives derivatives;

    template<class R>
    KOKKOS_INLINE_FUNCTION State<R> state(std::size_t cell) const {
        return transport::detail::read_primitive<R>(values, cell);
    }

    template<class R>
    KOKKOS_INLINE_FUNCTION Gradients<R> gradients(std::size_t cell) const {
        return {gradient<R>(cell, 3), gradient<R>(cell, 6), gradient<R>(cell, 9), gradient<R>(cell, 12)};
    }

private:
    template<class R>
    KOKKOS_INLINE_FUNCTION Vector<R> gradient(std::size_t cell, int first) const {
        return {static_cast<R>(derivatives(cell, first)), static_cast<R>(derivatives(cell, first + 1)),
                static_cast<R>(derivatives(cell, first + 2))};
    }
};

} // namespace krona::transport::viscous::detail
