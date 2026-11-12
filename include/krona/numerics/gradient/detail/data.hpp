#pragma once

#include <krona/fields/component.hpp>
#include <krona/numerics/gradient/types.hpp>

namespace krona::numerics::gradient::detail {

template<class T, std::size_t N, class Space>
struct Batch {
    Kokkos::Array<fields::ComponentDescriptor<const T, Space>, N> input;
    Kokkos::Array<fields::ComponentDescriptor<T, Space>, N> x, y, z;
};

template<class Space>
struct Metrics {
    template<class T> using Array = mesh::device::detail::MeshArray<T, Space, false>;
    // Capture only gradient inputs; node tables, halo plans and unrelated metrics stay outside kernels.
    struct MeshDescriptor {
        mesh::device::MeshSizes sizes;
        mesh::device::MeshData active_mask;
        struct { Array<LocalIndex> face_owner, face_neigh; } connectivity;
        struct {
            Array<GeometryReal> cell_centroid_x, cell_centroid_y, cell_centroid_z, cell_volume;
            Array<GeometryReal> face_centroid_x, face_centroid_y, face_centroid_z;
            Array<GeometryReal> face_normal_x, face_normal_y, face_normal_z, face_area;
        } geometry;
        bool has(mesh::device::MeshData flags) const
        {
            const std::uint64_t bits = static_cast<std::uint64_t>(flags);
            return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
        }
    } mesh;
    struct ConnectivityDescriptor {
        LocalIndex n_cells_own, n_faces;
        bool ready;
        Array<std::size_t> cell_faces_offsets;
        Array<LocalIndex> cell_faces;
        bool has_cell_faces() const { return ready; }
    } connectivity;
    struct GeometryDescriptor {
        LocalIndex n_faces;
        bool vectors, weights;
        Array<GeometryReal> face_cell_dist_x, face_cell_dist_y, face_cell_dist_z, face_interp_weight;
        KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_vector() const { return vectors; }
        bool has_face_interp_weights() const { return weights; }
    } geometry;
    Options options;

    Metrics(mesh::device::MeshView<Space> m, mesh::device::AuxConnectivityView<Space> c,
            mesh::device::AuxGeometryView<Space> g, Options o)
        : mesh{m.sizes, m.active_mask, {m.connectivity.face_owner, m.connectivity.face_neigh},
               {m.geometry.cell_centroid_x, m.geometry.cell_centroid_y, m.geometry.cell_centroid_z, m.geometry.cell_volume,
                m.geometry.face_centroid_x, m.geometry.face_centroid_y, m.geometry.face_centroid_z,
                m.geometry.face_normal_x, m.geometry.face_normal_y, m.geometry.face_normal_z, m.geometry.face_area}},
          connectivity{c.n_cells_own, c.n_faces, c.has_cell_faces(), c.cell_faces_offsets, c.cell_faces},
          geometry{g.n_faces, g.has_face_cell_dist_vector(), g.has_face_interp_weights(),
                   g.face_cell_dist_x, g.face_cell_dist_y, g.face_cell_dist_z, g.face_interp_weight}, options(o) {}

    KOKKOS_INLINE_FUNCTION std::size_t sample(LocalIndex cell, LocalIndex face) const {
        const std::size_t f = static_cast<std::size_t>(face);
        if (face >= mesh.sizes.n_inner_faces)
            return static_cast<std::size_t>(mesh.sizes.n_cells) + static_cast<std::size_t>(face - mesh.sizes.n_inner_faces);
        return static_cast<std::size_t>(mesh.connectivity.face_owner(f) == cell
            ? mesh.connectivity.face_neigh(f) : mesh.connectivity.face_owner(f));
    }

    template<class Real>
    KOKKOS_INLINE_FUNCTION void displacement(LocalIndex cell, LocalIndex face, Real d[3]) const {
        const std::size_t i = static_cast<std::size_t>(cell), f = static_cast<std::size_t>(face);
        const bool boundary = face >= mesh.sizes.n_inner_faces;
        if (geometry.has_face_cell_dist_vector()) {
            const Real sign = mesh.connectivity.face_owner(f) == cell ? Real{1} : Real{-1};
            d[0] = sign * static_cast<Real>(geometry.face_cell_dist_x(f));
            d[1] = sign * static_cast<Real>(geometry.face_cell_dist_y(f));
            d[2] = sign * static_cast<Real>(geometry.face_cell_dist_z(f));
        } else if (boundary) {
            d[0] = (static_cast<Real>(mesh.geometry.face_centroid_x(f)) - static_cast<Real>(mesh.geometry.cell_centroid_x(i)));
            d[1] = (static_cast<Real>(mesh.geometry.face_centroid_y(f)) - static_cast<Real>(mesh.geometry.cell_centroid_y(i)));
            d[2] = (static_cast<Real>(mesh.geometry.face_centroid_z(f)) - static_cast<Real>(mesh.geometry.cell_centroid_z(i)));
        } else {
            const auto j = sample(cell, face);
            d[0] = static_cast<Real>(mesh.geometry.cell_centroid_x(j)) - static_cast<Real>(mesh.geometry.cell_centroid_x(i));
            d[1] = static_cast<Real>(mesh.geometry.cell_centroid_y(j)) - static_cast<Real>(mesh.geometry.cell_centroid_y(i));
            d[2] = static_cast<Real>(mesh.geometry.cell_centroid_z(j)) - static_cast<Real>(mesh.geometry.cell_centroid_z(i));
        }
    }

    template<class Real>
    KOKKOS_INLINE_FUNCTION Real owner_weight(std::size_t face) const {
        return options.interpolation == Interpolation::AuxiliaryWeights ? static_cast<Real>(geometry.face_interp_weight(face)) : Real{0.5};
    }
};

} // namespace krona::numerics::gradient::detail
