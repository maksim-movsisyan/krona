#pragma once

#include <krona/fields/component.hpp>
#include <krona/mesh/device/aux_view.hpp>
#include <krona/numerics/gradient/types.hpp>

namespace krona::numerics::gradient::detail {

/// @brief Reads N input components and writes three derivative components for each of them.
/// @tparam T Scalar type stored in the input and output fields.
/// @tparam N Number of input components processed together.
/// @tparam Space Kokkos memory space used by the component descriptors.
template <class T, std::size_t N, class Space> 
struct Batch {
    Kokkos::Array<fields::ComponentDescriptor<const T, Space>, N> input;
    Kokkos::Array<fields::ComponentDescriptor<T, Space>, N> x, y, z;
};

/// @brief Borrows the mesh arrays and optional cached metrics used by the kernels.
/// @tparam Space Kokkos memory space of the mesh and auxiliary arrays.
template <class Space> 
struct StencilData {
    template <class T> using Array = mesh::device::detail::MeshArray<T, Space, false>;
    // Capture only gradient inputs; node tables, halo plans and unrelated metrics stay outside kernels.
    struct MeshDescriptor {
        mesh::device::MeshSizes sizes;
        struct {
            Array<LocalIndex> face_owner, face_neigh;
        } connectivity;
        struct {
            Array<GeometryReal> cell_centroid_x, cell_centroid_y, cell_centroid_z, cell_volume;
            Array<GeometryReal> face_centroid_x, face_centroid_y, face_centroid_z;
            Array<GeometryReal> face_normal_x, face_normal_y, face_normal_z, face_area;
        } geometry;
    } mesh;
    struct ConnectivityDescriptor {
        Array<std::size_t> cell_faces_offsets;
        Array<LocalIndex> cell_faces;
    } connectivity;
    struct GeometryDescriptor {
        bool vectors;
        Array<GeometryReal> face_cell_dist_x, face_cell_dist_y, face_cell_dist_z, face_interp_weight;
        KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_vector() const { return vectors; }
    } geometry;
    Options options;

    StencilData(mesh::device::MeshView<Space> m, mesh::device::AuxConnectivityView<Space> c,
            mesh::device::AuxGeometryView<Space> g, Options o)
        : mesh{m.sizes,
               {m.connectivity.face_owner, m.connectivity.face_neigh},
               {m.geometry.cell_centroid_x, m.geometry.cell_centroid_y, m.geometry.cell_centroid_z,
                m.geometry.cell_volume, m.geometry.face_centroid_x, m.geometry.face_centroid_y,
                m.geometry.face_centroid_z, m.geometry.face_normal_x, m.geometry.face_normal_y,
                m.geometry.face_normal_z, m.geometry.face_area}},
          connectivity{c.cell_faces_offsets, c.cell_faces},
          geometry{g.has_face_cell_dist_vector(), g.face_cell_dist_x, g.face_cell_dist_y, g.face_cell_dist_z,
                   g.face_interp_weight},
          options(o) {}

    /// @brief Finds the input field slot across a face from the current cell.
    KOKKOS_INLINE_FUNCTION std::size_t sample(LocalIndex cell, LocalIndex face) const {
        const std::size_t f = static_cast<std::size_t>(face);
        // The input stores physical-face values immediately after all local cells.
        if (face >= mesh.sizes.n_inner_faces)
            return static_cast<std::size_t>(mesh.sizes.n_cells) +
                        static_cast<std::size_t>(face - mesh.sizes.n_inner_faces);

        return static_cast<std::size_t>(mesh.connectivity.face_owner(f) == cell
                                            ? mesh.connectivity.face_neigh(f) : mesh.connectivity.face_owner(f));
    }

    /// @brief Gives least squares the vector from this cell centre to the sample point.
    /// @tparam Real Scalar type used to compute the displacement.
    template <class Real>
    KOKKOS_INLINE_FUNCTION void displacement(LocalIndex cell, LocalIndex face, Real d[3]) const {
        const std::size_t i = static_cast<std::size_t>(cell), f = static_cast<std::size_t>(face);
        const bool boundary = face >= mesh.sizes.n_inner_faces;
        if (geometry.has_face_cell_dist_vector()) {
            // The cached vector is in owner orientation. Flip it for the neighbour's fit.
            const Real sign = mesh.connectivity.face_owner(f) == cell ? Real{1} : Real{-1};
            d[0] = sign * static_cast<Real>(geometry.face_cell_dist_x(f));
            d[1] = sign * static_cast<Real>(geometry.face_cell_dist_y(f));
            d[2] = sign * static_cast<Real>(geometry.face_cell_dist_z(f));
        } else if (boundary) {
            // The boundary sample is at the face centroid.
            d[0] = (static_cast<Real>(mesh.geometry.face_centroid_x(f)) -
                    static_cast<Real>(mesh.geometry.cell_centroid_x(i)));
            d[1] = (static_cast<Real>(mesh.geometry.face_centroid_y(f)) -
                    static_cast<Real>(mesh.geometry.cell_centroid_y(i)));
            d[2] = (static_cast<Real>(mesh.geometry.face_centroid_z(f)) -
                    static_cast<Real>(mesh.geometry.cell_centroid_z(i)));
        } else {
            // Inside, the sample is at the opposite cell centre, including on MPI interfaces.
            const auto j = sample(cell, face);
            d[0] = static_cast<Real>(mesh.geometry.cell_centroid_x(j)) -
                   static_cast<Real>(mesh.geometry.cell_centroid_x(i));
            d[1] = static_cast<Real>(mesh.geometry.cell_centroid_y(j)) -
                   static_cast<Real>(mesh.geometry.cell_centroid_y(i));
            d[2] = static_cast<Real>(mesh.geometry.cell_centroid_z(j)) -
                   static_cast<Real>(mesh.geometry.cell_centroid_z(i));
        }
    }

    /// @brief Gives the owner's fraction in the Green-Gauss interior-face interpolation.
    /// @tparam Real Scalar type used for interpolation arithmetic.
    template <class Real> KOKKOS_INLINE_FUNCTION Real owner_weight(std::size_t face) const {
        return options.interpolation == Interpolation::AuxiliaryWeights
                   ? static_cast<Real>(geometry.face_interp_weight(face)) : Real{0.5};
    }
};

} // namespace krona::numerics::gradient::detail
