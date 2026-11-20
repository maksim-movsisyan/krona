#pragma once

#include <krona/fields/component.hpp>
#include <krona/mesh/device/aux_view.hpp>
#include <krona/numerics/gradient/types.hpp>

namespace krona::numerics::gradient::detail {

/// @brief Reads N input components and writes three derivative components for each of them.
/// @tparam T Scalar type stored in the input and output fields.
/// @tparam N Number of input components processed together.
/// @tparam Space Kokkos memory space used by the component descriptors.
template <class T, std::size_t N, class Space> struct Batch {
    Kokkos::Array<fields::ComponentDescriptor<const T, Space>, N> input;
    Kokkos::Array<fields::ComponentDescriptor<T, Space>, N> x, y, z;
};

/// @brief Borrows the mesh arrays and optional cached metrics used by the kernels.
/// @tparam Space Kokkos memory space of the mesh and auxiliary arrays.
template <class Space>
struct StencilData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;

    // Boundary samples follow the owned and ghost cells in the input field.
    LocalIndex n_cells;
    LocalIndex n_inner_faces;

    // Face-to-cell indices and the CSR list of faces for each owned cell.
    Index face_owner, face_neigh;
    Offset cell_faces_offsets;
    Index cell_faces;

    // Base geometry for Green-Gauss and for least squares without cached displacements.
    Metric cell_centroid_x, cell_centroid_y, cell_centroid_z, cell_volume;
    Metric face_centroid_x, face_centroid_y, face_centroid_z;
    Metric face_normal_x, face_normal_y, face_normal_z, face_area;

    // Optional displacements point from the owner to the opposite sample.
    Metric face_cell_dist_x, face_cell_dist_y, face_cell_dist_z;
    Metric face_interp_weight; ///< Optional interpolation weight of the face owner.
    bool has_cached_displacement;
    Options options;

    /// @brief Borrows the views needed by the gradient kernels without copying their arrays.
    /// @param mesh Local cell counts, geometry and face-to-cell indices.
    /// @param connectivity Cell-to-face lists for that mesh.
    /// @param geometry Optional cached displacements and interpolation weights.
    /// @param gradient_options Gradient scheme, interpolation and least-squares settings.
    StencilData(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> connectivity,
                mesh::device::AuxGeometryView<Space> geometry, Options gradient_options)
        : n_cells(mesh.sizes.n_cells), n_inner_faces(mesh.sizes.n_inner_faces),
          face_owner(mesh.connectivity.face_owner), face_neigh(mesh.connectivity.face_neigh),
          cell_faces_offsets(connectivity.cell_faces_offsets), cell_faces(connectivity.cell_faces),
          cell_centroid_x(mesh.geometry.cell_centroid_x), cell_centroid_y(mesh.geometry.cell_centroid_y),
          cell_centroid_z(mesh.geometry.cell_centroid_z), cell_volume(mesh.geometry.cell_volume),
          face_centroid_x(mesh.geometry.face_centroid_x), face_centroid_y(mesh.geometry.face_centroid_y),
          face_centroid_z(mesh.geometry.face_centroid_z), face_normal_x(mesh.geometry.face_normal_x),
          face_normal_y(mesh.geometry.face_normal_y), face_normal_z(mesh.geometry.face_normal_z),
          face_area(mesh.geometry.face_area), face_cell_dist_x(geometry.face_cell_dist_x),
          face_cell_dist_y(geometry.face_cell_dist_y), face_cell_dist_z(geometry.face_cell_dist_z),
          face_interp_weight(geometry.face_interp_weight), has_cached_displacement(geometry.has_face_cell_dist_vector()),
          options(gradient_options) {}

    /// @brief Finds the input field slot across a face from the current cell.
    KOKKOS_INLINE_FUNCTION std::size_t sample(LocalIndex cell, LocalIndex face) const {
        const std::size_t f = static_cast<std::size_t>(face);
        // The input stores physical-face values immediately after all local cells.
        if (face >= n_inner_faces)
            return static_cast<std::size_t>(n_cells) + static_cast<std::size_t>(face - n_inner_faces);
        const auto owner = face_owner(f);
        return static_cast<std::size_t>(owner == cell ? face_neigh(f) : owner);
    }

    /// @brief Gives least squares the vector from this cell centre to the sample point.
    /// @tparam Real Scalar type used to compute the displacement.
    template <class Real>
    KOKKOS_INLINE_FUNCTION void displacement(LocalIndex cell, LocalIndex face, Real d[3]) const {
        const std::size_t i = static_cast<std::size_t>(cell), f = static_cast<std::size_t>(face);
        const bool boundary = face >= n_inner_faces;
        if (has_cached_displacement) {
            // The cached vector is in owner orientation. Flip it for the neighbour's fit.
            const Real sign = face_owner(f) == cell ? Real{1} : Real{-1};
            d[0] = sign * static_cast<Real>(face_cell_dist_x(f));
            d[1] = sign * static_cast<Real>(face_cell_dist_y(f));
            d[2] = sign * static_cast<Real>(face_cell_dist_z(f));
        } else if (boundary) {
            // The boundary sample is at the face centroid.
            d[0] = static_cast<Real>(face_centroid_x(f)) - static_cast<Real>(cell_centroid_x(i));
            d[1] = static_cast<Real>(face_centroid_y(f)) - static_cast<Real>(cell_centroid_y(i));
            d[2] = static_cast<Real>(face_centroid_z(f)) - static_cast<Real>(cell_centroid_z(i));
        } else {
            // Inside, the sample is at the opposite cell centre, including on MPI interfaces.
            const auto j = sample(cell, face);
            d[0] = static_cast<Real>(cell_centroid_x(j)) - static_cast<Real>(cell_centroid_x(i));
            d[1] = static_cast<Real>(cell_centroid_y(j)) - static_cast<Real>(cell_centroid_y(i));
            d[2] = static_cast<Real>(cell_centroid_z(j)) - static_cast<Real>(cell_centroid_z(i));
        }
    }

    /// @brief Gives the owner's fraction in the Green-Gauss interior-face interpolation.
    /// @tparam Real Scalar type used for interpolation arithmetic.
    template <class Real> KOKKOS_INLINE_FUNCTION Real owner_weight(std::size_t face) const {
        return options.interpolation == Interpolation::AuxiliaryWeights
                   ? static_cast<Real>(face_interp_weight(face))
                   : Real{0.5};
    }
};

} // namespace krona::numerics::gradient::detail
