#pragma once

#include <krona/boundary/context.hpp>
#include <krona/math/vector3.hpp>
#include <krona/mesh/device/mesh_view.hpp>

namespace krona::boundary {

/// @brief Reads the mesh arrays needed to build a FaceContext for a boundary face.
/// @tparam Space Kokkos memory space holding the mesh arrays.
/// @details The view borrows the array storage. Keep the mesh buffers alive while the
/// scalar, physical or transport kernels use it.
template<class Space = execution::MemorySpace>
struct GeometryView {
    using memory_space = Space;
    using IndexArray = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using RealArray = mesh::device::detail::MeshArray<GeometryReal, Space, false>;

    LocalIndex n_cells = 0;       ///< Owned and MPI ghost cells before the boundary slots in an extended field.
    LocalIndex n_inner_faces = 0; ///< First boundary face ID when faces are grouped.
    IndexArray face_owner;        ///< Adjacent owned cell indexed by local face ID.

    // Centroid arrays use local cell IDs for cells and local face IDs for faces.
    RealArray cell_x, cell_y, cell_z;
    RealArray face_x, face_y, face_z;

    // Outward unit normal components and area indexed by local face ID.
    RealArray normal_x, normal_y, normal_z, face_area;

    /// @brief Reads the face geometry and computes the normal distance and tangential offset.
    /// @tparam Real Floating-point type of the context and geometric calculations.
    /// @param face Local ID of a face on the boundary of the computational domain.
    /// @param time Time copied into the context for use by conditions and profiles.
    template<class Real>
    KOKKOS_INLINE_FUNCTION FaceContext<Real> context(LocalIndex face, Real time = 0) const {
        const std::size_t f = static_cast<std::size_t>(face);
        FaceContext<Real> c;
        c.face = face;
        c.owner = face_owner(f);
        const std::size_t i = static_cast<std::size_t>(c.owner);

        // Boundary-only arrays start at zero. Extended fields put the same face after all cells.
        c.boundary_index = static_cast<std::size_t>(face - n_inner_faces);
        c.extended_index = static_cast<std::size_t>(n_cells) + c.boundary_index;

        c.x = static_cast<Real>(face_x(f));
        c.y = static_cast<Real>(face_y(f));
        c.z = static_cast<Real>(face_z(f));
        c.nx = static_cast<Real>(normal_x(f));
        c.ny = static_cast<Real>(normal_y(f));
        c.nz = static_cast<Real>(normal_z(f));

        // Split the centre-to-face displacement into normal and tangential parts.
        const math::Vector3<Real> displacement{
            c.x-static_cast<Real>(cell_x(i)), c.y-static_cast<Real>(cell_y(i)),
            c.z-static_cast<Real>(cell_z(i))};
        const math::Vector3<Real> normal{c.nx, c.ny, c.nz};
        c.distance = displacement.dot(normal);
        const auto tangent = displacement-normal*c.distance;
        c.tx = tangent.x; c.ty = tangent.y; c.tz = tangent.z;
        c.area = static_cast<Real>(face_area(f));
        c.time = time;
        return c;
    }
};

} // namespace krona::boundary
