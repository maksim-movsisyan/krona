#pragma once

#include <krona/boundary/context.hpp>
#include <krona/mesh/compute/mesh_view.hpp>

namespace krona::boundary {

/// @brief Borrowed geometry shared by scalar and physical boundary operations.
template<class Space = execution::MemorySpace>
struct GeometryView {
    using memory_space = Space;
    using IndexArray = mesh::compute::detail::MeshArray<LocalIndex, Space, false>;
    using RealArray = mesh::compute::detail::MeshArray<GeometryReal, Space, false>;

    LocalIndex n_cells = 0, n_inner_faces = 0;
    IndexArray face_owner;
    RealArray cell_x, cell_y, cell_z;
    RealArray face_x, face_y, face_z;
    RealArray normal_x, normal_y, normal_z, face_area;

    template<class Real>
    KOKKOS_INLINE_FUNCTION FaceContext<Real> context(LocalIndex face, Real time = 0) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        FaceContext<Real> c;
        c.face = face;
        c.owner = face_owner(f);
        const std::size_t i = static_cast<std::size_t>(c.owner);
        c.boundary_index = static_cast<std::size_t>(face - n_inner_faces);
        c.extended_index = static_cast<std::size_t>(n_cells) + c.boundary_index;
        c.x = static_cast<Real>(face_x(f));
        c.y = static_cast<Real>(face_y(f));
        c.z = static_cast<Real>(face_z(f));
        c.nx = static_cast<Real>(normal_x(f));
        c.ny = static_cast<Real>(normal_y(f));
        c.nz = static_cast<Real>(normal_z(f));
        const Real dx = c.x - static_cast<Real>(cell_x(i));
        const Real dy = c.y - static_cast<Real>(cell_y(i));
        const Real dz = c.z - static_cast<Real>(cell_z(i));
        c.distance = dx*c.nx + dy*c.ny + dz*c.nz;
        c.tx = dx - c.distance*c.nx;
        c.ty = dy - c.distance*c.ny;
        c.tz = dz - c.distance*c.nz;
        c.area = static_cast<Real>(face_area(f));
        c.time = time;
        return c;
    }
};

} // namespace krona::boundary
