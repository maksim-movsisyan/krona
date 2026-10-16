#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/boundary/geometry_view.hpp>

namespace krona::boundary::physical {

/// @brief Physical context built from borrowed condition-independent geometry.
template<class Space = execution::MemorySpace>
struct GeometryView {
    boundary::GeometryView<Space> storage;

    template<class Real>
    KOKKOS_INLINE_FUNCTION Context<Real> context(LocalIndex face, Real time = 0) const
    {
        const auto f = static_cast<std::size_t>(face);
        const auto owner = storage.face_owner(f);
        const auto i = static_cast<std::size_t>(owner);
        const GeometryReal x = storage.face_x(f), y = storage.face_y(f), z = storage.face_z(f);
        const GeometryReal nx = storage.normal_x(f), ny = storage.normal_y(f), nz = storage.normal_z(f);
        // Subtract in geometry precision, before any fluid-model conversion.
        const GeometryReal distance = (x - storage.cell_x(i)) * nx +
            (y - storage.cell_y(i)) * ny + (z - storage.cell_z(i)) * nz;
        Context<Real> c;
        c.face = face; c.owner = owner;
        c.boundary_index = static_cast<std::size_t>(face - storage.n_inner_faces);
        c.extended_index = extended_index(face);
        c.position = {x, y, z};
        c.displacement = {x - storage.cell_x(i), y - storage.cell_y(i), z - storage.cell_z(i)};
        c.tangential_displacement = {c.displacement.x - distance*nx,
                                    c.displacement.y - distance*ny, c.displacement.z - distance*nz};
        c.normal = {static_cast<Real>(nx), static_cast<Real>(ny), static_cast<Real>(nz)};
        c.distance = distance; c.area = storage.face_area(f); c.time = time;
        return c;
    }
    KOKKOS_INLINE_FUNCTION std::size_t extended_index(LocalIndex face) const
    { return storage.extended_index(face); }
};

} // namespace krona::boundary::physical
