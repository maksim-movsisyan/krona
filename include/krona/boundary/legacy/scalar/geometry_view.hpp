#pragma once

#include <krona/boundary/scalar/conditions.hpp>
#include <krona/mesh/compute/mesh_view.hpp>

namespace krona::boundary::scalar {

/// @brief Compact borrowed descriptors; contains only geometry needed at exterior faces.
template<class Space = execution::MemorySpace>
struct GeometryView {
    using IndexArray = mesh::compute::detail::MeshArray<LocalIndex, Space, false>;
    using RealArray = mesh::compute::detail::MeshArray<GeometryReal, Space, false>;
    LocalIndex n_cells = 0, n_inner_faces = 0;
    IndexArray face_owner;
    RealArray cell_x, cell_y, cell_z;
    RealArray face_x, face_y, face_z;
    RealArray normal_x, normal_y, normal_z, face_area;

    template<class Real, class Diffusivity, class Correction = NoCorrection>
    KOKKOS_INLINE_FUNCTION Context<Real> context(LocalIndex face, const Diffusivity& diffusion,
                                                 const Correction& correction = {}, Real time = 0) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = face_owner(f);
        const std::size_t i = static_cast<std::size_t>(owner);
        Context<Real> c;
        c.face = face;
        c.owner = owner;
        c.boundary_index = static_cast<std::size_t>(face - n_inner_faces);
        c.x = static_cast<Real>(face_x(f)); c.y = static_cast<Real>(face_y(f)); c.z = static_cast<Real>(face_z(f));
        c.nx = static_cast<Real>(normal_x(f)); c.ny = static_cast<Real>(normal_y(f)); c.nz = static_cast<Real>(normal_z(f));
        const Real dx = c.x - static_cast<Real>(cell_x(i));
        const Real dy = c.y - static_cast<Real>(cell_y(i));
        const Real dz = c.z - static_cast<Real>(cell_z(i));
        c.distance = dx * c.nx + dy * c.ny + dz * c.nz;
        c.area = static_cast<Real>(face_area(f));
        c.diffusivity = static_cast<Real>(diffusion(f));
        c.time = time;
        c.tangential_increment = correction(i, dx - c.distance * c.nx,
                                            dy - c.distance * c.ny, dz - c.distance * c.nz);
        KOKKOS_ASSERT(Kokkos::isfinite(c.diffusivity) && c.diffusivity >= Real{0});
        return c;
    }

    KOKKOS_INLINE_FUNCTION std::size_t extended_index(LocalIndex face) const
    { return static_cast<std::size_t>(n_cells) + static_cast<std::size_t>(face - n_inner_faces); }
};

} // namespace krona::boundary::scalar
