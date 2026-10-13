#pragma once

#include <krona/mesh/compute/mesh_view.hpp>

#include <limits>

namespace krona::boundary {

/// @brief Borrowed face-owner geometry; contains no scalar or physical-condition data.
template<class Space = execution::MemorySpace>
struct GeometryView {
    using IndexArray = mesh::compute::detail::MeshArray<LocalIndex, Space, false>;
    using RealArray = mesh::compute::detail::MeshArray<GeometryReal, Space, false>;
    LocalIndex n_cells = 0, n_inner_faces = 0;
    IndexArray face_owner;
    RealArray cell_x, cell_y, cell_z;
    RealArray face_x, face_y, face_z;
    RealArray normal_x, normal_y, normal_z, face_area;

    KOKKOS_INLINE_FUNCTION std::size_t extended_index(LocalIndex face) const
    { return static_cast<std::size_t>(n_cells) + static_cast<std::size_t>(face - n_inner_faces); }
};

namespace detail {
template<class Space, class Real>
struct ValidateGeometry {
    GeometryView<Space> geometry;
    LocalIndex n_own = 0;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, std::size_t& bad) const
    {
        const auto f = static_cast<std::size_t>(face);
        const auto owner = geometry.face_owner(f);
        if (owner < 0 || owner >= n_own) { ++bad; return; }
        const auto i = static_cast<std::size_t>(owner);
        const Real x = static_cast<Real>(geometry.face_x(f)), y = static_cast<Real>(geometry.face_y(f));
        const Real z = static_cast<Real>(geometry.face_z(f));
        const Real cx = static_cast<Real>(geometry.cell_x(i)), cy = static_cast<Real>(geometry.cell_y(i));
        const Real cz = static_cast<Real>(geometry.cell_z(i));
        const Real nx = static_cast<Real>(geometry.normal_x(f)), ny = static_cast<Real>(geometry.normal_y(f));
        const Real nz = static_cast<Real>(geometry.normal_z(f));
        const Real area = static_cast<Real>(geometry.face_area(f));
        const Real norm2 = nx*nx + ny*ny + nz*nz;
        constexpr Real tolerance = Real{128} * static_cast<Real>(std::numeric_limits<GeometryReal>::epsilon()) +
                                   Real{128} * std::numeric_limits<Real>::epsilon();
        const Real h = (x-cx)*nx + (y-cy)*ny + (z-cz)*nz;
        if (!Kokkos::isfinite(x) || !Kokkos::isfinite(y) || !Kokkos::isfinite(z) ||
            !Kokkos::isfinite(cx) || !Kokkos::isfinite(cy) || !Kokkos::isfinite(cz) ||
            !Kokkos::isfinite(norm2) || Kokkos::abs(norm2-Real{1}) > tolerance ||
            !Kokkos::isfinite(h) || !(h > Real{0}) || !Kokkos::isfinite(area) || !(area > Real{0})) ++bad;
    }
};

} // namespace detail
} // namespace krona::boundary
