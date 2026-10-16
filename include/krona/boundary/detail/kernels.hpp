#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/mpi/error.hpp>

#include <Kokkos_NumericTraits.hpp>

namespace krona::boundary::detail {

template<class Space, class Real>
struct ValidateGeometry {
    GeometryView<Space> geometry;
    LocalIndex n_own = 0;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, std::size_t& bad) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = geometry.face_owner(f);
        if (owner < 0 || owner >= n_own) { ++bad; return; }
        const auto c = geometry.template context<Real>(face);
        const Real norm2 = c.nx*c.nx + c.ny*c.ny + c.nz*c.nz;
        constexpr Real tolerance = Real{128} *
            (static_cast<Real>(Kokkos::epsilon_v<GeometryReal>) +
            Kokkos::epsilon_v<Real>);
        if (!Kokkos::isfinite(c.x) || !Kokkos::isfinite(c.y) || !Kokkos::isfinite(c.z) ||
            !Kokkos::isfinite(c.tx) || !Kokkos::isfinite(c.ty) || !Kokkos::isfinite(c.tz) ||
            !Kokkos::isfinite(norm2) || Kokkos::abs(norm2 - Real{1}) > tolerance ||
            !Kokkos::isfinite(c.distance) || !(c.distance > Real{0}) ||
            !Kokkos::isfinite(c.area) || !(c.area > Real{0})) ++bad;
    }
};

/// @brief Delivers geometry and a concrete condition type; no relation or material lookup is imposed.
template<class Space, class Real, class Sink, class Condition>
struct ForEachFace {
    GeometryView<Space> geometry;
    Sink sink;
    Condition condition;
    Real time = 0;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    { sink(geometry.template context<Real>(face, time), condition); }
};

/// @brief Validates borrowed exterior geometry once and returns its compact descriptor.
template<class Real, class Exec, class Space>
GeometryView<Space> attach_geometry(const Exec& exec, mesh::compute::MeshView<Space> mesh_view)
{
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    using mesh::compute::MeshData;
    const auto& s = mesh_view.sizes;
    mpi::require(s.faces_grouped && s.n_own >= 0 && s.n_cells >= s.n_own &&
                 s.n_internal_faces >= 0 && s.n_internal_faces <= s.n_inner_faces &&
                 s.n_inner_faces <= s.n_faces, "Boundary: invalid counts or face grouping");
    mpi::require(mesh_view.has(MeshData::FaceCells | MeshData::Geometry),
                 "Boundary: attach face connectivity and cell/face geometry");
    const auto& g = mesh_view.geometry;
    GeometryView<Space> result{s.n_cells, s.n_inner_faces, mesh_view.connectivity.face_owner,
        g.cell_centroid_x, g.cell_centroid_y, g.cell_centroid_z,
        g.face_centroid_x, g.face_centroid_y, g.face_centroid_z,
        g.face_normal_x, g.face_normal_y, g.face_normal_z, g.face_area};
    const std::size_t faces = static_cast<std::size_t>(s.n_faces);
    const std::size_t owned = static_cast<std::size_t>(s.n_own);
    mpi::require(result.face_owner.extent(0) == faces && result.face_x.extent(0) == faces &&
                 result.face_y.extent(0) == faces && result.face_z.extent(0) == faces &&
                 result.normal_x.extent(0) == faces && result.normal_y.extent(0) == faces &&
                 result.normal_z.extent(0) == faces && result.face_area.extent(0) == faces &&
                 result.cell_x.extent(0) >= owned && result.cell_y.extent(0) >= owned &&
                 result.cell_z.extent(0) >= owned, "Boundary: geometry extent mismatch");
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    std::size_t bad = 0;
    Kokkos::parallel_reduce("boundary/validate_geometry", Policy(exec, s.n_inner_faces, s.n_faces),
        ValidateGeometry<Space, Real>{result, s.n_own}, bad);
    mpi::require(bad == 0, "Boundary: invalid owner, unit normal, area or positive normal distance");
    return result;
}

} // namespace krona::boundary::detail
