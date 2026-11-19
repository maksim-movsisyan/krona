#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/mpi/error.hpp>

#include <Kokkos_NumericTraits.hpp>

namespace krona::boundary::detail {

/// @brief Builds one face context and passes it with the selected condition to a device callback.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Floating-point type of the face context.
/// @tparam Sink Device-callable object accepting (context, condition).
/// @tparam Condition Concrete condition type selected on the host for this patch.
template<class Space, class Real, class Sink, class Condition>
struct ForEachFace {
    GeometryView<Space> geometry;
    Sink sink;
    Condition condition;
    Real time = 0;

    /// @brief Runs the callback for one face in the dispatched patch.
    /// @param face Face ID in local mesh numbering.
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    { sink(geometry.template context<Real>(face, time), condition); }
};

/// @brief Counts faces with an invalid owner, orientation or geometry before boundary work begins.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Floating-point type in which boundary contexts will be built.
template<class Space, class Real>
struct ValidateGeometry {
    GeometryView<Space> geometry; ///< Borrowed arrays used to build the face context.
    LocalIndex n_own = 0;         ///< Size of the owned-cell range, starting at zero.

    /// @brief Adds one to the reduction count when this face fails validation.
    /// @param face Local ID of the boundary face to check.
    /// @param bad Running count of invalid faces in the reduction.
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, std::size_t& bad) const {
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = geometry.face_owner(f);
        // Check the owner before context() uses it to load the cell centroid.
        if (owner < 0 || owner >= n_own) { ++bad; return; }
        const auto c = geometry.template context<Real>(face);
        const Real norm2 = c.nx*c.nx + c.ny*c.ny + c.nz*c.nz;
        // The unit-length check allows for both storage and arithmetic rounding.
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

/// @brief Borrows the required mesh arrays and checks the boundary-face geometry on this rank.
/// @tparam Real Floating-point type used when evaluating the face geometry.
/// @tparam Exec Kokkos execution space running the geometry check.
/// @tparam Space Kokkos memory space of the borrowed mesh arrays.
/// @param exec Execution instance for the reduction. The host reads its count before returning.
/// @param mesh_view Grouped device mesh with face owners, centroids, normals and areas.
template<class Real, class Exec, class Space>
GeometryView<Space> attach_geometry(const Exec& exec, mesh::device::MeshView<Space> mesh_view) {
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    using mesh::device::MeshData;
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

    // Face IDs address the whole local face range. Only owned cell centroids are used here.
    const std::size_t faces = static_cast<std::size_t>(s.n_faces);
    const std::size_t owned = static_cast<std::size_t>(s.n_own);
    mpi::require(result.face_owner.extent(0) == faces && result.face_x.extent(0) == faces &&
                 result.face_y.extent(0) == faces && result.face_z.extent(0) == faces &&
                 result.normal_x.extent(0) == faces && result.normal_y.extent(0) == faces &&
                 result.normal_z.extent(0) == faces && result.face_area.extent(0) == faces &&
                 result.cell_x.extent(0) >= owned && result.cell_y.extent(0) >= owned &&
                 result.cell_z.extent(0) >= owned, "Boundary: geometry extent mismatch");

    // Check only faces on the domain boundary. MPI interface faces belong to the inner range.
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    std::size_t bad = 0;
    Kokkos::parallel_reduce("boundary/validate_geometry", Policy(exec, s.n_inner_faces, s.n_faces),
        ValidateGeometry<Space, Real>{result, s.n_own}, bad);
    mpi::require(bad == 0, "Boundary: invalid owner, unit normal, area or positive normal distance");
    return result;
}

} // namespace krona::boundary::detail
