#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/mesh/host/mesh_part.hpp>
#include <krona/transport/scalar/diffusion/types.hpp>

#include <KokkosKernels_ArithTraits.hpp>

#include <cmath>

namespace krona::transport::scalar::diffusion::detail {

template<class Space, class Real>
struct ValidateMetric {
    mesh::device::MeshView<Space> mesh;
    mesh::device::AuxGeometryView<Space> geometry;
    DiffusionOptions options;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, std::size_t& bad) const
    {
        const auto f = static_cast<std::size_t>(face);
        const auto i = static_cast<std::size_t>(mesh.connectivity.face_owner(f));
        const auto neighbour = mesh.connectivity.face_neigh(f);
        const auto j = neighbour < 0 ? std::size_t{0} : static_cast<std::size_t>(neighbour);
        const auto& g = mesh.geometry;
        const Real dx = static_cast<Real>((neighbour < 0 ? g.face_centroid_x(f) : g.cell_centroid_x(j)) - g.cell_centroid_x(i));
        const Real dy = static_cast<Real>((neighbour < 0 ? g.face_centroid_y(f) : g.cell_centroid_y(j)) - g.cell_centroid_y(i));
        const Real dz = static_cast<Real>((neighbour < 0 ? g.face_centroid_z(f) : g.cell_centroid_z(j)) - g.cell_centroid_z(i));
        const Real nx = static_cast<Real>(g.face_normal_x(f)), ny = static_cast<Real>(g.face_normal_y(f)), nz = static_cast<Real>(g.face_normal_z(f));
        const Real h = dx*nx + dy*ny + dz*nz;
        const Real cached = static_cast<Real>(geometry.face_normal_dist(f));
        const Real area_over_distance = static_cast<Real>(geometry.face_area_over_normal_dist(f));
        const Real tolerance = Real{128}*(static_cast<Real>(KokkosKernels::ArithTraits<GeometryReal>::epsilon()) +
                                                        KokkosKernels::ArithTraits<Real>::epsilon());
        bool valid = Kokkos::isfinite(h) && h > Real{0} && Kokkos::isfinite(cached) &&
            Kokkos::abs(cached-h) <= tolerance*h && Kokkos::isfinite(area_over_distance) && area_over_distance > Real{0} &&
            Kokkos::abs(area_over_distance*h-static_cast<Real>(g.face_area(f))) <= tolerance*static_cast<Real>(g.face_area(f));
        if (geometry.has_face_interp_weights()) {
            const Real left = (static_cast<Real>(g.face_centroid_x(f))-static_cast<Real>(g.cell_centroid_x(i)))*nx +
                              (static_cast<Real>(g.face_centroid_y(f))-static_cast<Real>(g.cell_centroid_y(i)))*ny +
                              (static_cast<Real>(g.face_centroid_z(f))-static_cast<Real>(g.cell_centroid_z(i)))*nz;
            const Real w = static_cast<Real>(geometry.face_interp_weight(f));
            const Real expected = neighbour < 0 ? Real{0} : (h-left)/h;
            valid = valid && Kokkos::isfinite(w) && w >= Real{0} && w <= Real{1} &&
                    Kokkos::abs(w-expected) <= tolerance;
        }
        if (geometry.has_face_non_ortho()) {
            const Real area = static_cast<Real>(g.face_area(f));
            const Real tx = static_cast<Real>(geometry.face_non_ortho_x(f));
            const Real ty = static_cast<Real>(geometry.face_non_ortho_y(f));
            const Real tz = static_cast<Real>(geometry.face_non_ortho_z(f));
            const Real normal_scale = (nx*nx+ny*ny+nz*nz)/h;
            valid = valid && Kokkos::isfinite(tx) && Kokkos::isfinite(ty) && Kokkos::isfinite(tz) &&
                Kokkos::abs(tx-area*(nx-normal_scale*dx)) <= tolerance*area*(Real{1}+Kokkos::abs(normal_scale*dx)) &&
                Kokkos::abs(ty-area*(ny-normal_scale*dy)) <= tolerance*area*(Real{1}+Kokkos::abs(normal_scale*dy)) &&
                Kokkos::abs(tz-area*(nz-normal_scale*dz)) <= tolerance*area*(Real{1}+Kokkos::abs(normal_scale*dz));
        }
        if (options.require_orthogonal) {
            const Real tx = dx-h*nx, ty = dy-h*ny, tz = dz-h*nz;
            const Real tangential = Kokkos::sqrt(tx*tx+ty*ty+tz*tz);
            valid = valid && Kokkos::isfinite(tangential) && tangential <= static_cast<Real>(options.orthogonality_tolerance)*h;
        }
        if (!valid) ++bad;
    }
};

template<class Exec, class Real>
void validate_plan(const mesh::host::MeshPart& host,
    mesh::device::MeshView<typename Exec::memory_space> mesh_view,
    mesh::device::AuxConnectivityView<typename Exec::memory_space> connectivity,
    mesh::device::AuxGeometryView<typename Exec::memory_space> geometry,
    mpi::CommView comm, const Exec& exec, DiffusionOptions options)
{
    mpi::guarded([&] {
        execution::require_kokkos();
        const auto& s = mesh_view.sizes;
        mpi::require(host.rank == comm.rank() && host.nprocs == comm.size(),
                     "Scalar diffusion: mesh/communicator mismatch");
        mpi::require(host.faces_grouped && s.faces_grouped && host.n_own == s.n_own && host.n_cells == s.n_cells &&
            host.n_faces == s.n_faces && host.n_internal_faces == s.n_internal_faces && host.n_inner_faces == s.n_inner_faces,
            "Scalar diffusion: grouped host/view mesh mismatch");
        mpi::require(std::isfinite(options.orthogonality_tolerance) && options.orthogonality_tolerance >= 0 &&
                     options.orthogonality_tolerance < 1, "Scalar diffusion: invalid orthogonality tolerance");
        using mesh::device::MeshData;
        namespace validation = mesh::device::validation;
        constexpr auto metrics = mesh::AuxGeomType::FaceNormalDistance | mesh::AuxGeomType::FaceAreaOverNormalDistance;
        // Existing shared validators check shapes, face intervals and complete cell-face incidence.
        const auto status = validation::validate<MeshData::FaceCells | MeshData::FaceGeometry,
            mesh::AuxConnType::CellFaces, metrics, true>(exec, mesh_view, connectivity, geometry);
        mpi::require(status.ok(), "Scalar diffusion: invalid mesh, cell-face incidence or normal metrics");
        const auto cells = static_cast<std::size_t>(s.n_cells);
        const auto owned = static_cast<std::size_t>(s.n_own);
        const auto& g = mesh_view.geometry;
        mpi::require(mesh_view.has(MeshData::CellGeometry) && g.cell_centroid_x.extent(0) >= cells &&
            g.cell_centroid_y.extent(0) >= cells && g.cell_centroid_z.extent(0) >= cells && g.cell_volume.extent(0) >= owned,
            "Scalar diffusion: attach cell centres and owned volumes");
        mpi::require(!geometry.has_face_interp_weights() || geometry.face_interp_weight.extent(0) == static_cast<std::size_t>(s.n_faces),
                     "Scalar diffusion: interpolation weight extent mismatch");
        const auto faces = static_cast<std::size_t>(s.n_faces);
        mpi::require(!geometry.has_face_non_ortho() || (geometry.face_non_ortho_x.extent(0) == faces &&
            geometry.face_non_ortho_y.extent(0) == faces && geometry.face_non_ortho_z.extent(0) == faces),
            "Scalar diffusion: nonorthogonality vector extent mismatch");
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
        std::size_t bad = 0;
        Kokkos::parallel_reduce("diffusion/validate_normal_metric", Policy(exec, 0, s.n_faces),
            ValidateMetric<typename Exec::memory_space, Real>{mesh_view, geometry, options}, bad);
        mpi::require(bad == 0, "Scalar diffusion: stale normal metrics or nonorthogonal geometry");
        Kokkos::parallel_reduce("diffusion/validate_owned_cells", Policy(exec, 0, s.n_own),
            validation::ValidateCells<MeshData::CellGeometry, mesh::AuxGeomType::None, typename Exec::memory_space>{mesh_view, {}}, bad);
        mpi::require(bad == 0, "Scalar diffusion: invalid owned cell geometry");

    });
}

} // namespace krona::transport::scalar::diffusion::detail