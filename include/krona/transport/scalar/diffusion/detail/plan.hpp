#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/transport/scalar/diffusion/types.hpp>
#include <krona/transport/scalar/diffusion/detail/geometry.hpp>

#include <Kokkos_NumericTraits.hpp>

#include <cmath>
#include <type_traits>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Checks primary geometry and compares any supplied caches against it.
/// @tparam Space Memory space of the mesh arrays.
/// @tparam Real Arithmetic type of the operator consuming the metrics.
template<class Space, class Real>
struct ValidateMetric {
    mesh::device::MeshView<Space> mesh;
    mesh::device::AuxGeometryView<Space> geometry;
    DiffusionOptions options;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, std::size_t& bad) const {
        using MetricReal = std::common_type_t<Real, GeometryReal>;
        using Vector = math::Vector3<MetricReal>;
        const auto f = static_cast<std::size_t>(face);
        const auto i = static_cast<std::size_t>(mesh.connectivity.face_owner(f));
        const auto neighbour = mesh.connectivity.face_neigh(f);
        const auto& g = mesh.geometry;
        const Vector owner{g.cell_centroid_x(i), g.cell_centroid_y(i), g.cell_centroid_z(i)};
        const Vector centre{g.face_centroid_x(f), g.face_centroid_y(f), g.face_centroid_z(f)};
        Vector d = centre - owner;
        if (neighbour >= 0) {
            const auto j = static_cast<std::size_t>(neighbour);
            d = Vector{g.cell_centroid_x(j), g.cell_centroid_y(j), g.cell_centroid_z(j)} - owner;
        }
        const Vector normal{g.face_normal_x(f), g.face_normal_y(f), g.face_normal_z(f)};
        const MetricReal h = d.dot(normal), area = g.face_area(f);
        if (!positive(static_cast<Real>(h))) {
            ++bad;
            return;
        }
        const auto ratio = area / h;
        const MetricReal tolerance = MetricReal{128} *
            (static_cast<MetricReal>(Kokkos::epsilon_v<GeometryReal>) + Kokkos::epsilon_v<Real>);
        bool valid = positive(static_cast<Real>(ratio));

        // Check only attached caches. Empty views are valid when their metric is computed on demand.
        if (geometry.has_face_normal_dist()) {
            const MetricReal cached = geometry.face_normal_dist(f);
            valid = valid && positive(static_cast<Real>(cached)) &&
                Kokkos::abs(cached - h) <= tolerance*h;
        }
        if (geometry.has_face_area_over_normal_dist()) {
            const MetricReal cached = geometry.face_area_over_normal_dist(f);
            valid = valid && positive(static_cast<Real>(cached)) &&
                Kokkos::abs(cached - ratio) <= tolerance*ratio;
        }
        if (geometry.has_face_interp_weights()) {
            const MetricReal w = geometry.face_interp_weight(f);
            const auto expected = neighbour < 0 ? MetricReal{0} : geometric_owner_weight(centre - owner, normal, h);
            valid = valid && w >= MetricReal{0} && w <= MetricReal{1} &&
                Kokkos::abs(w - expected) <= tolerance;
        }
        if (geometry.has_face_non_ortho()) {
            const Vector cached{geometry.face_non_ortho_x(f), geometry.face_non_ortho_y(f),
                                geometry.face_non_ortho_z(f)};
            const auto expected = nonorthogonal_area(d, normal, area, h);
            const auto scaled_d = d * (normal.norm2() / h);
            for (std::size_t axis = 0; axis < 3; ++axis)
                valid = valid && Kokkos::isfinite(static_cast<Real>(cached[axis])) &&
                    Kokkos::abs(cached[axis] - expected[axis]) <=
                        tolerance*area*(MetricReal{1} + Kokkos::abs(scaled_d[axis]));
        }
        if (options.require_orthogonal) {
            const auto tangent = d - h * normal;
            const auto length = Kokkos::sqrt(tangent.norm2());
            valid = valid && Kokkos::isfinite(length) && length <= options.orthogonality_tolerance*h;
        }
        if (!valid) ++bad;
    }

private:
    KOKKOS_INLINE_FUNCTION static bool positive(Real value) {
        return Kokkos::isfinite(value) && value > Real{0};
    }
};

/// @brief Verifies the local topology, primary geometry and supplied caches before submitting work.
template<class Exec, class Real>
void validate_plan(mesh::device::MeshView<typename Exec::memory_space> mesh_view,
    mesh::device::AuxConnectivityView<typename Exec::memory_space> connectivity,
    mesh::device::AuxGeometryView<typename Exec::memory_space> geometry,
    mpi::CommView comm, const Exec& exec, DiffusionOptions options) {
    mpi::guarded([&] {
        execution::require_kokkos();
        const auto& s = mesh_view.sizes;
        mpi::require(comm.native() != MPI_COMM_NULL, "Scalar diffusion: null communicator");
        mpi::require(std::isfinite(options.orthogonality_tolerance) && options.orthogonality_tolerance >= 0 &&
                     options.orthogonality_tolerance < 1, "Scalar diffusion: invalid orthogonality tolerance");
        using mesh::device::MeshData;
        namespace validation = mesh::device::validation;
        // Existing shared validators check shapes, face intervals and complete cell-face incidence.
        const auto status = validation::validate<MeshData::FaceCells | MeshData::FaceGeometry,
            mesh::AuxConnType::CellFaces, mesh::AuxGeomType::None, true>(exec, mesh_view, connectivity, geometry);
        mpi::require(status.ok(), "Scalar diffusion: invalid mesh, cell-face incidence or normal metrics");
        const auto cells = static_cast<std::size_t>(s.n_cells);
        const auto owned = static_cast<std::size_t>(s.n_own);
        const auto& g = mesh_view.geometry;
        mpi::require(mesh_view.has(MeshData::CellGeometry) && g.cell_centroid_x.extent(0) >= cells &&
            g.cell_centroid_y.extent(0) >= cells && g.cell_centroid_z.extent(0) >= cells && g.cell_volume.extent(0) >= owned,
            "Scalar diffusion: attach cell centres and owned volumes");
        const auto faces = static_cast<std::size_t>(s.n_faces);
        if (geometry.has_face_normal_dist() || geometry.has_face_area_over_normal_dist() ||
            geometry.has_face_interp_weights() || geometry.has_face_non_ortho())
            mpi::require(geometry.n_faces == s.n_faces, "Scalar diffusion: normal metrics face count mismatch");
        mpi::require((!geometry.has_face_normal_dist() || geometry.face_normal_dist.extent(0) == faces) &&
                     (!geometry.has_face_area_over_normal_dist() || geometry.face_area_over_normal_dist.extent(0) == faces),
                     "Scalar diffusion: normal metrics extent mismatch");
        mpi::require(!geometry.has_face_interp_weights() || geometry.face_interp_weight.extent(0) == faces,
                     "Scalar diffusion: interpolation weight extent mismatch");
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
