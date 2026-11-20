#pragma once

#include <krona/mesh/device/mesh_view.hpp>
#include <krona/mesh/device/aux_view.hpp>
#include <krona/transport/scalar/diffusion/detail/geometry.hpp>

#include <cstdint>
#include <type_traits>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Data passed to coefficient and correction providers for a face between real cells.
/// The normal and cell ordering follow the mesh owner, including when processing the neighbour row.
/// @tparam Real Arithmetic type used by the diffusion operator.
template<class Real>
struct InteriorFaceContext {
    LocalIndex face = 0, owner = 0, neighbour = -1;
    Real x = 0, y = 0, z = 0;      ///< Face centroid in mesh coordinates.
    Real nx = 0, ny = 0, nz = 0;   ///< Unit normal pointing from owner to neighbour.
    Real distance = 0;             ///< Positive centre separation projected onto the normal.
    Real area = 0, time = 0;        ///< Face area and provider evaluation time.
    Real owner_weight = 0;         ///< Geometric weight used for cell coefficients and gradients.
};

/// @brief Geometry sampled once for a face contribution.
/// @tparam Real Arithmetic type used by the diffusion operator.
template<class Real>
struct InteriorMetrics {
    InteriorFaceContext<Real> context;
    Real area_over_distance;            ///< Multiplier for the two-point diffusive conductance.
    math::Vector3<Real> nonorthogonal{}; ///< Area vector used to project the correction gradient.
};

/// @brief Borrows row incidence, primary mesh geometry and optional metric arrays.
/// @tparam Space Kokkos memory space of the borrowed views.
template<class Space>
struct StencilData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;

    LocalIndex n_inner_faces;
    Index face_owner, face_neigh, faces;
    Offset offsets;
    Metric cell_x, cell_y, cell_z, face_x, face_y, face_z, normal_x, normal_y, normal_z, area;
    Metric distance, area_over_distance, owner_weight, non_ortho_x, non_ortho_y, non_ortho_z;
    mesh::AuxGeomType cached;

    /// @brief Captures the arrays needed by interior-face kernels.
    StencilData(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> conn,
                mesh::device::AuxGeometryView<Space> geometry)
        : n_inner_faces(mesh.sizes.n_inner_faces), face_owner(mesh.connectivity.face_owner),
          face_neigh(mesh.connectivity.face_neigh), faces(conn.cell_faces), offsets(conn.cell_faces_offsets),
          cell_x(mesh.geometry.cell_centroid_x), cell_y(mesh.geometry.cell_centroid_y),
          cell_z(mesh.geometry.cell_centroid_z), face_x(mesh.geometry.face_centroid_x),
          face_y(mesh.geometry.face_centroid_y), face_z(mesh.geometry.face_centroid_z),
          normal_x(mesh.geometry.face_normal_x), normal_y(mesh.geometry.face_normal_y),
          normal_z(mesh.geometry.face_normal_z), area(mesh.geometry.face_area),
          distance(geometry.face_normal_dist), area_over_distance(geometry.face_area_over_normal_dist),
          owner_weight(geometry.face_interp_weight), non_ortho_x(geometry.face_non_ortho_x),
          non_ortho_y(geometry.face_non_ortho_y), non_ortho_z(geometry.face_non_ortho_z),
          cached(geometry.active_mask) {}

    /// @brief Gets the metrics for an interior or MPI interface face, computing missing entries.
    /// @tparam Real Arithmetic type used to return the metrics.
    /// @tparam Corrected True when the caller also projects a face gradient.
    /// @param face Local ID of a face between two real cells.
    /// @param time Evaluation time for custom coefficient and correction providers.
    template<class Real, bool Corrected>
    KOKKOS_INLINE_FUNCTION InteriorMetrics<Real> interior(LocalIndex face, Real time) const {
        using MetricReal = std::common_type_t<Real, GeometryReal>;
        using Vector = math::Vector3<MetricReal>;
        using mesh::AuxGeomType;
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = face_owner(f), neighbour = face_neigh(f);
        const std::size_t i = static_cast<std::size_t>(owner);
        const Vector n{normal_x(f), normal_y(f), normal_z(f)};
        const Vector centre{face_x(f), face_y(f), face_z(f)};
        const MetricReal face_area = area(f);
        const bool cached_height = has(AuxGeomType::FaceNormalDistance);
        const bool cached_correction = has(AuxGeomType::FaceNonOrthogonalityVector);
        const bool cached_weight = has(AuxGeomType::FaceInterpolationWeights);
        const bool needs_connection = !cached_height || (Corrected && !cached_correction);
        Vector origin{};
        if (needs_connection || !cached_weight) origin = cell_centre<MetricReal>(i);
        Vector d{};
        if (needs_connection)
            d = cell_centre<MetricReal>(static_cast<std::size_t>(neighbour)) - origin;
        const MetricReal h = cached_height ? MetricReal{distance(f)} : d.dot(n);
        const auto ratio = has(AuxGeomType::FaceAreaOverNormalDistance)
            ? MetricReal{area_over_distance(f)} : face_area / h;
        const auto weight = cached_weight
            ? MetricReal{owner_weight(f)} : geometric_owner_weight(centre - origin, n, h);

        InteriorMetrics<Real> result{{face, owner, neighbour, static_cast<Real>(centre.x),
            static_cast<Real>(centre.y), static_cast<Real>(centre.z), static_cast<Real>(n.x),
            static_cast<Real>(n.y), static_cast<Real>(n.z), static_cast<Real>(h),
            static_cast<Real>(face_area), time, static_cast<Real>(weight)}, static_cast<Real>(ratio)};
        if constexpr (Corrected) {
            const auto vector = cached_correction
                ? Vector{non_ortho_x(f), non_ortho_y(f), non_ortho_z(f)}
                : nonorthogonal_area(d, n, face_area, h);
            result.nonorthogonal = math::cast<Real>(vector);
        }
        return result;
    }

private:
    KOKKOS_INLINE_FUNCTION bool has(mesh::AuxGeomType metric) const {
        return (static_cast<std::uint64_t>(cached) & static_cast<std::uint64_t>(metric)) != 0;
    }

    template<class Real>
    KOKKOS_INLINE_FUNCTION math::Vector3<Real> cell_centre(std::size_t i) const {
        return {cell_x(i), cell_y(i), cell_z(i)};
    }
};

} // namespace krona::transport::scalar::diffusion::detail
