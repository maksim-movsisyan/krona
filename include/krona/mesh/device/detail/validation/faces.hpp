#pragma once

#include <krona/mesh/device/detail/validation/common.hpp>
#include <krona/mesh/device/aux_view.hpp>

namespace krona::mesh::device::validation {

/// @brief Checks one face's cell links, geometry and auxiliary metrics.
/// @tparam Data Base mesh groups to read, such as FaceCells and FaceGeometry.
/// @tparam Geometry Auxiliary metric groups to read, such as FaceNormalDistance.
/// @tparam Compare Also compare related values when the selected groups supply them.
/// @tparam Space Kokkos memory space of the mesh and geometry views.
/// Run over all local faces after checking that the selected arrays have the expected sizes.
template <MeshData Data, AuxGeomType Geometry, bool Compare, class Space>
struct ValidateFaces {
    MeshView<Space> mesh;
    AuxGeometryView<Space> geometry;
    Tolerance tolerance;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t f, std::size_t& bad) const {
        const auto& s = mesh.sizes;

        // Check the owner and neighbour before using them to read cell data.
        if constexpr (has(Data, MeshData::FaceCells)) {
            const auto owner = mesh.connectivity.face_owner(f), neigh = mesh.connectivity.face_neigh(f);
            bool valid = owner >= 0 && owner < s.n_own &&
                         (neigh == -1 || (neigh >= 0 && neigh < s.n_cells && neigh != owner));
            if (s.faces_grouped) {
                if (f < static_cast<std::size_t>(s.n_internal_faces))
                    valid &= neigh >= 0 && neigh < s.n_own;
                else if (f < static_cast<std::size_t>(s.n_inner_faces))
                    valid &= neigh >= s.n_own && neigh < s.n_cells;
                else
                    valid &= neigh == -1;
            }
            if (!valid) {
                ++bad;
                return;
            }
        }

        // Check base geometry and each selected auxiliary field on its own.
        bool valid = true;
        if constexpr (has(Data, MeshData::FaceGeometry)) {
            const auto& g = mesh.geometry;
            const GeometryReal nx = g.face_normal_x(f), ny = g.face_normal_y(f), nz = g.face_normal_z(f);
            const GeometryReal n2 = nx * nx + ny * ny + nz * nz;
            valid &= detail::positive(g.face_area(f)) &&
                     detail::finite3(g.face_centroid_x(f), g.face_centroid_y(f), g.face_centroid_z(f)) &&
                     detail::finite3(nx, ny, nz) && Kokkos::isfinite(n2) &&
                     Kokkos::abs(n2 - GeometryReal{1}) <= tolerance.normal_absolute;
        }

        if constexpr (has(Geometry, AuxGeomType::FaceCellDistance))
            valid &= detail::positive(geometry.face_cell_dist(f));
        if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceInv))
            valid &= detail::positive(geometry.face_cell_dist_inv(f));
        if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance))
            valid &= detail::positive(geometry.face_normal_dist(f));
        if constexpr (has(Geometry, AuxGeomType::FaceNormalDistanceInv))
            valid &= detail::positive(geometry.face_normal_dist_inv(f));
        if constexpr (has(Geometry, AuxGeomType::FaceAreaOverNormalDistance))
            valid &= detail::positive(geometry.face_area_over_normal_dist(f));
        if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceVector))
            valid &= detail::nonzero3(geometry.face_cell_dist_x(f), geometry.face_cell_dist_y(f),
                                      geometry.face_cell_dist_z(f));
        if constexpr (has(Geometry, AuxGeomType::FaceNonOrthogonalityVector))
            valid &= detail::finite3(geometry.face_non_ortho_x(f), geometry.face_non_ortho_y(f),
                                     geometry.face_non_ortho_z(f));
        if constexpr (has(Geometry, AuxGeomType::FaceInterpolationWeights)) {
            const GeometryReal w = geometry.face_interp_weight(f);
            valid &= Kokkos::isfinite(w) && w >= 0 && w <= 1;
            if constexpr (has(Data, MeshData::FaceCells)) {
                // A boundary face has no neighbour to interpolate with.
                if (mesh.connectivity.face_neigh(f) == -1)
                    valid &= w == 0;
            }
        }

        if constexpr (Compare) {
            // Cross-check distances, their reciprocals and the stored area/distance ratio.
            if constexpr ((has(Geometry, AuxGeomType::FaceCellDistance) &&
                           has(Geometry, AuxGeomType::FaceCellDistanceInv)))
                valid &= detail::close(geometry.face_cell_dist(f) * geometry.face_cell_dist_inv(f), 1,
                                       tolerance.relative);
            if constexpr ((has(Geometry, AuxGeomType::FaceNormalDistance) &&
                           has(Geometry, AuxGeomType::FaceNormalDistanceInv)))
                valid &= detail::close(geometry.face_normal_dist(f) * geometry.face_normal_dist_inv(f), 1,
                                       tolerance.relative);
            if constexpr (has(Data, MeshData::FaceGeometry) &&
                          (has(Geometry, AuxGeomType::FaceNormalDistance) &&
                           has(Geometry, AuxGeomType::FaceAreaOverNormalDistance)))
                valid &= detail::close(geometry.face_normal_dist(f) * geometry.face_area_over_normal_dist(f),
                                       mesh.geometry.face_area(f), tolerance.relative);

            if constexpr ((has(Data, MeshData::FaceCells) && has(Data, MeshData::CellGeometry) &&
                           has(Data, MeshData::FaceGeometry))) {
                const auto& g = mesh.geometry;
                const auto o = static_cast<std::size_t>(mesh.connectivity.face_owner(f));
                const auto neighbour = mesh.connectivity.face_neigh(f);
                const auto n = neighbour < 0 ? std::size_t{0} : static_cast<std::size_t>(neighbour);

                // Measure from the owner centre to the neighbour centre or to the boundary face centre.
                const GeometryReal dx =
                    (neighbour < 0 ? g.face_centroid_x(f) : g.cell_centroid_x(n)) - g.cell_centroid_x(o);
                const GeometryReal dy =
                    (neighbour < 0 ? g.face_centroid_y(f) : g.cell_centroid_y(n)) - g.cell_centroid_y(o);
                const GeometryReal dz =
                    (neighbour < 0 ? g.face_centroid_z(f) : g.cell_centroid_z(n)) - g.cell_centroid_z(o);

                // Along the normal the face must lie ahead of the owner and before any neighbour.
                const GeometryReal nx = g.face_normal_x(f), ny = g.face_normal_y(f), nz = g.face_normal_z(f);
                const GeometryReal dn = dx * nx + dy * ny + dz * nz;
                valid &= detail::nonzero3(dx, dy, dz) && detail::positive(dn);
                const GeometryReal left = (g.face_centroid_x(f) - g.cell_centroid_x(o)) * nx +
                                          (g.face_centroid_y(f) - g.cell_centroid_y(o)) * ny +
                                          (g.face_centroid_z(f) - g.cell_centroid_z(o)) * nz;
                valid &= detail::positive(left) && (neighbour < 0 || detail::positive(dn - left));

                // Scaled norm avoids overflow in dx*dx for large finite coordinates.
                const GeometryReal scale =
                    detail::max(Kokkos::abs(dx), detail::max(Kokkos::abs(dy), Kokkos::abs(dz)));
                const GeometryReal norm = scale > 0 ? scale * Kokkos::sqrt((dx / scale) * (dx / scale) +
                                                                           (dy / scale) * (dy / scale) +
                                                                           (dz / scale) * (dz / scale))
                                                    : 0;

                // Use the reconstructed distances and displacement to check the stored values.
                if constexpr (has(Geometry, AuxGeomType::FaceCellDistance))
                    valid &= detail::close(geometry.face_cell_dist(f), norm, tolerance.relative);
                if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance))
                    valid &= detail::close(geometry.face_normal_dist(f), dn, tolerance.relative);
                if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceVector)) {
                    // Use the vector scale, allowing cancellation in individual components.
                    valid &= detail::finite3(dx, dy, dz) &&
                             Kokkos::abs(geometry.face_cell_dist_x(f) - dx) <= tolerance.relative * scale &&
                             Kokkos::abs(geometry.face_cell_dist_y(f) - dy) <= tolerance.relative * scale &&
                             Kokkos::abs(geometry.face_cell_dist_z(f) - dz) <= tolerance.relative * scale;
                }

                // The owner's weight is the face-to-neighbour fraction of the projected distance.
                if constexpr (has(Geometry, AuxGeomType::FaceInterpolationWeights)) {
                    GeometryReal w = 0;
                    if (neighbour >= 0) {
                        w = (dn - left) / dn;
                    }
                    valid &= Kokkos::isfinite(w) &&
                             Kokkos::abs(geometry.face_interp_weight(f) - w) <= tolerance.relative;
                }

                // Rebuild the non-orthogonal part of the face area vector.
                if constexpr (has(Geometry, AuxGeomType::FaceNonOrthogonalityVector)) {
                    const GeometryReal area = g.face_area(f), factor = (nx * nx + ny * ny + nz * nz) / dn;
                    const GeometryReal tx = area * (nx - factor * dx), ty = area * (ny - factor * dy),
                                       tz = area * (nz - factor * dz);
                    const GeometryReal ts = detail::max(
                        area, detail::max(Kokkos::abs(tx), detail::max(Kokkos::abs(ty), Kokkos::abs(tz))));
                    valid &= detail::finite3(tx, ty, tz) &&
                             Kokkos::abs(geometry.face_non_ortho_x(f) - tx) <= tolerance.relative * ts &&
                             Kokkos::abs(geometry.face_non_ortho_y(f) - ty) <= tolerance.relative * ts &&
                             Kokkos::abs(geometry.face_non_ortho_z(f) - tz) <= tolerance.relative * ts;
                }
            }
        }

        if (!valid)
            ++bad;
    }
};

} // namespace krona::mesh::device::validation
