#include <krona/mesh/host/aux_geometry.hpp>

#include <cassert>

#include "detail/aux_detail.hpp"

namespace krona::mesh::host {

/// @brief Computes each missing face metric from one shared centre-displacement calculation.
void MeshAuxGeometry::add_geometry(const MeshPart& m, AuxGeomType requested)
{
    const std::uint64_t bits = static_cast<std::uint64_t>(requested);
    if ((bits & ~static_cast<std::uint64_t>(AuxGeomType::All)) != 0) mpi::fatal("Unknown auxiliary geometry flag");
    const auto pending = static_cast<AuxGeomType>(bits & ~static_cast<std::uint64_t>(active_mask) &
                                                 static_cast<std::uint64_t>(AuxGeomType::AllFaces));
    // Face-only construction cannot advertise an uncomputed wall-distance field.
    if (pending == AuxGeomType::None) return;
    assert((active_mask & AuxGeomType::AllFaces) == AuxGeomType::None || n_faces == m.n_faces);
    aux_detail::require_geometry(m);
    n_faces = m.n_faces;
    n_cells = m.n_cells;
    const std::size_t count = static_cast<std::size_t>(n_faces);
    const bool distance = has_flag(pending, AuxGeomType::FaceCellDistance);
    const bool vector = has_flag(pending, AuxGeomType::FaceCellDistanceVector);
    const bool inverse = has_flag(pending, AuxGeomType::FaceCellDistanceInv);
    const bool weights = has_flag(pending, AuxGeomType::FaceInterpolationWeights);
    const bool correction = has_flag(pending, AuxGeomType::FaceNonOrthogonalityVector);
    const bool normal_dist = has_flag(pending, AuxGeomType::FaceNormalDistance);
    const bool normal_dist_inverse = has_flag(pending, AuxGeomType::FaceNormalDistanceInv);
    const bool area_over_normal_dist = has_flag(pending, AuxGeomType::FaceAreaOverNormalDistance);

    // Allocate only explicit outputs; vector/distance arrays are not implicit dependencies.
    if (distance) face_cell_dist.resize(count);
    if (vector) {
        face_cell_dist_x.resize(count); face_cell_dist_y.resize(count); face_cell_dist_z.resize(count);
    }
    if (inverse) face_cell_dist_inv.resize(count);
    if (weights) face_interp_weight.resize(count);
    if (correction) {
        face_non_ortho_x.resize(count); face_non_ortho_y.resize(count); face_non_ortho_z.resize(count);
    }
    if (normal_dist)
        face_normal_dist.resize(count);
    if (normal_dist_inverse)
        face_normal_dist_inv.resize(count);
    if (area_over_normal_dist)
        face_area_over_normal_dist.resize(count);

    for (std::size_t f = 0; f < count; ++f) {
        const std::size_t owner = static_cast<std::size_t>(m.face_owner[f]);
        const bool interior = f < static_cast<std::size_t>(m.n_inner_faces);
        const auto origin = aux_detail::cell_center(m, owner);
        aux_detail::Vector from_owner{}, d{};
        if (interior) {
            d = aux_detail::subtract(aux_detail::cell_center(m, static_cast<std::size_t>(m.face_neigh[f])),
                                      origin);
            if (weights) from_owner = aux_detail::subtract(aux_detail::face_center(m, f), origin);
        } else {
            // Boundary stencil ends at the face centroid.
            d = aux_detail::subtract(aux_detail::face_center(m, f), origin);
            from_owner = d;
        }
        if (vector) {
            face_cell_dist_x[f] = aux_detail::stored(d[0]);
            face_cell_dist_y[f] = aux_detail::stored(d[1]);
            face_cell_dist_z[f] = aux_detail::stored(d[2]);
        }
        if (distance || inverse) {
            const auto norm = aux_detail::length(d);
            if (!(norm > 0) || !std::isfinite(norm)) mpi::fatal("Auxiliary centre separation must be finite and positive");
            if (distance) face_cell_dist[f] = aux_detail::positive(norm);
            if (inverse) face_cell_dist_inv[f] = aux_detail::positive(1 / norm);
        }
        if (weights || correction || normal_dist ||
                normal_dist_inverse || area_over_normal_dist) {
            const auto normal = aux_detail::face_normal(m, f);
            const double projection = aux_detail::dot(d, normal);
            if (!(projection > 0) || !std::isfinite(projection)) mpi::fatal("Auxiliary stencil requires a resolved outward centre separation");
            if (weights) {
                double weight = 0.0; // Boundary value is already located at the face.
                if (interior) {
                    const double left = aux_detail::dot(from_owner, normal);
                    const double right = projection - left;
                    if (!(left > 0 && right > 0))
                        mpi::fatal("Face interpolation requires centres on opposite sides of the face plane");
                    weight = right / projection;
                }
                face_interp_weight[f] = aux_detail::stored(weight);
            }
            if (correction) {
                // T=A*(n - (n dot n)/(d dot n)*d), avoiding an unnecessary A*A intermediate.
                const double factor = aux_detail::dot(normal, normal) / projection;
                const double area = m.face_area[f];
                face_non_ortho_x[f] = aux_detail::stored(area * (normal[0] - factor * d[0]));
                face_non_ortho_y[f] = aux_detail::stored(area * (normal[1] - factor * d[1]));
                face_non_ortho_z[f] = aux_detail::stored(area * (normal[2] - factor * d[2]));
            }

            if (normal_dist)
                face_normal_dist[f] = aux_detail::positive(projection);

            if (normal_dist_inverse)
                face_normal_dist_inv[f] = aux_detail::positive(1.0 / projection);

            if (area_over_normal_dist)
                face_area_over_normal_dist[f] =
                    aux_detail::positive(m.face_area[f] / projection);

        }
    }
    // Publish readiness only after all requested arrays have been filled.
    active_mask = active_mask | pending;
}

/// @brief Creates an empty metric cache and fills requested local face arrays.
MeshAuxGeometry build_aux_geometry(const MeshPart& m, AuxGeomType mask)
{
    MeshAuxGeometry result;
    result.add_geometry(m, mask);
    return result;
}

} // namespace krona::mesh::host
