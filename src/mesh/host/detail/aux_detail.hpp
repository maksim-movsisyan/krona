#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>

namespace krona::mesh::host::aux_detail {

using Vector = std::array<double, 3>;

/// @brief Checks the required metric shapes once without rescanning validated topology.
inline void require_geometry(const MeshPart& m)
{
    const auto cells = static_cast<std::size_t>(m.n_cells), faces = static_cast<std::size_t>(m.n_faces);
    if (!m.faces_grouped || m.n_inner_faces < 0 || m.n_inner_faces > m.n_faces ||
        m.cell_centroid_x.size() != cells || m.cell_centroid_y.size() != cells || m.cell_centroid_z.size() != cells ||
        m.face_centroid_x.size() != faces || m.face_centroid_y.size() != faces || m.face_centroid_z.size() != faces ||
        m.face_normal_x.size() != faces || m.face_normal_y.size() != faces || m.face_normal_z.size() != faces ||
        m.face_area.size() != faces)
        mpi::fatal("Auxiliary geometry requires grouped faces and complete MeshPart metrics");
}

/// @brief Loads a cell centroid into double before taking coordinate differences.
[[nodiscard]] inline Vector cell_center(const MeshPart& m, std::size_t cell)
{
    return {m.cell_centroid_x[cell], m.cell_centroid_y[cell], m.cell_centroid_z[cell]};
}
/// @brief Loads the stored area-weighted face centroid into double.
[[nodiscard]] inline Vector face_center(const MeshPart& m, std::size_t face)
{
    return {m.face_centroid_x[face], m.face_centroid_y[face], m.face_centroid_z[face]};
}
/// @brief Loads the owner's outward normal without assuming exactly unit stored precision.
[[nodiscard]] inline Vector face_normal(const MeshPart& m, std::size_t face)
{
    return {m.face_normal_x[face], m.face_normal_y[face], m.face_normal_z[face]};
}
/// @brief Computes a three-component difference in double arithmetic.
[[nodiscard]] inline Vector subtract(const Vector& a, const Vector& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
/// @brief Computes a three-component scalar product.
[[nodiscard]] inline double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
/// @brief Computes Euclidean length without explicitly squaring extreme coordinates.
[[nodiscard]] inline double length(const Vector& v)
{
    // hypot avoids unnecessary overflow/underflow from dx*dx + dy*dy + dz*dz.
    return std::hypot(v[0], v[1], v[2]);
}
/// @brief Converts a finite computed metric to GeometryReal, rejecting storage overflow.
[[nodiscard]] inline GeometryReal stored(double value)
{
    const auto result = static_cast<GeometryReal>(value);
    if (!std::isfinite(value) || !std::isfinite(result))
        mpi::fatal("Auxiliary metric is nonfinite or exceeds GeometryReal");
    return result;
}
/// @brief Stores a positive measure without hiding zero distances behind an absolute cutoff.
[[nodiscard]] inline GeometryReal positive(double value)
{
    const auto result = stored(value);
    // Positivity must survive storage as well as the double calculation.
    if (!(result > 0)) mpi::fatal("Auxiliary distance must be positive and representable");
    return result;
}

} // namespace krona::mesh::host::aux_detail
