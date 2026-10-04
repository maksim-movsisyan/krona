#include <krona/mesh/geometry.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <span>

namespace krona::mesh {
namespace {

struct Vec3 { double x = 0, y = 0, z = 0; };

/// @brief Adds local vectors or centroid moments componentwise.
[[nodiscard]] Vec3 operator+(Vec3 a, Vec3 b) noexcept
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

/// @brief Forms a displacement before geometric integration.
[[nodiscard]] Vec3 operator-(Vec3 a, Vec3 b) noexcept
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

/// @brief Scales a local vector or geometric moment.
[[nodiscard]] Vec3 operator*(Vec3 a, double scale) noexcept
{
    return {a.x * scale, a.y * scale, a.z * scale};
}

/// @brief Projects a displacement onto an oriented normal or area vector.
[[nodiscard]] double dot(Vec3 a, Vec3 b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// @brief Computes an oriented cross product from local edge vectors.
[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b) noexcept
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// @brief Computes a vector norm without squaring large or tiny components directly.
[[nodiscard]] double norm(Vec3 a) noexcept
{
    return std::hypot(a.x, a.y, a.z);
}

/// @brief Rejects NaN and infinity in a computed centroid or normal.
[[nodiscard]] bool finite(Vec3 a) noexcept
{
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

/// @brief Accepts positive finite measures without a unit-dependent cutoff.
[[nodiscard]] bool positive(double value) noexcept
{
    return value > 0 && std::isfinite(value);
}

/// @brief Reads one coordinate triple into double arithmetic.
[[nodiscard]] Vec3 point(const MeshPart& mesh, LocalIndex node) noexcept
{
    assert(node >= 0 && node < mesh.n_nodes);
    const auto index = static_cast<std::size_t>(node);
    return {static_cast<double>(mesh.node_x[index]), static_cast<double>(mesh.node_y[index]),
            static_cast<double>(mesh.node_z[index])};
}

/// @brief Returns a cell's stable coordinate origin from its first node.
[[nodiscard]] Vec3 cell_origin(const MeshPart& mesh, std::size_t cell) noexcept
{
    return point(mesh, mesh.cell_nodes[mesh.cell_nodes_offsets[cell]]);
}

/// @brief Reads a temporary centroid offset before its final absolute-coordinate conversion.
[[nodiscard]] Vec3 cell_offset(const MeshPart& mesh, std::size_t cell) noexcept
{
    return {static_cast<double>(mesh.cell_centroid_x[cell]),
            static_cast<double>(mesh.cell_centroid_y[cell]),
            static_cast<double>(mesh.cell_centroid_z[cell])};
}

/// @brief Reports a cell failure with its current local/global IDs and computed value.
[[noreturn]] void fail_cell(const MeshPart& mesh, std::size_t cell, const char* reason, double value)
{
    char message[320];
    // Formatting allocates no strings on the fatal path.
    std::snprintf(message, sizeof(message),
        "Geometry: rank=%d cell=%zu gid=%lld type=%.*s: %s (value=%.17g)",
        mesh.rank, cell, static_cast<long long>(mesh.cell_gid[cell]),
        static_cast<int>(cell_type_name(mesh.cell_type[cell]).size()),
        cell_type_name(mesh.cell_type[cell]).data(), reason, value);
    mpi::fatal(message);
}

/// @brief Reports a face failure with its owner, neighbour and offending value.
[[noreturn]] void fail_face(const MeshPart& mesh, std::size_t face, const char* reason, double value)
{
    char message[320];
    std::snprintf(message, sizeof(message),
        "Geometry: rank=%d face=%zu owner=%lld neighbour=%lld: %s (value=%.17g)",
        mesh.rank, face, static_cast<long long>(mesh.face_owner[face]),
        static_cast<long long>(mesh.face_neigh[face]), reason, value);
    mpi::fatal(message);
}

/// @brief Checks array dimensions once; topology and node-index validity are caller contracts.
void check_shape(const MeshPart& mesh)
{
    if (mesh.n_own < 0 || mesh.n_cells < mesh.n_own || mesh.n_nodes < 0 ||
        mesh.n_nodes_own < 0 || mesh.n_nodes_own > mesh.n_nodes || mesh.n_faces < 0)
        mpi::fatal("Invalid mesh entity counts for geometry");
    const auto cells = static_cast<std::size_t>(mesh.n_cells);
    const auto nodes = static_cast<std::size_t>(mesh.n_nodes);
    const auto faces = static_cast<std::size_t>(mesh.n_faces);
    if (mesh.cell_type.size() != cells || mesh.cell_gid.size() != cells ||
        mesh.cell_nodes_offsets.size() != cells + 1 || mesh.cell_nodes_offsets.front() != 0 ||
        mesh.cell_nodes_offsets.back() != mesh.cell_nodes.size() ||
        mesh.node_x.size() != nodes || mesh.node_y.size() != nodes || mesh.node_z.size() != nodes ||
        mesh.face_owner.size() != faces || mesh.face_neigh.size() != faces ||
        mesh.face_nodes_offsets.size() != faces + 1 || mesh.face_nodes_offsets.front() != 0 ||
        mesh.face_nodes_offsets.back() != mesh.face_nodes.size())
        mpi::fatal("Mesh array dimensions do not match geometry input counts");
}

struct CellMetrics { double volume = 0; Vec3 centre; };

/// @brief Integrates signed tetrahedron volumes and moments around a local cell anchor.
[[nodiscard]] CellMetrics cell_metrics(CellType type, std::span<Vec3> points) noexcept
{
    Vec3 anchor;
    for (const auto p : points) anchor = anchor + p;
    anchor = anchor * (1.0 / static_cast<double>(points.size()));
    // Points were already shifted by the first vertex; moments stay local too.
    for (auto& p : points) p = p - anchor;
    const auto ti = static_cast<std::size_t>(type);
    double volume = 0;
    Vec3 moment;
    for (int facet = 0; facet < faces_per_type[ti]; ++facet) {
        const int count = face_nodes[ti][facet];
        std::array<Vec3, 4> face;
        for (int i = 0; i < count; ++i)
            face[static_cast<std::size_t>(i)] = points[static_cast<std::size_t>(face_table[ti][facet][i])];
        if (count == 3) {
            const auto area = cross(face[1] - face[0], face[2] - face[0]) * 0.5;
            const double v = dot(area, face[0]) / 3.0;
            volume += v;
            moment = moment + (face[0] + face[1] + face[2]) * (v * 0.25);
        } else {
            // The same vertex-mean fan defines quad surfaces in face_metrics().
            const auto centre = (face[0] + face[1] + face[2] + face[3]) * 0.25;
            for (std::size_t i = 0; i < 4; ++i) {
                const auto a = face[i], b = face[(i + 1) % 4];
                const auto area = cross(a - centre, b - centre) * 0.5;
                const double v = dot(area, centre) / 3.0;
                volume += v;
                moment = moment + (centre + a + b) * (v * 0.25);
            }
        }
    }
    return {volume, positive(volume) ? anchor + moment * (1.0 / volume) : anchor};
}

struct FaceMetrics { double area = 0; Vec3 centre, normal; };

/// @brief Integrates a triangle or vertex-mean quad fan in local coordinates.
[[nodiscard]] FaceMetrics face_metrics(std::span<const Vec3> points) noexcept
{
    Vec3 area, centre;
    if (points.size() == 3) {
        area = cross(points[1] - points[0], points[2] - points[0]) * 0.5;
        centre = (points[0] + points[1] + points[2]) * (1.0 / 3.0);
    } else {
        const auto anchor = (points[0] + points[1] + points[2] + points[3]) * 0.25;
        double surface_area = 0;
        Vec3 moment;
        for (std::size_t i = 0; i < 4; ++i) {
            const auto a = points[i] - anchor, b = points[(i + 1) % 4] - anchor;
            const auto triangle_area = cross(a, b) * 0.5;
            const double weight = norm(triangle_area);
            area = area + triangle_area;
            surface_area += weight;
            moment = moment + (a + b) * (weight / 3.0);
        }
        // Actual subtriangle areas weight the centre; |area vector| is the flux area.
        centre = anchor + moment * (1.0 / surface_area);
    }
    const double magnitude = norm(area);
    return {magnitude, centre, positive(magnitude) ? area * (1.0 / magnitude) : Vec3{}};
}

/// @brief Fills reusable metric arrays and returns statistics for this rank only.
[[nodiscard]] GeometryStats compute(MeshPart& mesh)
{
    check_shape(mesh);
    const auto cells = static_cast<std::size_t>(mesh.n_cells), faces = static_cast<std::size_t>(mesh.n_faces);
    mesh.cell_centroid_x.resize(cells); mesh.cell_centroid_y.resize(cells); mesh.cell_centroid_z.resize(cells);
    mesh.cell_volume.resize(cells);
    mesh.face_centroid_x.resize(faces); mesh.face_centroid_y.resize(faces); mesh.face_centroid_z.resize(faces);
    mesh.face_normal_x.resize(faces); mesh.face_normal_y.resize(faces); mesh.face_normal_z.resize(faces);
    mesh.face_area.resize(faces);
    GeometryStats stats;
    double correction = 0;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const auto type = mesh.cell_type[cell];
        if (static_cast<std::size_t>(type) >= 4) mpi::fatal("Geometry supports only TET/PYRA/PRISM/HEXA cells");
        const auto begin = mesh.cell_nodes_offsets[cell], end = mesh.cell_nodes_offsets[cell + 1];
        if (end - begin != static_cast<std::size_t>(nodes_per_element(type)) || end > mesh.cell_nodes.size())
            fail_cell(mesh, cell, "Invalid cell connectivity extent", static_cast<double>(end - begin));
        std::array<Vec3, 8> vertices;
        const auto origin = cell_origin(mesh, cell);
        for (std::size_t i = begin; i < end; ++i) vertices[i - begin] = point(mesh, mesh.cell_nodes[i]) - origin;
        const auto metrics = cell_metrics(type, std::span{vertices}.first(end - begin));
        if (!positive(metrics.volume) || !finite(metrics.centre))
            fail_cell(mesh, cell, "Nonpositive or nonfinite cell geometry", metrics.volume);
        const auto volume = static_cast<GeometryReal>(metrics.volume);
        const Vec3 centre{static_cast<GeometryReal>(metrics.centre.x),
                          static_cast<GeometryReal>(metrics.centre.y),
                          static_cast<GeometryReal>(metrics.centre.z)};
        if (!positive(static_cast<double>(volume)) || !finite(centre))
            fail_cell(mesh, cell, "Cell geometry does not fit GeometryReal", metrics.volume);
        mesh.cell_volume[cell] = volume;
        // Reuse final arrays for offsets until winding checks finish; no second centroid buffer.
        mesh.cell_centroid_x[cell] = static_cast<GeometryReal>(centre.x);
        mesh.cell_centroid_y[cell] = static_cast<GeometryReal>(centre.y);
        mesh.cell_centroid_z[cell] = static_cast<GeometryReal>(centre.z);
        if (cell < static_cast<std::size_t>(mesh.n_own)) {
            stats.min_cell_volume = std::min(stats.min_cell_volume, metrics.volume);
            stats.max_cell_volume = std::max(stats.max_cell_volume, metrics.volume);
            // Compensated local summation reduces loss across a wide volume range.
            const double increment = metrics.volume - correction;
            const double total = stats.total_volume + increment;
            correction = (total - stats.total_volume) - increment;
            stats.total_volume = total;
        }
    }
    for (std::size_t face = 0; face < faces; ++face) {
        const auto begin = mesh.face_nodes_offsets[face], end = mesh.face_nodes_offsets[face + 1];
        if ((end - begin != 3 && end - begin != 4) || end > mesh.face_nodes.size())
            fail_face(mesh, face, "Invalid face connectivity extent", static_cast<double>(end - begin));
        assert(mesh.face_owner[face] >= 0 && mesh.face_owner[face] < mesh.n_own);
        const auto owner = static_cast<std::size_t>(mesh.face_owner[face]);
        const auto origin = cell_origin(mesh, owner);
        std::array<Vec3, 4> vertices;
        for (std::size_t i = begin; i < end; ++i) vertices[i - begin] = point(mesh, mesh.face_nodes[i]) - origin;
        const auto metrics = face_metrics(std::span{vertices}.first(end - begin));
        const auto area = static_cast<GeometryReal>(metrics.area);
        if (!positive(metrics.area) || !positive(static_cast<double>(area)) ||
            !finite(metrics.centre) || !finite(metrics.normal))
            fail_face(mesh, face, "Nonpositive, nonfinite or unrepresentable face geometry", metrics.area);
        // Compare local offsets, not rounded absolute centroids on shifted meshes.
        const double owner_side = dot(metrics.normal, metrics.centre - cell_offset(mesh, owner));
        if (!positive(owner_side)) fail_face(mesh, face, "Normal does not point out of owner", owner_side);
        const auto neighbour = mesh.face_neigh[face];
        if (neighbour != invalid_local_index) {
            assert(neighbour >= 0 && neighbour < mesh.n_cells);
            const auto other = static_cast<std::size_t>(neighbour);
            const auto neighbour_centre = (cell_origin(mesh, other) - origin) + cell_offset(mesh, other);
            const double neighbour_side = dot(metrics.normal, neighbour_centre - metrics.centre);
            if (!positive(neighbour_side))
                fail_face(mesh, face, "Normal does not point toward neighbour", neighbour_side);
        }
        const auto centre = origin + metrics.centre;
        if (!finite(centre)) fail_face(mesh, face, "Nonfinite absolute face centroid", metrics.area);
        mesh.face_centroid_x[face] = static_cast<GeometryReal>(centre.x);
        mesh.face_centroid_y[face] = static_cast<GeometryReal>(centre.y);
        mesh.face_centroid_z[face] = static_cast<GeometryReal>(centre.z);
        mesh.face_normal_x[face] = static_cast<GeometryReal>(metrics.normal.x);
        mesh.face_normal_y[face] = static_cast<GeometryReal>(metrics.normal.y);
        mesh.face_normal_z[face] = static_cast<GeometryReal>(metrics.normal.z);
        mesh.face_area[face] = area;
        stats.min_face_area = std::min(stats.min_face_area, metrics.area);
        stats.max_face_area = std::max(stats.max_face_area, metrics.area);
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const auto centre = cell_origin(mesh, cell) + cell_offset(mesh, cell);
        if (!finite(centre)) fail_cell(mesh, cell, "Nonfinite absolute cell centroid", mesh.cell_volume[cell]);
        mesh.cell_centroid_x[cell] = static_cast<GeometryReal>(centre.x);
        mesh.cell_centroid_y[cell] = static_cast<GeometryReal>(centre.y);
        mesh.cell_centroid_z[cell] = static_cast<GeometryReal>(centre.z);
    }
    // Coordinate motion requires fresh bounds; owned-used nodes avoid ghost-only outliers.
    for (LocalIndex node = 0; node < mesh.n_nodes_own; ++node) {
        const auto p = point(mesh, node);
        const std::array<double, 3> value{p.x, p.y, p.z};
        for (std::size_t d = 0; d < 3; ++d) {
            stats.bbox_lo[d] = std::min(stats.bbox_lo[d], value[d]);
            stats.bbox_hi[d] = std::max(stats.bbox_hi[d], value[d]);
        }
    }
    return stats;
}

/// @brief Packs extrema and bounds into one MIN reduction and sums owned volume separately.
[[nodiscard]] GeometryStats global_stats(GeometryStats stats, MeshPart& mesh, mpi::CommView comm)
{
    std::array<double, 10> extrema{stats.min_cell_volume, -stats.max_cell_volume,
        stats.min_face_area, -stats.max_face_area, stats.bbox_lo[0], stats.bbox_lo[1], stats.bbox_lo[2],
        -stats.bbox_hi[0], -stats.bbox_hi[1], -stats.bbox_hi[2]};
    if (comm.size() > 1) {
        mpi::allreduce_in_place<mpi::Reduction::min>(comm, std::span{extrema});
        stats.total_volume = mpi::allreduce<mpi::Reduction::sum>(comm, stats.total_volume);
    }
    stats.max_cell_volume = -extrema[1]; stats.max_face_area = -extrema[3];
    stats.min_cell_volume = stats.max_cell_volume > 0 ? extrema[0] : 0;
    stats.min_face_area = stats.max_face_area > 0 ? extrema[2] : 0;
    if (!std::isfinite(stats.total_volume)) mpi::fatal("Total owned volume is not finite");
    for (std::size_t d = 0; d < 3; ++d) {
        stats.bbox_lo[d] = stats.max_cell_volume > 0 ? extrema[d + 4] : 0;
        stats.bbox_hi[d] = stats.max_cell_volume > 0 ? -extrema[d + 7] : 0;
        mesh.bbox_lo[d] = static_cast<GeometryReal>(stats.bbox_lo[d]);
        mesh.bbox_hi[d] = static_cast<GeometryReal>(stats.bbox_hi[d]);
    }
    return stats;
}

} // namespace

/// @brief Recomputes local metrics, reusing storage and performing no collective operations.
GeometryStats compute_mesh_geometry(MeshPart& mesh)
{
    if (mpi::is_active()) mpi::require_main_thread();
    try {
        return compute(mesh);
    } catch (const std::exception& error) {
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unknown exception during local geometry computation");
    }
}

/// @brief Computes metrics collectively and refreshes the global coordinate bounds.
GeometryStats compute_mesh_geometry(MeshPart& mesh, mpi::CommView comm)
{
    if (comm.is_null() || !mpi::is_active()) mpi::fatal("Collective geometry requires active MPI");
    mpi::require_main_thread();
    if (mesh.rank != comm.rank() || mesh.nprocs != comm.size())
        mpi::fatal("Mesh geometry communicator does not match migration ranks");
    try {
        return global_stats(compute(mesh), mesh, comm);
    } catch (const std::exception& error) {
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unknown exception during collective geometry computation");
    }
}

} // namespace krona::mesh
