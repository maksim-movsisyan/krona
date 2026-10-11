#pragma once

#include <krona/mesh/compute/aux_view.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace krona::mesh::compute::validation {

// Local diagnostics only. Selected arrays must remain alive and unchanged.
// Producers on another execution-space instance must be synchronized first.
// Public validators return host counters and therefore complete their reductions.
// A counter counts bad rows/entities, not individual failed predicates.
// Kernels may be used directly after validating counts, availability and shapes.

template<class Mask>
KOKKOS_INLINE_FUNCTION constexpr bool has(Mask mask, Mask flags) noexcept
{
    return (static_cast<std::uint64_t>(mask) & static_cast<std::uint64_t>(flags)) ==
           static_cast<std::uint64_t>(flags);
}

struct Tolerance {
    GeometryReal relative = GeometryReal{64} * std::numeric_limits<GeometryReal>::epsilon();
    GeometryReal normal_absolute = GeometryReal{64} * std::numeric_limits<GeometryReal>::epsilon();
};

struct Result {
    std::size_t shapes = 0, cells = 0, faces = 0, nodes = 0;
    std::size_t cell_nodes = 0, face_nodes = 0;
    std::size_t cell_faces = 0, cell_cells_face = 0, cell_cells_node = 0;
    std::size_t node_cells = 0, node_faces = 0;
    std::size_t halo_send = 0, halo_recv = 0;
    std::size_t cell_incidence = 0, face_incidence = 0, node_incidence = 0;

    [[nodiscard]] bool ok() const noexcept
    {
        return shapes == 0 && cells == 0 && faces == 0 && nodes == 0 &&
               cell_nodes == 0 && face_nodes == 0 && cell_faces == 0 &&
               cell_cells_face == 0 && cell_cells_node == 0 && node_cells == 0 &&
               node_faces == 0 && halo_send == 0 && halo_recv == 0 &&
               cell_incidence == 0 && face_incidence == 0 && node_incidence == 0;
    }
};

namespace detail {

KOKKOS_INLINE_FUNCTION bool positive(GeometryReal x) { return Kokkos::isfinite(x) && x > 0; }
KOKKOS_INLINE_FUNCTION bool finite3(GeometryReal x, GeometryReal y, GeometryReal z)
{
    return Kokkos::isfinite(x) && Kokkos::isfinite(y) && Kokkos::isfinite(z);
}
KOKKOS_INLINE_FUNCTION GeometryReal max(GeometryReal a, GeometryReal b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION bool close(GeometryReal a, GeometryReal b, GeometryReal tolerance)
{
    return Kokkos::isfinite(a) && Kokkos::isfinite(b) &&
           Kokkos::abs(a - b) <= tolerance * max(Kokkos::abs(a), Kokkos::abs(b));
}
KOKKOS_INLINE_FUNCTION bool nonzero3(GeometryReal x, GeometryReal y, GeometryReal z)
{
    return finite3(x, y, z) && max(Kokkos::abs(x), max(Kokkos::abs(y), Kokkos::abs(z))) > 0;
}
template<class Offsets, class Values, class Id>
KOKKOS_INLINE_FUNCTION bool contains(Offsets offsets, Values values, std::size_t row, Id id)
{
    for (std::size_t j = offsets(row); j < offsets(row + 1); ++j)
        if (values(j) == id) return true;
    return false;
}

inline bool valid_sizes(const MeshSizes& s)
{
    return s.n_own >= 0 && s.n_cells >= s.n_own && s.n_nodes_own >= 0 &&
           s.n_nodes >= s.n_nodes_own && s.n_faces >= 0 &&
           (!s.faces_grouped || (s.n_internal_faces >= 0 &&
             s.n_inner_faces >= s.n_internal_faces && s.n_faces >= s.n_inner_faces));
}
inline bool valid_tolerance(Tolerance t)
{
    return Kokkos::isfinite(t.relative) && t.relative >= 0 &&
           Kokkos::isfinite(t.normal_absolute) && t.normal_absolute >= 0;
}
template<class View>
bool shape(View v, std::size_t count)
{
    return v.extent(0) == count && (count == 0 || v.data() != nullptr);
}
template<class Offsets, class Values>
bool csr_shape(Offsets offsets, Values values, std::size_t rows)
{
    return rows != std::numeric_limits<std::size_t>::max() && shape(offsets, rows + 1) &&
           (values.extent(0) == 0 || values.data() != nullptr);
}
template<class Exec, class Kernel>
std::size_t reduce(const char* label, const Exec& exec, std::size_t count, Kernel kernel)
{
    std::size_t bad = 0;
    if (count != 0)
        Kokkos::parallel_reduce(label,
            Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>(exec, 0, count), kernel, bad);
    return bad;
}

} // namespace detail

/// @brief Checks finite cell centres and strictly positive volumes, including MPI ghosts.
template<MeshData Data, AuxGeomType Geometry, class Space>
struct ValidateCells {
    MeshView<Space> mesh;
    AuxGeometryView<Space> geometry;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t c, std::size_t& bad) const
    {
        bool valid = true;
        if constexpr (has(Data, MeshData::CellGeometry)) {
            const auto& g = mesh.geometry;
            valid &= detail::positive(g.cell_volume(c)) &&
                detail::finite3(g.cell_centroid_x(c), g.cell_centroid_y(c), g.cell_centroid_z(c));
        }
        if constexpr (has(Geometry, AuxGeomType::WallDistance)) {
            const GeometryReal d = geometry.cell_wall_dist(c);
            // Positive infinity is the documented unreachable-cell sentinel.
            valid &= d >= 0 && (Kokkos::isfinite(d) || d == std::numeric_limits<GeometryReal>::infinity());
        }
        if (!valid) ++bad;
    }
};

template<MeshData Data, class Space>
struct ValidateNodes {
    MeshView<Space> mesh;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t n, std::size_t& bad) const
    {
        if constexpr (has(Data, MeshData::NodeCoordinates))
            if (!detail::finite3(mesh.geometry.node_x(n), mesh.geometry.node_y(n), mesh.geometry.node_z(n))) ++bad;
    }
};

/// @brief Checks face grouping, unit normals and selected auxiliary metrics.
/// Compare additionally checks metric identities from the selected source arrays.
template<MeshData Data, AuxGeomType Geometry, bool Compare, class Space>
struct ValidateFaces {
    MeshView<Space> mesh;
    AuxGeometryView<Space> geometry;
    Tolerance tolerance;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t f, std::size_t& bad) const
    {
        const auto& s = mesh.sizes;
        if constexpr (has(Data, MeshData::FaceCells)) {
            const auto owner = mesh.connectivity.face_owner(f), neigh = mesh.connectivity.face_neigh(f);
            bool valid = owner >= 0 && owner < s.n_own &&
                (neigh == -1 || (neigh >= 0 && neigh < s.n_cells && neigh != owner));
            if (s.faces_grouped) {
                if (f < static_cast<std::size_t>(s.n_internal_faces)) valid &= neigh >= 0 && neigh < s.n_own;
                else if (f < static_cast<std::size_t>(s.n_inner_faces)) valid &= neigh >= s.n_own && neigh < s.n_cells;
                else valid &= neigh == -1;
            }
            // Never gather through invalid owner/neighbour indices.
            if (!valid) { ++bad; return; }
        }
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
        if constexpr (has(Geometry, AuxGeomType::FaceCellDistance)) valid &= detail::positive(geometry.face_cell_dist(f));
        if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceInv)) valid &= detail::positive(geometry.face_cell_dist_inv(f));
        if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance)) valid &= detail::positive(geometry.face_normal_dist(f));
        if constexpr (has(Geometry, AuxGeomType::FaceNormalDistanceInv)) valid &= detail::positive(geometry.face_normal_dist_inv(f));
        if constexpr (has(Geometry, AuxGeomType::FaceAreaOverNormalDistance)) valid &= detail::positive(geometry.face_area_over_normal_dist(f));
        if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceVector))
            valid &= detail::nonzero3(geometry.face_cell_dist_x(f), geometry.face_cell_dist_y(f), geometry.face_cell_dist_z(f));
        if constexpr (has(Geometry, AuxGeomType::FaceNonOrthogonalityVector))
            valid &= detail::finite3(geometry.face_non_ortho_x(f), geometry.face_non_ortho_y(f), geometry.face_non_ortho_z(f));
        if constexpr (has(Geometry, AuxGeomType::FaceInterpolationWeights)) {
            const GeometryReal w = geometry.face_interp_weight(f);
            valid &= Kokkos::isfinite(w) && w >= 0 && w <= 1;
            if constexpr (has(Data, MeshData::FaceCells)) {
                if (mesh.connectivity.face_neigh(f) == -1) valid &= w == 0;
            }
        }
        if constexpr (Compare) {
            if constexpr (has(Geometry, AuxGeomType::FaceCellDistance | AuxGeomType::FaceCellDistanceInv))
                valid &= detail::close(geometry.face_cell_dist(f) * geometry.face_cell_dist_inv(f), 1, tolerance.relative);
            if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance | AuxGeomType::FaceNormalDistanceInv))
                valid &= detail::close(geometry.face_normal_dist(f) * geometry.face_normal_dist_inv(f), 1, tolerance.relative);
            if constexpr (has(Data, MeshData::FaceGeometry) &&
                          has(Geometry, AuxGeomType::FaceNormalDistance | AuxGeomType::FaceAreaOverNormalDistance))
                valid &= detail::close(geometry.face_normal_dist(f) * geometry.face_area_over_normal_dist(f), mesh.geometry.face_area(f), tolerance.relative);
            if constexpr (has(Data, MeshData::FaceCells | MeshData::CellGeometry | MeshData::FaceGeometry)) {
                const auto& g = mesh.geometry;
                const auto o = static_cast<std::size_t>(mesh.connectivity.face_owner(f));
                const auto neighbour = mesh.connectivity.face_neigh(f);
                const auto n = neighbour < 0 ? std::size_t{0} : static_cast<std::size_t>(neighbour);
                const GeometryReal dx = (neighbour < 0 ? g.face_centroid_x(f) : g.cell_centroid_x(n)) - g.cell_centroid_x(o);
                const GeometryReal dy = (neighbour < 0 ? g.face_centroid_y(f) : g.cell_centroid_y(n)) - g.cell_centroid_y(o);
                const GeometryReal dz = (neighbour < 0 ? g.face_centroid_z(f) : g.cell_centroid_z(n)) - g.cell_centroid_z(o);
                const GeometryReal nx = g.face_normal_x(f), ny = g.face_normal_y(f), nz = g.face_normal_z(f);
                const GeometryReal dn = dx * nx + dy * ny + dz * nz;
                valid &= detail::nonzero3(dx, dy, dz) && detail::positive(dn);
                const GeometryReal left = (g.face_centroid_x(f) - g.cell_centroid_x(o)) * nx +
                    (g.face_centroid_y(f) - g.cell_centroid_y(o)) * ny +
                    (g.face_centroid_z(f) - g.cell_centroid_z(o)) * nz;
                valid &= detail::positive(left) && (neighbour < 0 || detail::positive(dn - left));
                // Scaled norm avoids overflow in dx*dx for large finite coordinates.
                const GeometryReal scale = detail::max(Kokkos::abs(dx), detail::max(Kokkos::abs(dy), Kokkos::abs(dz)));
                const GeometryReal norm = scale > 0 ? scale * Kokkos::sqrt((dx / scale) * (dx / scale) + (dy / scale) * (dy / scale) + (dz / scale) * (dz / scale)) : 0;
                if constexpr (has(Geometry, AuxGeomType::FaceCellDistance)) valid &= detail::close(geometry.face_cell_dist(f), norm, tolerance.relative);
                if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance)) valid &= detail::close(geometry.face_normal_dist(f), dn, tolerance.relative);
                if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceVector)) {
                    // Use the vector scale, allowing cancellation in individual components.
                    valid &= detail::finite3(dx, dy, dz) &&
                        Kokkos::abs(geometry.face_cell_dist_x(f) - dx) <= tolerance.relative * scale &&
                        Kokkos::abs(geometry.face_cell_dist_y(f) - dy) <= tolerance.relative * scale &&
                        Kokkos::abs(geometry.face_cell_dist_z(f) - dz) <= tolerance.relative * scale;
                }
                if constexpr (has(Geometry, AuxGeomType::FaceInterpolationWeights)) {
                    GeometryReal w = 0;
                    if (neighbour >= 0) {
                        w = (dn - left) / dn;
                    }
                    valid &= Kokkos::isfinite(w) && Kokkos::abs(geometry.face_interp_weight(f) - w) <= tolerance.relative;
                }
                if constexpr (has(Geometry, AuxGeomType::FaceNonOrthogonalityVector)) {
                    const GeometryReal area = g.face_area(f), factor = (nx * nx + ny * ny + nz * nz) / dn;
                    const GeometryReal tx = area * (nx - factor * dx), ty = area * (ny - factor * dy), tz = area * (nz - factor * dz);
                    const GeometryReal ts = detail::max(area, detail::max(Kokkos::abs(tx), detail::max(Kokkos::abs(ty), Kokkos::abs(tz))));
                    valid &= detail::finite3(tx, ty, tz) &&
                        Kokkos::abs(geometry.face_non_ortho_x(f) - tx) <= tolerance.relative * ts &&
                        Kokkos::abs(geometry.face_non_ortho_y(f) - ty) <= tolerance.relative * ts &&
                        Kokkos::abs(geometry.face_non_ortho_z(f) - tz) <= tolerance.relative * ts;
                }
            }
        }
        if (!valid) ++bad;
    }
};

enum class RowOrder { Unordered, Nondecreasing, Strict };

/// @brief Checks CSR endpoints, row bounds, IDs and optional ordering/self exclusion.
/// Run over rows+1: the final iteration checks the terminal offset, even for zero rows.
template<RowOrder Order, bool RejectSelf, bool UniqueUnordered, class Offsets, class Values>
struct ValidateCsr {
    Offsets offsets;
    Values values;
    std::size_t rows, upper_bound;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t row, std::size_t& bad) const
    {
        if (row == rows) {
            if (offsets(row) != values.extent(0) || (rows == 0 && offsets(row) != 0)) ++bad;
            return;
        }
        const auto begin = offsets(row), end = offsets(row + 1);
        if ((row == 0 && begin != 0) || begin > end || end > values.extent(0)) { ++bad; return; }
        bool valid = true;
        for (std::size_t j = begin; j < end; ++j) {
            const auto id = values(j);
            valid &= id >= 0 && static_cast<std::size_t>(id) < upper_bound;
            if constexpr (RejectSelf) valid &= id < 0 || static_cast<std::size_t>(id) != row;
            if constexpr (Order != RowOrder::Unordered) {
                if (j != begin) {
                    if constexpr (Order == RowOrder::Strict) valid &= values(j - 1) < id;
                    else valid &= values(j - 1) <= id;
                }
            }
            if constexpr (UniqueUnordered)
                for (std::size_t k = begin; k < j; ++k) valid &= values(k) != id;
        }
        if (!valid) ++bad;
    }
};

template<class Map>
struct ValidateIndices {
    Map map;
    LocalIndex first, end;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, std::size_t& bad) const
    {
        const auto id = map(i);
        if (id < first || id >= end) ++bad;
    }
};

/// @brief Checks selected cell incidences after all CSR/index checks have succeeded.
template<MeshData Data, AuxConnType Connectivity, class Space>
struct ValidateCellIncidence {
    MeshView<Space> mesh;
    AuxConnectivityView<Space> conn;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell, std::size_t& bad) const
    {
        const auto& a = mesh.connectivity;
        bool valid = true;
        if constexpr (has(Data, MeshData::CellNodes))
            for (std::size_t j = a.cell_nodes_offsets(cell); j < a.cell_nodes_offsets(cell + 1); ++j)
                valid &= a.cell_nodes(j) < mesh.sizes.n_nodes_own;
        if constexpr (has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)) {
            for (std::size_t j = conn.cell_faces_offsets(cell); j < conn.cell_faces_offsets(cell + 1); ++j) {
                const auto face = static_cast<std::size_t>(conn.cell_faces(j));
                const auto owner = a.face_owner(face), neighbour = a.face_neigh(face);
                const auto id = static_cast<LocalIndex>(cell);
                valid &= owner == id || neighbour == id;
                if constexpr (has(Connectivity, AuxConnType::CellCellsByFace)) {
                    const auto other = owner == id ? neighbour : owner;
                    if (other >= 0) valid &= detail::contains(conn.cell_cells_face_offsets, conn.cell_cells_face, cell, other);
                }
            }
            if constexpr (has(Connectivity, AuxConnType::CellCellsByFace)) {
                for (std::size_t j = conn.cell_cells_face_offsets(cell); j < conn.cell_cells_face_offsets(cell + 1); ++j) {
                    const auto other = conn.cell_cells_face(j);
                    std::size_t expected = 0, actual = 0;
                    for (std::size_t k = conn.cell_faces_offsets(cell); k < conn.cell_faces_offsets(cell + 1); ++k) {
                        const auto f = static_cast<std::size_t>(conn.cell_faces(k));
                        const auto id = static_cast<LocalIndex>(cell);
                        if ((a.face_owner(f) == id && a.face_neigh(f) == other) ||
                            (a.face_neigh(f) == id && a.face_owner(f) == other)) ++expected;
                    }
                    for (std::size_t k = conn.cell_cells_face_offsets(cell); k < conn.cell_cells_face_offsets(cell + 1); ++k)
                        if (conn.cell_cells_face(k) == other) ++actual;
                    valid &= expected == actual;
                }
            }
        }
        if constexpr (has(Data, MeshData::CellNodes) && has(Connectivity, AuxConnType::NodeCells)) {
            for (std::size_t j = a.cell_nodes_offsets(cell); j < a.cell_nodes_offsets(cell + 1); ++j) {
                const auto node = a.cell_nodes(j);
                if (node < mesh.sizes.n_nodes_own)
                    valid &= detail::contains(conn.node_cells_offsets, conn.node_cells, static_cast<std::size_t>(node), static_cast<LocalIndex>(cell));
            }
        }
        if constexpr (has(Data, MeshData::CellNodes) && has(Connectivity, AuxConnType::CellCellsByNode)) {
            for (std::size_t j = conn.cell_cells_node_offsets(cell); j < conn.cell_cells_node_offsets(cell + 1); ++j) {
                const auto other = static_cast<std::size_t>(conn.cell_cells_node(j));
                bool shared = false;
                for (std::size_t k = a.cell_nodes_offsets(cell); k < a.cell_nodes_offsets(cell + 1); ++k)
                    shared |= detail::contains(a.cell_nodes_offsets, a.cell_nodes, other, a.cell_nodes(k));
                valid &= shared;
            }
            if constexpr (has(Connectivity, AuxConnType::NodeCells))
                for (std::size_t j = a.cell_nodes_offsets(cell); j < a.cell_nodes_offsets(cell + 1); ++j) {
                    const auto node = static_cast<std::size_t>(a.cell_nodes(j));
                    if (node >= static_cast<std::size_t>(mesh.sizes.n_nodes_own)) continue;
                    for (std::size_t k = conn.node_cells_offsets(node); k < conn.node_cells_offsets(node + 1); ++k)
                        if (conn.node_cells(k) != static_cast<LocalIndex>(cell))
                            valid &= detail::contains(conn.cell_cells_node_offsets, conn.cell_cells_node, cell, conn.node_cells(k));
                }
        }
        if (!valid) ++bad;
    }
};

template<MeshData Data, AuxConnType Connectivity, class Space>
struct ValidateFaceIncidence {
    MeshView<Space> mesh;
    AuxConnectivityView<Space> conn;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t face, std::size_t& bad) const
    {
        const auto& a = mesh.connectivity;
        bool valid = true;
        if constexpr (has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)) {
            valid &= detail::contains(conn.cell_faces_offsets, conn.cell_faces,
                static_cast<std::size_t>(a.face_owner(face)), static_cast<LocalIndex>(face));
            const auto n = a.face_neigh(face);
            if (n >= 0 && n < mesh.sizes.n_own)
                valid &= detail::contains(conn.cell_faces_offsets, conn.cell_faces, static_cast<std::size_t>(n), static_cast<LocalIndex>(face));
        }
        if constexpr (has(Data, MeshData::FaceCells | MeshData::FaceNodes | MeshData::CellNodes)) {
            const auto o = static_cast<std::size_t>(a.face_owner(face));
            const auto n = a.face_neigh(face);
            for (std::size_t j = a.face_nodes_offsets(face); j < a.face_nodes_offsets(face + 1); ++j) {
                const auto node = a.face_nodes(j);
                valid &= detail::contains(a.cell_nodes_offsets, a.cell_nodes, o, node);
                if (n >= 0) valid &= detail::contains(a.cell_nodes_offsets, a.cell_nodes, static_cast<std::size_t>(n), node);
            }
        }
        if constexpr (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces)) {
            for (std::size_t j = a.face_nodes_offsets(face); j < a.face_nodes_offsets(face + 1); ++j) {
                const auto node = a.face_nodes(j);
                if (node < mesh.sizes.n_nodes_own)
                    valid &= detail::contains(conn.node_faces_offsets, conn.node_faces, static_cast<std::size_t>(node), static_cast<LocalIndex>(face));
            }
        }
        if (!valid) ++bad;
    }
};

template<MeshData Data, AuxConnType Connectivity, class Space>
struct ValidateNodeIncidence {
    MeshView<Space> mesh;
    AuxConnectivityView<Space> conn;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t node, std::size_t& bad) const
    {
        const auto& a = mesh.connectivity;
        bool valid = true;
        if constexpr (has(Connectivity, AuxConnType::NodeCells)) {
            bool owned_user = false;
            for (std::size_t j = conn.node_cells_offsets(node); j < conn.node_cells_offsets(node + 1); ++j)
                owned_user |= conn.node_cells(j) < mesh.sizes.n_own;
            valid &= owned_user;
        }
        if constexpr (has(Data, MeshData::CellNodes) && has(Connectivity, AuxConnType::NodeCells))
            for (std::size_t j = conn.node_cells_offsets(node); j < conn.node_cells_offsets(node + 1); ++j)
                valid &= detail::contains(a.cell_nodes_offsets, a.cell_nodes, static_cast<std::size_t>(conn.node_cells(j)), static_cast<LocalIndex>(node));
        if constexpr (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces))
            for (std::size_t j = conn.node_faces_offsets(node); j < conn.node_faces_offsets(node + 1); ++j)
                valid &= detail::contains(a.face_nodes_offsets, a.face_nodes, static_cast<std::size_t>(conn.node_faces(j)), static_cast<LocalIndex>(node));
        if (!valid) ++bad;
    }
};

namespace detail {

template<MeshData Data, AuxConnType Connectivity, AuxGeomType Geometry, class Space>
std::size_t check_shapes(MeshView<Space> mesh, AuxConnectivityView<Space> conn, AuxGeometryView<Space> geom)
{
    static_assert((static_cast<std::uint64_t>(Data) & ~static_cast<std::uint64_t>(MeshData::All)) == 0);
    static_assert((static_cast<std::uint64_t>(Connectivity) & ~static_cast<std::uint64_t>(AuxConnType::All)) == 0);
    static_assert((static_cast<std::uint64_t>(Geometry) & ~static_cast<std::uint64_t>(AuxGeomType::All)) == 0);
    if (!valid_sizes(mesh.sizes)) return 1;
    std::size_t bad = 0;
    auto check = [&](bool valid) { if (!valid) ++bad; };
    check(mesh.has(Data));
    check(conn.has(Connectivity));
    check(geom.has(Geometry));
    const auto cells = static_cast<std::size_t>(mesh.sizes.n_cells);
    const auto own = static_cast<std::size_t>(mesh.sizes.n_own);
    const auto faces = static_cast<std::size_t>(mesh.sizes.n_faces);
    const auto nodes = static_cast<std::size_t>(mesh.sizes.n_nodes);
    const auto own_nodes = static_cast<std::size_t>(mesh.sizes.n_nodes_own);
    if constexpr (Connectivity != AuxConnType::None)
        check(conn.n_faces == mesh.sizes.n_faces && conn.n_cells_own == mesh.sizes.n_own && conn.n_nodes_own == mesh.sizes.n_nodes_own);
    if constexpr ((static_cast<std::uint64_t>(Geometry) & static_cast<std::uint64_t>(AuxGeomType::AllFaces)) != 0)
        check(geom.n_faces == mesh.sizes.n_faces && mesh.sizes.faces_grouped);
    if constexpr (has(Geometry, AuxGeomType::WallDistance)) check(geom.n_cells == mesh.sizes.n_cells);
    const auto& a = mesh.connectivity;
    const auto& g = mesh.geometry;
    if constexpr (has(Data, MeshData::FaceCells)) { check(shape(a.face_owner, faces)); check(shape(a.face_neigh, faces)); }
    if constexpr (has(Data, MeshData::CellNodes)) { check(shape(a.cell_type, cells)); check(csr_shape(a.cell_nodes_offsets, a.cell_nodes, cells)); }
    if constexpr (has(Data, MeshData::FaceNodes)) {
        check(shape(a.face_type, faces)); check(shape(a.face_local_index, faces));
        check(csr_shape(a.face_nodes_offsets, a.face_nodes, faces));
    }
    if constexpr (has(Data, MeshData::NodeCoordinates)) { check(shape(g.node_x, nodes)); check(shape(g.node_y, nodes)); check(shape(g.node_z, nodes)); }
    if constexpr (has(Data, MeshData::CellGeometry)) {
        check(shape(g.cell_centroid_x, cells)); check(shape(g.cell_centroid_y, cells)); check(shape(g.cell_centroid_z, cells)); check(shape(g.cell_volume, cells));
    }
    if constexpr (has(Data, MeshData::FaceGeometry)) {
        check(shape(g.face_centroid_x, faces)); check(shape(g.face_centroid_y, faces)); check(shape(g.face_centroid_z, faces));
        check(shape(g.face_normal_x, faces)); check(shape(g.face_normal_y, faces)); check(shape(g.face_normal_z, faces)); check(shape(g.face_area, faces));
    }
    if constexpr (has(Data, MeshData::Halo)) {
        check(mesh.halo.send_owned_local.extent(0) == 0 || mesh.halo.send_owned_local.data() != nullptr);
        check(mesh.halo.recv_ghost_local.extent(0) == 0 || mesh.halo.recv_ghost_local.data() != nullptr);
    }
    if constexpr (has(Connectivity, AuxConnType::CellFaces)) check(csr_shape(conn.cell_faces_offsets, conn.cell_faces, own));
    if constexpr (has(Connectivity, AuxConnType::CellCellsByFace)) check(csr_shape(conn.cell_cells_face_offsets, conn.cell_cells_face, own));
    if constexpr (has(Connectivity, AuxConnType::CellCellsByNode)) check(csr_shape(conn.cell_cells_node_offsets, conn.cell_cells_node, own));
    if constexpr (has(Connectivity, AuxConnType::NodeCells)) check(csr_shape(conn.node_cells_offsets, conn.node_cells, own_nodes));
    if constexpr (has(Connectivity, AuxConnType::NodeFaces)) check(csr_shape(conn.node_faces_offsets, conn.node_faces, own_nodes));
    if constexpr (has(Geometry, AuxGeomType::FaceCellDistance)) check(shape(geom.face_cell_dist, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceInv)) check(shape(geom.face_cell_dist_inv, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceNormalDistance)) check(shape(geom.face_normal_dist, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceNormalDistanceInv)) check(shape(geom.face_normal_dist_inv, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceAreaOverNormalDistance)) check(shape(geom.face_area_over_normal_dist, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceInterpolationWeights)) check(shape(geom.face_interp_weight, faces));
    if constexpr (has(Geometry, AuxGeomType::FaceCellDistanceVector)) {
        check(shape(geom.face_cell_dist_x, faces)); check(shape(geom.face_cell_dist_y, faces)); check(shape(geom.face_cell_dist_z, faces));
    }
    if constexpr (has(Geometry, AuxGeomType::FaceNonOrthogonalityVector)) {
        check(shape(geom.face_non_ortho_x, faces)); check(shape(geom.face_non_ortho_y, faces)); check(shape(geom.face_non_ortho_z, faces));
    }
    if constexpr (has(Geometry, AuxGeomType::WallDistance)) check(shape(geom.cell_wall_dist, cells));
    return bad;
}

} // namespace detail

/// @brief Validates selected view arrays locally; no allocations or MPI collectives.
/// Shape errors prevent every kernel launch. Compare does not require absent arrays.
/// This checks storage/numerical contracts and selected incidence relations, not manifoldness,
/// halo message agreement, patch metadata, or element-specific topology.
template<MeshData Data = MeshData::All, AuxConnType Connectivity = AuxConnType::None,
         AuxGeomType Geometry = AuxGeomType::None, bool Compare = false,
         class Exec, class Space>
[[nodiscard]] Result validate(const Exec& exec, MeshView<Space> mesh,
                              AuxConnectivityView<Space> conn = {}, AuxGeometryView<Space> geom = {},
                              Tolerance tolerance = {})
{
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    Result result;
    result.shapes = detail::check_shapes<Data, Connectivity, Geometry>(mesh, conn, geom);
    if (!detail::valid_tolerance(tolerance)) ++result.shapes;
    if (result.shapes != 0) return result;
    const auto cells = static_cast<std::size_t>(mesh.sizes.n_cells);
    const auto own = static_cast<std::size_t>(mesh.sizes.n_own);
    const auto faces = static_cast<std::size_t>(mesh.sizes.n_faces);
    const auto nodes = static_cast<std::size_t>(mesh.sizes.n_nodes);
    const auto own_nodes = static_cast<std::size_t>(mesh.sizes.n_nodes_own);
    if constexpr (has(Data, MeshData::CellGeometry) || has(Geometry, AuxGeomType::WallDistance))
        result.cells = detail::reduce("mesh/validate_cells", exec, cells, ValidateCells<Data, Geometry, Space>{mesh, geom});
    if constexpr (has(Data, MeshData::FaceCells) || has(Data, MeshData::FaceGeometry) ||
                  (static_cast<std::uint64_t>(Geometry) & static_cast<std::uint64_t>(AuxGeomType::AllFaces)) != 0)
        result.faces = detail::reduce("mesh/validate_faces", exec, faces, ValidateFaces<Data, Geometry, Compare, Space>{mesh, geom, tolerance});
    if constexpr (has(Data, MeshData::NodeCoordinates))
        result.nodes = detail::reduce("mesh/validate_nodes", exec, nodes, ValidateNodes<Data, Space>{mesh});
    auto csr = [&]<RowOrder Order, bool Self, bool Unique>(const char* label, auto offsets, auto values, std::size_t rows, std::size_t bound) {
        return detail::reduce(label, exec, rows + 1, ValidateCsr<Order, Self, Unique, decltype(offsets), decltype(values)>{offsets, values, rows, bound});
    };
    const auto& a = mesh.connectivity;
    if constexpr (has(Data, MeshData::CellNodes)) result.cell_nodes = csr.template operator()<RowOrder::Unordered, false, true>("mesh/validate_cell_nodes", a.cell_nodes_offsets, a.cell_nodes, cells, nodes);
    if constexpr (has(Data, MeshData::FaceNodes)) result.face_nodes = csr.template operator()<RowOrder::Unordered, false, true>("mesh/validate_face_nodes", a.face_nodes_offsets, a.face_nodes, faces, nodes);
    if constexpr (has(Connectivity, AuxConnType::CellFaces)) result.cell_faces = csr.template operator()<RowOrder::Strict, false, false>("mesh/validate_cell_faces", conn.cell_faces_offsets, conn.cell_faces, own, faces);
    if constexpr (has(Connectivity, AuxConnType::CellCellsByFace)) result.cell_cells_face = csr.template operator()<RowOrder::Nondecreasing, true, false>("mesh/validate_cell_cells_face", conn.cell_cells_face_offsets, conn.cell_cells_face, own, cells);
    if constexpr (has(Connectivity, AuxConnType::CellCellsByNode)) result.cell_cells_node = csr.template operator()<RowOrder::Strict, true, false>("mesh/validate_cell_cells_node", conn.cell_cells_node_offsets, conn.cell_cells_node, own, cells);
    if constexpr (has(Connectivity, AuxConnType::NodeCells)) result.node_cells = csr.template operator()<RowOrder::Strict, false, false>("mesh/validate_node_cells", conn.node_cells_offsets, conn.node_cells, own_nodes, cells);
    if constexpr (has(Connectivity, AuxConnType::NodeFaces)) result.node_faces = csr.template operator()<RowOrder::Strict, false, false>("mesh/validate_node_faces", conn.node_faces_offsets, conn.node_faces, own_nodes, faces);
    if constexpr (has(Data, MeshData::Halo)) {
        result.halo_send = detail::reduce("mesh/validate_halo_send", exec, mesh.halo.send_owned_local.extent(0), ValidateIndices<decltype(mesh.halo.send_owned_local)>{mesh.halo.send_owned_local, 0, mesh.sizes.n_own});
        result.halo_recv = detail::reduce("mesh/validate_halo_recv", exec, mesh.halo.recv_ghost_local.extent(0), ValidateIndices<decltype(mesh.halo.recv_ghost_local)>{mesh.halo.recv_ghost_local, mesh.sizes.n_own, mesh.sizes.n_cells});
    }
    // Invalid CSR/index data must never enter incidence gathers.
    if constexpr (Compare) {
        if (result.ok()) {
            if constexpr (has(Data, MeshData::CellNodes) || (has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)))
                result.cell_incidence = detail::reduce("mesh/validate_cell_incidence", exec, own, ValidateCellIncidence<Data, Connectivity, Space>{mesh, conn});
            if constexpr ((has(Data, MeshData::FaceCells) && has(Connectivity, AuxConnType::CellFaces)) ||
                          has(Data, MeshData::FaceCells | MeshData::FaceNodes | MeshData::CellNodes) ||
                          (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces)))
                result.face_incidence = detail::reduce("mesh/validate_face_incidence", exec, faces, ValidateFaceIncidence<Data, Connectivity, Space>{mesh, conn});
            if constexpr (has(Connectivity, AuxConnType::NodeCells) ||
                          (has(Data, MeshData::FaceNodes) && has(Connectivity, AuxConnType::NodeFaces)))
                result.node_incidence = detail::reduce("mesh/validate_node_incidence", exec, own_nodes, ValidateNodeIncidence<Data, Connectivity, Space>{mesh, conn});
        }
    }
    return result;
}

/// @brief Checks borrowed host peer metadata against map lengths and communicator ranks.
/// Plan must expose nb_ranks, send_offsets and recv_offsets (HaloPlanView).
template<class Space, class Plan>
[[nodiscard]] bool valid_halo_plan(MeshView<Space> mesh, const Plan& plan, int rank, int comm_size)
{
    if (!mesh.has(MeshData::Halo) || rank < 0 || rank >= comm_size) return false;
    const auto peers = plan.nb_ranks.size();
    if (peers == std::numeric_limits<std::size_t>::max() ||
        plan.send_offsets.size() != peers + 1 || plan.recv_offsets.size() != peers + 1) return false;
    if (plan.send_offsets.front() != 0 || plan.recv_offsets.front() != 0 ||
        plan.send_offsets.back() != mesh.halo.send_owned_local.extent(0) ||
        plan.recv_offsets.back() != mesh.halo.recv_ghost_local.extent(0)) return false;
    int previous = -1;
    for (std::size_t p = 0; p < peers; ++p) {
        const int peer = plan.nb_ranks[p];
        if (peer <= previous || peer >= comm_size || peer == rank ||
            plan.send_offsets[p] > plan.send_offsets[p + 1] ||
            plan.recv_offsets[p] > plan.recv_offsets[p + 1]) return false;
        previous = peer;
    }
    return true;
}

/// @brief Detects repeated receive destinations in O(n_cells + n_recv), with temporary storage.
template<class Exec, class Space>
[[nodiscard]] std::size_t duplicate_halo_receives(const Exec& exec, MeshView<Space> mesh)
{
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    if (!detail::valid_sizes(mesh.sizes) || !mesh.has(MeshData::Halo) ||
        (mesh.halo.recv_ghost_local.extent(0) != 0 && mesh.halo.recv_ghost_local.data() == nullptr)) return 1;
    if (mesh.halo.recv_ghost_local.extent(0) == 0) return 0;
    using Counts = Kokkos::View<std::size_t*, Kokkos::Device<Exec, typename Exec::memory_space>>;
    Counts counts(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "mesh/halo_destination_counts"), static_cast<std::size_t>(mesh.sizes.n_cells - mesh.sizes.n_own));
    // Keep temporary memory alive even if submission fails.
    try {
        Kokkos::deep_copy(exec, counts, std::size_t{0});
        const auto map = mesh.halo.recv_ghost_local;
        const auto own = mesh.sizes.n_own, end = mesh.sizes.n_cells;
        return detail::reduce("mesh/validate_unique_halo", exec, map.extent(0),
            KOKKOS_LAMBDA(std::size_t i, std::size_t& bad) {
                const auto id = map(i);
                if (id < own || id >= end) { ++bad; return; }
                if (Kokkos::atomic_fetch_add(&counts(static_cast<std::size_t>(id - own)), std::size_t{1}) != 0) ++bad;
            });
    } catch (...) { exec.fence("mesh/validate_unique_halo_failed"); throw; }
}

} // namespace krona::mesh::compute::validation
