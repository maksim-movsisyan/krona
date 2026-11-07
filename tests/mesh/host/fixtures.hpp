#pragma once

#include <krona/distribution/block.hpp>
#include <krona/mesh/host/faces.hpp>
#include <krona/mesh/host/geometry.hpp>
#include <krona/mesh/host/migrate.hpp>
#include <krona/mesh/host/reorder.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace krona::test::mesh_host {

namespace host = krona::mesh::host;
using Block = distribution::BlockDistribution<GlobalIndex>;

/// @brief Appends one input element without calling a production topology builder.
template <class T> void append(host::ElementTable& table, mesh::CellType type, const T& nodes) {
    table.types.push_back(type);
    table.node_ids.insert(table.node_ids.end(), nodes.begin(), nodes.end());
    table.offsets.push_back(table.node_ids.size());
}

/// @brief Creates balanced import ownership for independently supplied global input arrays.
inline host::RawMesh distribute(mpi::CommView comm, const host::ElementTable& cells,
                                std::span<const std::array<GeometryReal, 3>> points,
                                const host::BoundaryTable& boundary, std::vector<host::Patch> patches = {}) {
    host::RawMesh result;
    result.cell_dimension = result.physical_dimension = 3;
    result.distribution = {Block(static_cast<GlobalIndex>(cells.size()), comm.size()),
                           Block(static_cast<GlobalIndex>(points.size()), comm.size()),
                           Block(static_cast<GlobalIndex>(boundary.size()), comm.size()), comm.rank()};
    result.patches = std::move(patches);
    const auto& ownership = result.distribution;
    for (auto id = ownership.cells.begin(comm.rank()); id < ownership.cells.end(comm.rank()); ++id) {
        const auto i = static_cast<std::size_t>(id);
        append(result.cells, cells.types[i], cells.nodes(i));
    }
    for (auto id = ownership.nodes.begin(comm.rank()); id < ownership.nodes.end(comm.rank()); ++id) {
        const auto& p = points[static_cast<std::size_t>(id)];
        result.coordinates.x.push_back(p[0]);
        result.coordinates.y.push_back(p[1]);
        result.coordinates.z.push_back(p[2]);
    }
    for (auto id = ownership.boundary_ordinals.begin(comm.rank());
         id < ownership.boundary_ordinals.end(comm.rank()); ++id) {
        const auto i = static_cast<std::size_t>(id);
        append(result.boundary.elements, boundary.elements.types[i], boundary.elements.nodes(i));
        result.boundary.global_ids.push_back(boundary.global_ids[i]);
        result.boundary.patch_ids.push_back(boundary.patch_ids[i]);
    }
    return result;
}

/// @brief Returns the analytic coordinate of a node in a row of unit hexahedra.
inline std::array<GeometryReal, 3> chain_point(GlobalIndex node) {
    constexpr std::array<std::array<GeometryReal, 2>, 4> yz{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    const auto& p = yz[static_cast<std::size_t>(node % 4)];
    return {static_cast<GeometryReal>(node / 4), p[0], p[1]};
}

/// @brief Creates a labelled chain of unit cubes with analytically known face neighbours.
inline host::RawMesh chain(mpi::CommView comm, GlobalIndex count = 3, bool labelled = true) {
    host::ElementTable cells;
    host::BoundaryTable boundary;
    std::vector<std::array<GeometryReal, 3>> points;
    for (GlobalIndex n = 0; n < 4 * (count + 1); ++n)
        points.push_back(chain_point(n));
    // These loops describe input connectivity; expected counts below use chain identities.
    constexpr std::array<std::array<std::size_t, 4>, 6> loops{
        {{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}}};
    for (GlobalIndex cell = 0; cell < count; ++cell) {
        const GlobalIndex left = 4 * cell, right = left + 4;
        const std::array<GlobalIndex, 8> nodes{left,     right,     right + 1, left + 1,
                                               left + 3, right + 3, right + 2, left + 2};
        append(cells, mesh::CellType::HEXA, nodes);
        if (!labelled)
            continue;
        for (std::size_t face = 0; face < 6; ++face) {
            if ((face == 4 && cell != 0) || (face == 2 && cell + 1 != count))
                continue;
            std::array<GlobalIndex, 4> q{};
            for (std::size_t j = 0; j < 4; ++j)
                q[j] = nodes[loops[face][j]];
            append(boundary.elements, mesh::CellType::QUAD, q);
            // Sparse source IDs deliberately differ from dense boundary ordinals.
            boundary.global_ids.push_back(100 + 7 * static_cast<GlobalIndex>(boundary.size() - 1));
            boundary.patch_ids.push_back(cell == 0 && face == 4 ? 0 : 1);
        }
    }
    return distribute(comm, cells, points, boundary, {{"wall", "solid", "BCWall"}, {"outer", "", ""}});
}

/// @brief Creates isolated TET, PYRA, PRISM and HEXA cells with exact reference measures.
inline host::RawMesh mixed(mpi::CommView comm) {
    const std::array<std::vector<std::array<GeometryReal, 3>>, 4> shapes{
        {{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
         {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {GeometryReal{0.5}, GeometryReal{0.5}, 1}}},
         {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}},
         {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}}}};
    host::ElementTable cells;
    std::vector<std::array<GeometryReal, 3>> points;
    for (std::size_t c = 0; c < shapes.size(); ++c) {
        std::vector<GlobalIndex> nodes;
        for (auto p : shapes[c]) {
            nodes.push_back(static_cast<GlobalIndex>(points.size()));
            p[0] += static_cast<GeometryReal>(3 * c);
            points.push_back(p);
        }
        append(cells, static_cast<mesh::CellType>(c), nodes);
    }
    return distribute(comm, cells, points, {});
}

/// @brief Creates a connected planar QUAD/TRI pair with five labelled boundary edges.
inline host::RawMesh planar(mpi::CommView comm, bool tilted = false) {
    host::ElementTable cells;
    append(cells, mesh::CellType::QUAD, std::array<GlobalIndex, 4>{0, 1, 2, 3});
    append(cells, mesh::CellType::TRI, std::array<GlobalIndex, 3>{1, 4, 2});
    std::array<std::array<GeometryReal, 3>, 5> points{
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {2, 0, 0}}};
    if (tilted)
        for (auto& p : points)
            p = {2, p[0], p[1]};
    host::BoundaryTable boundary;
    constexpr std::array<std::array<GlobalIndex, 2>, 5> edges{{{0, 1}, {1, 4}, {4, 2}, {2, 3}, {3, 0}}};
    for (const auto& edge : edges) {
        append(boundary.elements, mesh::CellType::BAR, edge);
        boundary.global_ids.push_back(80 + static_cast<GlobalIndex>(boundary.size()));
        boundary.patch_ids.push_back(0);
    }
    auto result = distribute(comm, cells, points, boundary, {{"side", "", ""}});
    result.cell_dimension = 2;
    result.physical_dimension = tilted ? 3 : 2;
    return result;
}

/// @brief Builds a solver snapshot with caller-selected destinations and no partitioner dependency.
inline host::MeshPart migrate(host::RawMesh source, mpi::CommView comm, bool rotate = true) {
    auto topology = host::build_faces(source, comm);
    std::vector<int> targets(source.cells.size());
    for (std::size_t c = 0; c < targets.size(); ++c) {
        const auto id = source.global_cell_id(static_cast<LocalIndex>(c));
        targets[c] = (static_cast<int>(id % comm.size()) + static_cast<int>(rotate)) % comm.size();
    }
    return host::migrate_local_mesh(std::move(source), std::move(topology.faces), targets, comm);
}

/// @brief Completes metrics and face grouping before validation or auxiliary construction.
inline host::MeshPart prepared_chain(mpi::CommView comm) {
    auto result = migrate(chain(comm), comm);
    (void)host::compute_mesh_geometry(result, comm);
    host::reorder_local_mesh(result);
    return result;
}

} // namespace krona::test::mesh_host
