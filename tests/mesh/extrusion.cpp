#include "krona/mesh/extrusion.hpp"
#include "krona/mpi/collectives.hpp"
#include "krona/mpi/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using krona::GeometryReal;
using krona::GlobalIndex;
using krona::mesh::CellType;
using Block = krona::distribution::BlockDistribution<GlobalIndex>;

void verify(bool condition, krona::mpi::CommView comm, const char *message)
{
    if (!condition)
        krona::mpi::fatal(comm.abort_scope(), message);
}

std::array<GeometryReal, 3> point(GlobalIndex id, const std::string &mode)
{
    constexpr std::array<std::array<double, 2>, 5> xy{{{0, 0}, {2, 0}, {2, 1}, {0, 1}, {3, 0}}};
    auto [x, y] = xy[static_cast<std::size_t>(id)];
    if (mode == "degenerate")
        y = 0;
    if (mode == "concave" && id == 2)
    {
        x = 0.5;
        y = 0.25;
    }
    if (mode == "xz")
        return {static_cast<GeometryReal>(x + 7), 5, static_cast<GeometryReal>(y + 11)};
    if (mode == "yz")
        return {5, static_cast<GeometryReal>(x + 7), static_cast<GeometryReal>(y + 11)};
    const auto z = mode == "warped" && id == 4
                       ? 2.0
                       : (mode == "collapse"
                              ? static_cast<double>(std::numeric_limits<GeometryReal>::max()) / 4
                              : 5.0);
    return {static_cast<GeometryReal>(x + 7), static_cast<GeometryReal>(y + 11),
            static_cast<GeometryReal>(z)};
}

int main(int argc, char **argv)
{
    krona::mpi::Runtime runtime(argc, argv);
    auto owner = krona::mpi::Communicator::duplicate(krona::mpi::CommView::world());
    const auto comm = owner.view();
    const std::string mode = argc > 1 ? argv[1] : "xy";
    krona::mesh::RawMesh mesh;
    mesh.cell_dimension = 2;
    mesh.physical_dimension = 3;
    mesh.distribution = {Block(2, comm.size()), Block(5, comm.size()),
                         Block(mode == "no-boundary" ? 0 : 4, comm.size()), comm.rank()};
    constexpr std::array<std::array<GlobalIndex, 4>, 2> cells{{{0, 3, 2, 1}, {1, 2, 4, -1}}};
    // Both cells intentionally have clockwise winding.
    for (auto id = mesh.distribution.cells.begin(comm.rank());
         id < mesh.distribution.cells.end(comm.rank()); ++id)
    {
        mesh.cells.types.push_back(id == 0 ? CellType::QUAD : CellType::TRI);
        const auto count = id == 0 ? 4U : 3U;
        mesh.cells.node_ids.insert(mesh.cells.node_ids.end(),
                                   cells[static_cast<std::size_t>(id)].begin(),
                                   cells[static_cast<std::size_t>(id)].begin() + count);
        mesh.cells.offsets.push_back(mesh.cells.node_ids.size());
    }
    for (auto id = mesh.distribution.nodes.begin(comm.rank());
         id < mesh.distribution.nodes.end(comm.rank()); ++id)
    {
        auto p = point(id, mode);
        mesh.coordinates.x.push_back(p[0]);
        mesh.coordinates.y.push_back(p[1]);
        mesh.coordinates.z.push_back(p[2]);
    }
    constexpr std::array<std::array<GlobalIndex, 2>, 4> edges{{{0, 1}, {1, 4}, {4, 2}, {2, 3}}};
    for (auto id = mesh.distribution.boundary_ordinals.begin(comm.rank());
         id < mesh.distribution.boundary_ordinals.end(comm.rank()); ++id)
    {
        mesh.boundary.elements.types.push_back(CellType::BAR);
        const auto &edge = edges[static_cast<std::size_t>(id)];
        mesh.boundary.elements.node_ids.insert(mesh.boundary.elements.node_ids.end(), edge.begin(),
                                               edge.end());
        mesh.boundary.elements.offsets.push_back(mesh.boundary.elements.node_ids.size());
        mesh.boundary.global_ids.push_back(20 + 3 * id);
        mesh.boundary.patch_ids.push_back(mode == "no-bc" ? krona::mesh::invalid_patch_id : 0);
    }
    if (mode != "no-bc")
        mesh.patches.push_back({"Wall", "MovingWall", "BCWall"});
    if (mode == "collision")
        mesh.patches[0].name = "Z_MIN";
    if (mode == "bad-node" && !mesh.cells.node_ids.empty())
        mesh.cells.node_ids[0] = 5;
    if (mode == "bad-csr" && !mesh.cells.node_ids.empty())
        mesh.cells.offsets.back() += 1;
    if (mode == "overflow")
        mesh.distribution.nodes = Block(std::numeric_limits<GlobalIndex>::max(), comm.size());
    krona::mesh::ExtrusionOptions options;
    options.batch_elements = 1; // Force repeated exchanges, including ranks without requests/tasks.
    if (mode == "custom")
    {
        options.thickness_fraction = 0.2;
        options.bottom_patch = "Bottom";
        options.top_patch = "Top";
    }
    const auto result = krona::mesh::extrude_one_layer(mesh, comm, options);
    const int axis = mode == "yz" ? 0 : (mode == "xz" ? 1 : 2);
    const int sign = axis == 1 ? -1 : 1;
    const auto expected_thickness = static_cast<GeometryReal>(options.thickness_fraction * 3.0);
    verify(result.thickness == expected_thickness && result.normal_axis == axis &&
               result.normal_sign == sign,
           comm, "bbox thickness/normal");
    verify(mesh.cell_dimension == 3 && mesh.physical_dimension == 3 &&
               mesh.distribution.nodes.global_size() == 10 &&
               mesh.distribution.cells.global_size() == 2 &&
               mesh.distribution.boundary_ordinals.global_size() == (mode == "no-boundary" ? 4 : 8),
           comm, "global counts/dimensions");
    for (std::size_t i = 0; i < mesh.coordinates.size(); ++i)
    {
        const auto id = mesh.distribution.nodes.begin(comm.rank()) + static_cast<GlobalIndex>(i);
        auto expected = point(id % 5, mode);
        if (id >= 5)
            expected[static_cast<std::size_t>(axis)] +=
                static_cast<GeometryReal>(sign) * expected_thickness;
        verify(mesh.coordinates.x[i] == expected[0] && mesh.coordinates.y[i] == expected[1] &&
                   mesh.coordinates.z[i] == expected[2],
               comm, "owned coordinates/frame preservation");
    }
    constexpr std::array<std::array<GlobalIndex, 4>, 2> oriented{{{0, 1, 2, 3}, {1, 4, 2, -1}}};
    for (std::size_t i = 0; i < mesh.cells.size(); ++i)
    {
        const auto id = mesh.distribution.cells.begin(comm.rank()) + static_cast<GlobalIndex>(i);
        const auto nodes = mesh.cells.nodes(i);
        const std::size_t n = id == 0 ? 4 : 3;
        verify(mesh.cells.types[i] == (id == 0 ? CellType::HEXA : CellType::PRISM), comm,
               "extruded cell type");
        for (std::size_t j = 0; j < n; ++j)
            verify(nodes[j] == oriented[static_cast<std::size_t>(id)][j] &&
                       nodes[n + j] == nodes[j] + 5,
                   comm, "positive cell connectivity/retained global IDs");
        // Signed volume from outward face triangulation, independent of expected connectivity.
        std::array<std::array<double, 3>, 8> xyz{};
        for (std::size_t j = 0; j < nodes.size(); ++j)
        {
            auto p = point(nodes[j] % 5, mode);
            if (nodes[j] >= 5)
                p[static_cast<std::size_t>(axis)] +=
                    static_cast<GeometryReal>(sign) * expected_thickness;
            for (std::size_t a = 0; a < 3; ++a)
                xyz[j][a] = p[a];
        }
        const auto type = static_cast<std::size_t>(mesh.cells.types[i]);
        double volume = 0;
        for (int f = 0; f < krona::mesh::faces_per_type[type]; ++f)
        {
            const auto &face = krona::mesh::face_table[type][f];
            for (int t = 1; t + 1 < krona::mesh::face_nodes[type][f]; ++t)
            {
                std::array<double, 3> a{}, b{}, c{};
                for (std::size_t d = 0; d < 3; ++d)
                {
                    a[d] = xyz[static_cast<std::size_t>(face[0])][d] - xyz[0][d];
                    b[d] = xyz[static_cast<std::size_t>(face[t])][d] - xyz[0][d];
                    c[d] = xyz[static_cast<std::size_t>(face[t + 1])][d] - xyz[0][d];
                }
                volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) +
                           a[2] * (b[0] * c[1] - b[1] * c[0])) /
                          6;
            }
        }
        const double analytic = (id == 0 ? 2.0 : 0.5) * expected_thickness;
        verify(volume > 0 && std::abs(volume - analytic) <=
                                 100 * std::numeric_limits<GeometryReal>::epsilon() * analytic,
               comm, "positive analytic volume");
    }
    verify(mesh.patches.size() == (mode == "no-bc" ? 2U : 3U), comm, "patch count");
    const auto base_patch = static_cast<krona::mesh::PatchId>(mesh.patches.size() - 2);
    verify(mesh.patches[static_cast<std::size_t>(base_patch)].name == options.bottom_patch &&
               mesh.patches.back().name == options.top_patch &&
               mesh.patches.back().source_type == "SymmetryPlane",
           comm, "cap metadata");
    if (mode != "no-bc")
        verify(mesh.patches[0].family == "MovingWall" && mesh.patches[0].source_type == "BCWall",
               comm, "original patch metadata");
    verify(mesh.boundary.size() == static_cast<std::size_t>(
                                       mesh.distribution.boundary_ordinals.local_size(comm.rank())),
           comm, "balanced boundaries");
    const GlobalIndex side_count = mode == "no-boundary" ? 0 : 4;
    const GlobalIndex cap_id = mode == "no-boundary" ? 2 : 30;
    for (std::size_t i = 0; i < mesh.boundary.size(); ++i)
    {
        const auto ordinal =
            mesh.distribution.boundary_ordinals.begin(comm.rank()) + static_cast<GlobalIndex>(i);
        const auto nodes = mesh.boundary.elements.nodes(i);
        if (ordinal < side_count)
        {
            const auto &edge = edges[static_cast<std::size_t>(ordinal)];
            const std::array<GlobalIndex, 4> expected{edge[0], edge[1], edge[1] + 5, edge[0] + 5};
            verify(std::equal(nodes.begin(), nodes.end(), expected.begin(), expected.end()) &&
                       mesh.boundary.global_ids[i] == 20 + 3 * ordinal &&
                       mesh.boundary.patch_ids[i] ==
                           (mode == "no-bc" ? krona::mesh::invalid_patch_id : 0),
                   comm, "side extrusion/optional BC");
        }
        else
        {
            const auto offset = ordinal - side_count;
            const bool upper = offset >= 2;
            const auto id = static_cast<std::size_t>(offset % 2);
            const std::size_t n = id == 0 ? 4 : 3;
            verify(nodes.size() == n && mesh.boundary.global_ids[i] == cap_id + offset &&
                       mesh.boundary.patch_ids[i] == base_patch + (upper ? 1 : 0),
                   comm, "cap identity/patch");
            for (std::size_t j = 0; j < n; ++j)
                verify(nodes[j] == (upper ? oriented[id][j] + 5 : oriented[id][(n - j) % n]), comm,
                       "outward cap winding");
        }
    }
    if (comm.rank() == 0)
        std::puts("PASS extrusion");
}
