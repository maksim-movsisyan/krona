#include "krona/mesh/extrusion.hpp"
#include "krona/mpi/collectives.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace krona::mesh
{
namespace
{
using Block = distribution::BlockDistribution<GlobalIndex>;

void expect(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class F> void prepare(mpi::CommView comm, const char *stage, F &&function)
{
    mpi::require_collective_success(comm, mpi::LocalStatus::capture(std::forward<F>(function)),
                                    stage);
}

std::size_t size(GlobalIndex count)
{
    expect(count >= 0 && std::in_range<std::size_t>(count), "Extrusion size exceeds size_t");
    return static_cast<std::size_t>(count);
}

std::size_t product(std::size_t count, std::size_t width)
{
    expect(count <= std::numeric_limits<std::size_t>::max() / width,
           "Extrusion allocation size overflows size_t");
    return count * width;
}

// Every receiver can accept one batch from every sender using MPI 3.1 int counts.
std::size_t batch_size(mpi::CommView comm, std::size_t requested, std::size_t width)
{
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                       static_cast<std::size_t>(comm.size()) / width;
    expect(requested > 0 && limit > 0, "Extrusion batch/communicator exceeds MPI 3.1 limits");
    return std::min(requested, limit);
}

void validate_table(const ElementTable &table, GlobalIndex nodes, bool boundary)
{
    expect(table.offsets.size() == table.size() + 1 && table.offsets.front() == 0 &&
               table.offsets.back() == table.node_ids.size(),
           "Invalid extrusion CSR offsets");
    for (std::size_t i = 0; i < table.size(); ++i)
    {
        const auto type = table.types[i];
        expect(boundary ? type == CellType::BAR : type == CellType::TRI || type == CellType::QUAD,
               "Extrusion requires TRI/QUAD cells and BAR boundaries");
        const auto count = boundary ? 2U : (type == CellType::TRI ? 3U : 4U);
        expect(table.offsets[i] <= table.offsets[i + 1] &&
                   table.offsets[i + 1] - table.offsets[i] == count,
               "Invalid extrusion element arity");
    }
    for (auto id : table.node_ids)
        expect(id >= 0 && id < nodes, "Invalid extrusion node ID");
}

struct NodeCache
{
    std::vector<GlobalIndex> ids;
    std::vector<GeometryReal> xyz;

    std::array<GeometryReal, 3> get(GlobalIndex id) const
    {
        const auto found = std::lower_bound(ids.begin(), ids.end(), id);
        expect(found != ids.end() && *found == id, "Missing extrusion node coordinate");
        const auto index = static_cast<std::size_t>(found - ids.begin()) * 3;
        return {xyz[index], xyz[index + 1], xyz[index + 2]};
    }
};

// Fetch only referenced nodes, with typed payloads and bounded exchange buffers.
NodeCache fetch_nodes(const RawMesh &mesh, const Block &output_nodes, mpi::CommView comm,
                      std::size_t requested_batch)
{
    NodeCache cache;
    std::vector<GlobalIndex> remote, incoming;
    std::vector<std::size_t> positions;
    std::vector<int> counts, response_counts;
    std::vector<GeometryReal> response, received;
    mpi::AlltoallvWorkspace requests_workspace, response_workspace;
    std::size_t batch = 0;
    const auto nodes = mesh.distribution.nodes.global_size();
    const auto local_begin = mesh.distribution.nodes.begin(comm.rank());
    const auto local_end = mesh.distribution.nodes.end(comm.rank());
    prepare(comm, "Extrusion node requests",
            [&]
            {
                batch = batch_size(comm, requested_batch, 3);
                cache.ids = mesh.cells.node_ids;
                const auto lo = output_nodes.begin(comm.rank());
                const auto hi = output_nodes.end(comm.rank());
                for (auto id = lo; id < hi; ++id)
                    cache.ids.push_back(id < nodes ? id : id - nodes);
                std::sort(cache.ids.begin(), cache.ids.end());
                cache.ids.erase(std::unique(cache.ids.begin(), cache.ids.end()), cache.ids.end());
                cache.xyz.resize(product(cache.ids.size(), 3));
                for (std::size_t i = 0; i < cache.ids.size(); ++i)
                {
                    const auto id = cache.ids[i];
                    if (id >= local_begin && id < local_end)
                    {
                        const auto source = size(id - local_begin);
                        cache.xyz[3 * i] = mesh.coordinates.x[source];
                        cache.xyz[3 * i + 1] = mesh.coordinates.y[source];
                        cache.xyz[3 * i + 2] = mesh.coordinates.z[source];
                    }
                    else
                    {
                        remote.push_back(id);
                        positions.push_back(i);
                    }
                }
                counts.resize(static_cast<std::size_t>(comm.size()));
                response_counts.resize(counts.size());
            });
    const auto rounds = mpi::allreduce<mpi::Reduction::max>(
        comm, static_cast<std::uint64_t>(remote.size() / batch + (remote.size() % batch != 0)));
    std::size_t first = 0;
    for (std::uint64_t round = 0; round < rounds; ++round)
    {
        const auto count = std::min(batch, remote.size() - first);
        prepare(
            comm, "Extrusion request packing",
            [&]
            {
                std::fill(counts.begin(), counts.end(), 0);
                for (std::size_t i = first; i < first + count; ++i)
                    ++counts[static_cast<std::size_t>(mesh.distribution.nodes.owner(remote[i]))];
            });
        mpi::alltoallv_packed(comm, std::span<const int>{counts},
                              std::span<const GlobalIndex>{remote}.subspan(first, count), incoming,
                              requests_workspace);
        prepare(comm, "Extrusion coordinate response",
                [&]
                {
                    response.resize(product(incoming.size(), 3));
                    for (std::size_t i = 0; i < incoming.size(); ++i)
                    {
                        expect(incoming[i] >= local_begin && incoming[i] < local_end,
                               "Extrusion node request reached wrong owner");
                        const auto source = size(incoming[i] - local_begin);
                        response[3 * i] = mesh.coordinates.x[source];
                        response[3 * i + 1] = mesh.coordinates.y[source];
                        response[3 * i + 2] = mesh.coordinates.z[source];
                    }
                    const auto received_counts = requests_workspace.received_counts();
                    for (std::size_t i = 0; i < counts.size(); ++i)
                        response_counts[i] = 3 * received_counts[i];
                });
        mpi::alltoallv_packed(comm, std::span<const int>{response_counts},
                              std::span<const GeometryReal>{response}, received,
                              response_workspace);
        prepare(comm, "Extrusion coordinate unpacking",
                [&]
                {
                    expect(received.size() == 3 * count,
                           "Invalid extrusion coordinate response size");
                    for (std::size_t i = 0; i < count; ++i)
                        for (std::size_t axis = 0; axis < 3; ++axis)
                            cache.xyz[3 * positions[first + i] + axis] = received[3 * i + axis];
                });
        first += count;
    }
    return cache;
}

// Positive, strictly convex projected polygons are required by PRISM/HEXA geometry.
bool reverse_cell(const NodeCache &cache, std::span<const GlobalIndex> ids, int u, int v,
                  double length)
{
    std::array<std::array<long double, 2>, 4> xy{};
    const auto origin = cache.get(ids[0]);
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        const auto point = cache.get(ids[i]);
        xy[i] = {(static_cast<long double>(point[static_cast<std::size_t>(u)]) -
                  origin[static_cast<std::size_t>(u)]) /
                     length,
                 (static_cast<long double>(point[static_cast<std::size_t>(v)]) -
                  origin[static_cast<std::size_t>(v)]) /
                     length};
    }
    int orientation = 0;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        const auto &a = xy[i], &b = xy[(i + 1) % ids.size()], &c = xy[(i + 2) % ids.size()];
        const auto cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        expect(std::isfinite(cross) && cross != 0, "Degenerate extrusion cell");
        const int sign = cross > 0 ? 1 : -1;
        expect(orientation == 0 || orientation == sign, "Non-convex or folded extrusion QUAD");
        orientation = sign;
    }
    return orientation < 0;
}

// Fixed-width, typed rows: ordinal, ID, patch, arity, and four node slots.
BoundaryTable build_boundary(const RawMesh &mesh, const Block &output, GlobalIndex first_cap_id,
                             PatchId bottom, PatchId top, mpi::CommView comm, std::size_t requested)
{
    constexpr std::size_t width = 8;
    BoundaryTable result;
    std::vector<GlobalIndex> send, recv;
    std::vector<int> counts;
    std::vector<std::size_t> offsets, cursor;
    mpi::AlltoallvWorkspace workspace;
    std::size_t batch = 0, tasks = 0, filled = 0;
    const auto nodes = mesh.distribution.nodes.global_size() / 2;
    const auto total_cells = mesh.distribution.cells.global_size();
    const auto old_boundaries = mesh.distribution.boundary_ordinals.global_size();
    const auto begin = output.begin(comm.rank());
    auto make_row = [&](std::size_t task)
    {
        std::array<GlobalIndex, width> row{};
        if (task < mesh.boundary.size())
        {
            const auto edge = mesh.boundary.elements.nodes(task);
            row = {mesh.distribution.boundary_ordinals.begin(comm.rank()) +
                       static_cast<GlobalIndex>(task),
                   mesh.boundary.global_ids[task],
                   mesh.boundary.patch_ids[task],
                   4,
                   edge[0],
                   edge[1],
                   edge[1] + nodes,
                   edge[0] + nodes};
        }
        else
        {
            const auto offset = task - mesh.boundary.size();
            const auto cell = offset / 2;
            const bool upper = offset % 2 != 0;
            const auto id =
                mesh.distribution.cells.begin(comm.rank()) + static_cast<GlobalIndex>(cell);
            const auto volume = mesh.cells.nodes(cell);
            const auto arity = volume.size() / 2;
            row = {old_boundaries + (upper ? total_cells : 0) + id,
                   first_cap_id + (upper ? total_cells : 0) + id,
                   upper ? top : bottom,
                   static_cast<GlobalIndex>(arity),
                   -1,
                   -1,
                   -1,
                   -1};
            for (std::size_t i = 0; i < arity; ++i)
                row[4 + i] = upper ? volume[arity + i] : volume[(arity - i) % arity];
        }
        return row;
    };
    prepare(comm, "Extrusion boundary allocation",
            [&]
            {
                batch = batch_size(comm, requested, width);
                expect(mesh.cells.size() <=
                           (std::numeric_limits<std::size_t>::max() - mesh.boundary.size()) / 2,
                       "Extrusion task count overflows size_t");
                tasks = mesh.boundary.size() + 2 * mesh.cells.size();
                const auto count = size(output.local_size(comm.rank()));
                expect(std::in_range<LocalIndex>(count),
                       "Extrusion boundary count exceeds LocalIndex");
                result.elements.types.resize(count);
                result.elements.offsets.resize(count + 1);
                result.elements.node_ids.resize(product(count, 4));
                result.global_ids.assign(count, -1);
                result.patch_ids.resize(count);
                counts.resize(static_cast<std::size_t>(comm.size()));
                offsets.resize(counts.size());
                cursor.resize(counts.size());
            });
    const auto rounds = mpi::allreduce<mpi::Reduction::max>(
        comm, static_cast<std::uint64_t>(tasks / batch + (tasks % batch != 0)));
    std::size_t first = 0;
    for (std::uint64_t round = 0; round < rounds; ++round)
    {
        const auto count = std::min(batch, tasks - first);
        prepare(comm, "Extrusion boundary packing",
                [&]
                {
                    std::fill(counts.begin(), counts.end(), 0);
                    for (std::size_t i = first; i < first + count; ++i)
                        counts[static_cast<std::size_t>(output.owner(make_row(i)[0]))] +=
                            static_cast<int>(width);
                    std::size_t offset = 0;
                    for (std::size_t p = 0; p < counts.size(); ++p)
                    {
                        offsets[p] = offset;
                        offset += static_cast<std::size_t>(counts[p]);
                    }
                    cursor = offsets;
                    send.resize(count * width);
                    for (std::size_t i = first; i < first + count; ++i)
                    {
                        const auto row = make_row(i);
                        auto &position = cursor[static_cast<std::size_t>(output.owner(row[0]))];
                        std::copy(row.begin(), row.end(),
                                  send.begin() + static_cast<std::ptrdiff_t>(position));
                        position += width;
                    }
                });
        mpi::alltoallv_packed(comm, std::span<const int>{counts},
                              std::span<const GlobalIndex>{send}, recv, workspace);
        prepare(comm, "Extrusion boundary unpacking",
                [&]
                {
                    expect(recv.size() % width == 0, "Invalid extrusion boundary response size");
                    for (std::size_t offset = 0; offset < recv.size(); offset += width)
                    {
                        expect(recv[offset] >= begin && recv[offset] < output.end(comm.rank()),
                               "Extrusion boundary reached wrong owner");
                        const auto i = size(recv[offset] - begin);
                        expect(result.global_ids[i] == -1, "Duplicate extrusion boundary ordinal");
                        const auto arity = recv[offset + 3];
                        expect(arity == 3 || arity == 4, "Invalid extrusion boundary arity");
                        result.elements.types[i] = arity == 3 ? CellType::TRI : CellType::QUAD;
                        result.global_ids[i] = recv[offset + 1];
                        result.patch_ids[i] = static_cast<PatchId>(recv[offset + 2]);
                        for (std::size_t j = 0; j < 4; ++j)
                            result.elements.node_ids[4 * i + j] = recv[offset + 4 + j];
                        ++filled;
                    }
                });
        first += count;
    }
    prepare(comm, "Extrusion boundary CSR",
            [&]
            {
                expect(filled == result.size(), "Missing extrusion boundary ordinal");
                std::size_t offset = 0;
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    result.elements.offsets[i] = offset;
                    const std::size_t arity = result.elements.types[i] == CellType::TRI ? 3 : 4;
                    for (std::size_t j = 0; j < arity; ++j)
                        result.elements.node_ids[offset + j] = result.elements.node_ids[4 * i + j];
                    offset += arity;
                }
                result.elements.offsets.back() = offset;
                result.elements.node_ids.resize(offset);
            });
    return result;
}
} // namespace

ExtrusionInfo extrude_one_layer(RawMesh &mesh, mpi::CommView comm, const ExtrusionOptions &options)
{
    GlobalIndex nodes = 0, cells = 0, boundaries = 0, local_max_id = -1;
    std::array<double, 3> low{}, high{};
    prepare(
        comm, "Extrusion input",
        [&]
        {
            expect(!comm.is_null() && !comm.is_intercommunicator(),
                   "Extrusion needs an intracommunicator");
            expect(mesh.cell_dimension == 2 &&
                       (mesh.physical_dimension == 2 || mesh.physical_dimension == 3),
                   "Extrusion requires a 2D mesh");
            expect(std::isfinite(options.thickness_fraction) && options.thickness_fraction > 0 &&
                       std::isfinite(options.planarity_tolerance) &&
                       options.planarity_tolerance >= 0 && options.planarity_tolerance < 1,
                   "Invalid extrusion thickness/planarity tolerance");
            expect(!options.bottom_patch.empty() && !options.top_patch.empty() &&
                       options.bottom_patch != options.top_patch,
                   "Invalid extrusion cap names");
            expect(mesh.patches.size() <=
                       static_cast<std::size_t>(std::numeric_limits<PatchId>::max()) - 1,
                   "Extrusion patch IDs overflow");
            for (const auto &patch : mesh.patches)
                expect(patch.name != options.bottom_patch && patch.name != options.top_patch,
                       "Extrusion cap name collides with an existing patch");
            const auto &distribution = mesh.distribution;
            expect(distribution.rank == comm.rank() && distribution.nodes.parts() == comm.size() &&
                       distribution.cells.parts() == comm.size() &&
                       distribution.boundary_ordinals.parts() == comm.size(),
                   "Extrusion requires matching block ownership");
            nodes = distribution.nodes.global_size();
            cells = distribution.cells.global_size();
            boundaries = distribution.boundary_ordinals.global_size();
            const auto maximum = std::numeric_limits<GlobalIndex>::max();
            expect(nodes > 0 && cells > 0 && nodes <= maximum / 2 &&
                       cells <= (maximum - boundaries) / 2,
                   "Extrusion global counts are empty or overflow GlobalIndex");
            expect(mesh.coordinates.size() == size(distribution.nodes.local_size(comm.rank())) &&
                       mesh.coordinates.y.size() == mesh.coordinates.size() &&
                       mesh.coordinates.z.size() == mesh.coordinates.size() &&
                       mesh.cells.size() == size(distribution.cells.local_size(comm.rank())) &&
                       mesh.boundary.size() ==
                           size(distribution.boundary_ordinals.local_size(comm.rank())) &&
                       mesh.boundary.global_ids.size() == mesh.boundary.size() &&
                       mesh.boundary.patch_ids.size() == mesh.boundary.size(),
                   "Invalid extrusion owned arrays");
            validate_table(mesh.cells, nodes, false);
            validate_table(mesh.boundary.elements, nodes, true);
            for (std::size_t i = 0; i < mesh.boundary.size(); ++i)
            {
                expect(mesh.boundary.global_ids[i] >= cells, "Boundary ID overlaps cell IDs");
                local_max_id = std::max(local_max_id, mesh.boundary.global_ids[i]);
                const auto patch = mesh.boundary.patch_ids[i];
                expect(patch == invalid_patch_id ||
                           (patch >= 0 && static_cast<std::size_t>(patch) < mesh.patches.size()),
                       "Invalid extrusion boundary patch ID");
            }
            low.fill(std::numeric_limits<double>::infinity());
            high.fill(-std::numeric_limits<double>::infinity());
            for (std::size_t i = 0; i < mesh.coordinates.size(); ++i)
            {
                const std::array<double, 3> point{mesh.coordinates.x[i], mesh.coordinates.y[i],
                                                  mesh.coordinates.z[i]};
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    expect(std::isfinite(point[axis]), "Non-finite extrusion coordinate");
                    low[axis] = std::min(low[axis], point[axis]);
                    high[axis] = std::max(high[axis], point[axis]);
                }
            }
        });
    mpi::allreduce_in_place<mpi::Reduction::min>(comm, std::span<double>{low});
    mpi::allreduce_in_place<mpi::Reduction::max>(comm, std::span<double>{high});
    const auto maximum_id =
        std::max(cells - 1, mpi::allreduce<mpi::Reduction::max>(comm, local_max_id));
    ExtrusionInfo info;
    Block output_nodes, output_boundary;
    double length = 0;
    int u = 0, v = 1;
    prepare(comm, "Extrusion geometry contract",
            [&]
            {
                std::array<double, 3> spans{};
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    spans[axis] = high[axis] - low[axis];
                    expect(std::isfinite(spans[axis]), "Extrusion bbox span overflows");
                }
                length = *std::max_element(spans.begin(), spans.end());
                expect(length > 0, "Degenerate extrusion bbox");
                const auto axis = static_cast<std::size_t>(
                    std::min_element(spans.begin(), spans.end()) - spans.begin());
                expect(spans[axis] / length <= options.planarity_tolerance,
                       "Extrusion requires an axis-aligned planar mesh");
                info.normal_axis = static_cast<int>(axis);
                info.normal_sign = axis == 1 ? -1 : 1;
                u = axis == 0 ? 1 : 0;
                v = axis == 2 ? 1 : 2;
                const double h = options.thickness_fraction * length;
                expect(std::isfinite(h) && h > 0 && h <= std::numeric_limits<GeometryReal>::max(),
                       "Extrusion thickness exceeds GeometryReal");
                info.thickness = static_cast<GeometryReal>(h);
                expect(info.thickness > 0, "Extrusion thickness underflows GeometryReal");
                expect(maximum_id <= std::numeric_limits<GlobalIndex>::max() - 2 * cells,
                       "Extrusion cap IDs overflow GlobalIndex");
                output_nodes = Block(2 * nodes, comm.size());
                output_boundary = Block(boundaries + 2 * cells, comm.size());
                expect(std::in_range<LocalIndex>(output_nodes.local_size(comm.rank())),
                       "Extrusion owned nodes exceed LocalIndex");
            });
    auto cache = fetch_nodes(mesh, output_nodes, comm, options.batch_elements);
    Coordinates coordinates;
    std::vector<bool> reverse;
    prepare(comm, "Extrusion local geometry",
            [&]
            {
                reverse.resize(mesh.cells.size());
                for (std::size_t i = 0; i < mesh.cells.size(); ++i)
                    reverse[i] = reverse_cell(cache, mesh.cells.nodes(i), u, v, length);
                const auto count = size(output_nodes.local_size(comm.rank()));
                coordinates.x.resize(count);
                coordinates.y.resize(count);
                coordinates.z.resize(count);
                const auto axis = static_cast<std::size_t>(info.normal_axis);
                const auto shift = static_cast<GeometryReal>(info.normal_sign) * info.thickness;
                for (std::size_t i = 0; i < count; ++i)
                {
                    const auto id = output_nodes.begin(comm.rank()) + static_cast<GlobalIndex>(i);
                    auto point = cache.get(id < nodes ? id : id - nodes);
                    if (id >= nodes)
                    {
                        const auto original = point[axis];
                        point[axis] += shift;
                        expect(std::isfinite(point[axis]) && point[axis] != original,
                               "Extrusion layer collapses/overflows at GeometryReal precision");
                    }
                    coordinates.x[i] = point[0];
                    coordinates.y[i] = point[1];
                    coordinates.z[i] = point[2];
                }
            });
    // Release the coordinate cache before expanding connectivity and boundaries.
    cache = {};
    const auto bottom = static_cast<PatchId>(mesh.patches.size());
    const auto top = static_cast<PatchId>(bottom + 1);
    prepare(comm, "Extrusion volume connectivity",
            [&]
            {
                const auto old_size = mesh.cells.node_ids.size();
                mesh.cells.node_ids.resize(product(old_size, 2));
                for (std::size_t i = mesh.cells.size(); i-- > 0;)
                {
                    const auto start = mesh.cells.offsets[i];
                    const std::size_t arity = mesh.cells.types[i] == CellType::TRI ? 3 : 4;
                    std::array<GlobalIndex, 4> ids{};
                    std::copy_n(mesh.cells.node_ids.begin() + static_cast<std::ptrdiff_t>(start),
                                arity, ids.begin());
                    if (reverse[i])
                        std::reverse(ids.begin() + 1,
                                     ids.begin() + static_cast<std::ptrdiff_t>(arity));
                    for (std::size_t j = 0; j < arity; ++j)
                    {
                        mesh.cells.node_ids[2 * start + j] = ids[j];
                        mesh.cells.node_ids[2 * start + arity + j] = ids[j] + nodes;
                    }
                    mesh.cells.types[i] = arity == 3 ? CellType::PRISM : CellType::HEXA;
                    mesh.cells.offsets[i] = 2 * start;
                }
                mesh.cells.offsets.back() = 2 * old_size;
                mesh.patches.push_back({options.bottom_patch, {}, "SymmetryPlane"});
                mesh.patches.push_back({options.top_patch, {}, "SymmetryPlane"});
                mesh.coordinates = std::move(coordinates);
                mesh.distribution.nodes = output_nodes;
            });
    auto boundary = build_boundary(mesh, output_boundary, maximum_id + 1, bottom, top, comm,
                                   options.batch_elements);
    prepare(comm, "Extrusion commit",
            [&]
            {
                mesh.boundary = std::move(boundary);
                mesh.distribution.boundary_ordinals = output_boundary;
                mesh.cell_dimension = 3;
                mesh.physical_dimension = 3;
            });
    return info;
}
} // namespace krona::mesh
