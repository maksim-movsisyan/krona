#include <krona/mesh/extrusion.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <utility>
#include <vector>
 
namespace krona::mesh {
namespace {
using Block = distribution::BlockDistribution<GlobalIndex>;

/// @brief Rejects invalid geometry or an unrepresentable output size.
void ensure(bool condition, const char* message,
            std::source_location where = std::source_location::current()) noexcept
{
    // Abort WORLD directly; no status collective is needed on the failure path.
    if (!condition) mpi::fatal(message, where);
}

/// @brief Converts a nonnegative entity count to an allocation size.
[[nodiscard]] std::size_t size(GlobalIndex count)
{
    ensure(count >= 0 && std::in_range<std::size_t>(count), "Extrusion size exceeds size_t");
    return static_cast<std::size_t>(count);
}

/// @brief Computes a buffer size without overflowing host indices.
[[nodiscard]] std::size_t product(std::size_t count, std::size_t width)
{
    ensure(count <= std::numeric_limits<std::size_t>::max() / width,
           "Extrusion allocation size overflows size_t");
    return count * width;
}

/// @brief Bounds exchange batches even when every sender targets the same receiver.
[[nodiscard]] std::size_t batch_size(mpi::CommView comm, std::size_t requested, std::size_t width)
{
    // MPI 3.1 represents counts and displacements as int.
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                       static_cast<std::size_t>(comm.size()) / width;
    ensure(requested > 0 && limit > 0, "Extrusion batch/communicator exceeds MPI 3.1 limits");
    return std::min(requested, limit);
}

/// @brief Stores only coordinates referenced by local cells or output node ownership.
struct NodeCache {
    std::vector<GlobalIndex> ids;
    std::vector<GeometryReal> xyz;

    /// @brief Looks up a coordinate in the sorted request cache.
    [[nodiscard]] std::array<GeometryReal, 3> get(GlobalIndex id) const noexcept
    {
        const auto found = std::lower_bound(ids.begin(), ids.end(), id);
        assert(found != ids.end() && *found == id);
        const auto index = static_cast<std::size_t>(found - ids.begin()) * 3;
        return {xyz[index], xyz[index + 1], xyz[index + 2]};
    }
};

/// @brief Fetches remote coordinates with bounded requests and reuses the return layout.
[[nodiscard]] NodeCache fetch_nodes(const RawMesh& mesh, const Block& output_nodes,
                                    mpi::CommView comm, std::size_t requested_batch)
{
    NodeCache cache;
    const auto nodes = mesh.distribution.nodes.global_size();
    const auto local_begin = mesh.distribution.nodes.begin(comm.rank());
    const auto local_end = mesh.distribution.nodes.end(comm.rank());
    const auto batch = batch_size(comm, requested_batch, 3);
    const auto lo = output_nodes.begin(comm.rank()), hi = output_nodes.end(comm.rank());
    const auto output_count = size(hi - lo);
    ensure(mesh.cells.node_ids.size() <= cache.ids.max_size() &&
               output_count <= cache.ids.max_size() - mesh.cells.node_ids.size(),
           "Extrusion node requests exceed host storage");
    cache.ids.reserve(mesh.cells.node_ids.size() + output_count);
    cache.ids.insert(cache.ids.end(), mesh.cells.node_ids.begin(), mesh.cells.node_ids.end());
    for (auto id = lo; id < hi; ++id) cache.ids.push_back(id < nodes ? id : id - nodes);
    std::sort(cache.ids.begin(), cache.ids.end());
    cache.ids.erase(std::unique(cache.ids.begin(), cache.ids.end()), cache.ids.end());
    cache.xyz.resize(product(cache.ids.size(), 3));
    std::vector<GlobalIndex> remote, incoming;
    std::vector<std::size_t> positions;
    const auto owned_first = std::lower_bound(cache.ids.begin(), cache.ids.end(), local_begin);
    const auto owned_last = std::lower_bound(owned_first, cache.ids.end(), local_end);
    const auto remote_count = cache.ids.size() - static_cast<std::size_t>(owned_last - owned_first);
    remote.reserve(remote_count);
    positions.reserve(remote_count);
    for (std::size_t i = 0; i < cache.ids.size(); ++i) {
        const auto id = cache.ids[i];
        if (id >= local_begin && id < local_end) {
            const auto source = size(id - local_begin);
            cache.xyz[3 * i] = mesh.coordinates.x[source];
            cache.xyz[3 * i + 1] = mesh.coordinates.y[source];
            cache.xyz[3 * i + 2] = mesh.coordinates.z[source];
        } else {
            remote.push_back(id);
            positions.push_back(i);
        }
    }
    const auto peers = static_cast<std::size_t>(comm.size());
    std::vector<int> counts(peers), send_counts(peers), recv_counts(peers);
    std::vector<int> send_offsets(peers), recv_offsets(peers);
    std::vector<GeometryReal> response, received;
    mpi::AlltoallvWorkspace requests;
    // This reduction matches communication rounds for ranks with different request counts.
    const auto rounds = mpi::allreduce<mpi::Reduction::max>(
        comm, static_cast<std::uint64_t>(remote.size() / batch + (remote.size() % batch != 0)));
    std::size_t first = 0;
    for (std::uint64_t round = 0; round < rounds; ++round) {
        const auto count = std::min(batch, remote.size() - first);
        std::fill(counts.begin(), counts.end(), 0);
        // Sorted IDs are already grouped by their contiguous block owners.
        for (std::size_t i = first; i < first + count; ++i)
            ++counts[static_cast<std::size_t>(mesh.distribution.nodes.owner(remote[i]))];
        mpi::alltoallv_packed(comm, std::span<const int>{counts},
                             std::span<const GlobalIndex>{remote}.subspan(first, count),
                             incoming, requests);
        response.resize(product(incoming.size(), 3));
        for (std::size_t i = 0; i < incoming.size(); ++i) {
            assert(incoming[i] >= local_begin && incoming[i] < local_end);
            const auto source = size(incoming[i] - local_begin);
            response[3 * i] = mesh.coordinates.x[source];
            response[3 * i + 1] = mesh.coordinates.y[source];
            response[3 * i + 2] = mesh.coordinates.z[source];
        }
        received.resize(3 * count);
        for (std::size_t peer = 0; peer < peers; ++peer) {
            // Reverse the known request layout; do not exchange response counts again.
            send_counts[peer] = 3 * requests.recv_counts[peer];
            send_offsets[peer] = 3 * requests.recv_displacements[peer];
            recv_counts[peer] = 3 * requests.send_counts[peer];
            recv_offsets[peer] = 3 * requests.send_displacements[peer];
        }
        mpi::alltoallv(comm, std::span<const GeometryReal>{response}, send_counts, send_offsets,
                      std::span<GeometryReal>{received}, recv_counts, recv_offsets);
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t axis = 0; axis < 3; ++axis)
                cache.xyz[3 * positions[first + i] + axis] = received[3 * i + axis];
        first += count;
    }
    return cache;
}

/// @brief Finds a reproducible normal and verifies that owned coordinates lie in one plane.
[[nodiscard]] std::array<double, 3> plane_normal(const RawMesh& mesh, const NodeCache& cache,
                                               mpi::CommView comm, double length,
                                               const ExtrusionOptions& options)
{
    std::array<double, 9> sample{};
    if (mesh.distribution.cells.begin(comm.rank()) == 0 && !mesh.cells.types.empty()) {
        const auto ids = mesh.cells.nodes(0);
        for (std::size_t i = 0; i < 3; ++i) {
            const auto point = cache.get(ids[i]);
            for (std::size_t axis = 0; axis < 3; ++axis) sample[3 * i + axis] = point[axis];
        }
    }
    // Only the owner of cell zero contributes this constant-size geometry sample.
    mpi::allreduce_in_place<mpi::Reduction::sum>(comm, std::span<double>{sample});
    auto normal = options.direction;
    for (const auto component : normal) ensure(std::isfinite(component), "Invalid extrusion direction");
    if (normal == std::array<double, 3>{}) {
        std::array<double, 3> a{}, b{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            a[axis] = (sample[3 + axis] - sample[axis]) / length;
            b[axis] = (sample[6 + axis] - sample[axis]) / length;
        }
        normal = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0]};
        const auto axis = static_cast<std::size_t>(std::max_element(normal.begin(), normal.end(),
            [](double left, double right) { return std::abs(left) < std::abs(right); }) - normal.begin());
        // Preserve the conventional +X, -Y, +Z directions for coordinate planes.
        if ((normal[axis] < 0) != (axis == 1))
            for (auto& component : normal) component = -component;
    }
    const auto norm = std::hypot(normal[0], normal[1], normal[2]);
    ensure(std::isfinite(norm) && norm > 0, "Degenerate extrusion plane or direction");
    for (auto& component : normal) component /= norm;
    double deviation = 0;
    for (std::size_t i = 0; i < mesh.coordinates.size(); ++i) {
        const std::array<double, 3> point{mesh.coordinates.x[i], mesh.coordinates.y[i], mesh.coordinates.z[i]};
        double distance = 0;
        for (std::size_t axis = 0; axis < 3; ++axis)
            distance += ((point[axis] - sample[axis]) / length) * normal[axis];
        deviation = std::max(deviation, std::abs(distance));
    }
    deviation = mpi::allreduce<mpi::Reduction::max>(comm, deviation);
    ensure(deviation <= options.planarity_tolerance,
           "Extrusion requires a planar mesh and a direction normal to its plane");
    return normal;
}

/// @brief Checks a TRI/QUAD's convex orientation relative to the extrusion normal.
[[nodiscard]] bool reverse_cell(const NodeCache& cache, std::span<const GlobalIndex> ids,
                                const std::array<double, 3>& normal, double length)
{
    std::array<std::array<long double, 3>, 4> points{};
    const auto origin = cache.get(ids[0]);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto point = cache.get(ids[i]);
        for (std::size_t axis = 0; axis < 3; ++axis)
            points[i][axis] = (static_cast<long double>(point[axis]) - origin[axis]) / length;
    }
    int orientation = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::array<long double, 3> a{}, b{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            a[axis] = points[(i + 1) % ids.size()][axis] - points[i][axis];
            b[axis] = points[(i + 2) % ids.size()][axis] - points[(i + 1) % ids.size()][axis];
        }
        const auto turn = (a[1] * b[2] - a[2] * b[1]) * normal[0] +
                          (a[2] * b[0] - a[0] * b[2]) * normal[1] +
                          (a[0] * b[1] - a[1] * b[0]) * normal[2];
        ensure(std::isfinite(turn) && turn != 0, "Degenerate extrusion cell");
        const int sign = turn > 0 ? 1 : -1;
        ensure(orientation == 0 || orientation == sign, "Non-convex or folded extrusion QUAD");
        orientation = sign;
    }
    return orientation < 0;
}

/// @brief Routes side faces and caps to their balanced output boundary owners.
[[nodiscard]] BoundaryTable build_boundary(const RawMesh& mesh, const Block& output,
                                           GlobalIndex first_cap_id, PatchId bottom, PatchId top,
                                           mpi::CommView comm, std::size_t requested)
{
    // Fixed-width rows carry ordinal, source ID, patch, arity and four node slots.
    constexpr std::size_t width = 8;
    const auto nodes = mesh.distribution.nodes.global_size() / 2;
    const auto total_cells = mesh.distribution.cells.global_size();
    const auto old_boundaries = mesh.distribution.boundary_ordinals.global_size();
    const auto begin = output.begin(comm.rank());
    const auto batch = batch_size(comm, requested, width);
    const auto cap_tasks = product(mesh.cells.size(), 2);
    ensure(mesh.boundary.size() <= std::numeric_limits<std::size_t>::max() - cap_tasks,
           "Extrusion boundary task count overflows size_t");
    const auto tasks = mesh.boundary.size() + cap_tasks;
    const auto count = size(output.local_size(comm.rank()));
    ensure(std::in_range<LocalIndex>(count), "Extrusion owned boundary count exceeds LocalIndex");
    BoundaryTable result;
    result.elements.types.resize(count);
    result.elements.offsets.resize(count + 1);
    result.elements.node_ids.resize(product(count, 4));
    result.global_ids.assign(count, -1);
    result.patch_ids.resize(count);
    std::vector<GlobalIndex> send, recv;
    std::vector<int> counts(static_cast<std::size_t>(comm.size()));
    std::vector<std::size_t> cursor(counts.size());
    mpi::AlltoallvWorkspace workspace;
    const auto make_row = [&](std::size_t task) {
        std::array<GlobalIndex, width> row{};
        if (task < mesh.boundary.size()) {
            const auto edge = mesh.boundary.elements.nodes(task);
            row = {mesh.distribution.boundary_ordinals.begin(comm.rank()) + static_cast<GlobalIndex>(task),
                   mesh.boundary.global_ids[task], mesh.boundary.patch_ids[task], 4,
                   edge[0], edge[1], edge[1] + nodes, edge[0] + nodes};
        } else {
            const auto offset = task - mesh.boundary.size(), cell = offset / 2;
            const bool upper = offset % 2 != 0;
            const auto id = mesh.distribution.cells.begin(comm.rank()) + static_cast<GlobalIndex>(cell);
            const auto volume = mesh.cells.nodes(cell);
            const auto arity = volume.size() / 2;
            row = {old_boundaries + (upper ? total_cells : 0) + id,
                   first_cap_id + (upper ? total_cells : 0) + id,
                   upper ? top : bottom, static_cast<GlobalIndex>(arity), -1, -1, -1, -1};
            // Caps follow outward CGNS facet ordering after volume orientation is corrected.
            for (std::size_t i = 0; i < arity; ++i)
                row[4 + i] = upper ? volume[arity + i] : volume[(arity - i) % arity];
        }
        return row;
    };
    const auto rounds = mpi::allreduce<mpi::Reduction::max>(
        comm, static_cast<std::uint64_t>(tasks / batch + (tasks % batch != 0)));
    std::size_t first = 0;
    [[maybe_unused]] std::size_t filled = 0;
    for (std::uint64_t round = 0; round < rounds; ++round) {
        const auto n = std::min(batch, tasks - first);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = first; i < first + n; ++i)
            counts[static_cast<std::size_t>(output.owner(make_row(i)[0]))] += static_cast<int>(width);
        std::size_t prefix = 0;
        for (std::size_t peer = 0; peer < counts.size(); ++peer) {
            cursor[peer] = prefix;
            prefix += static_cast<std::size_t>(counts[peer]);
        }
        send.resize(n * width);
        for (std::size_t i = first; i < first + n; ++i) {
            const auto row = make_row(i);
            auto& position = cursor[static_cast<std::size_t>(output.owner(row[0]))];
            std::copy(row.begin(), row.end(), send.begin() + static_cast<std::ptrdiff_t>(position));
            position += width;
        }
        mpi::alltoallv_packed(comm, std::span<const int>{counts}, std::span<const GlobalIndex>{send}, recv, workspace);
        assert(recv.size() % width == 0);
        for (std::size_t offset = 0; offset < recv.size(); offset += width) {
            assert(output.owner(recv[offset]) == comm.rank());
            const auto i = size(recv[offset] - begin);
            assert(result.global_ids[i] == -1);
            const auto arity = recv[offset + 3];
            assert(arity == 3 || arity == 4);
            result.elements.types[i] = arity == 3 ? CellType::TRI : CellType::QUAD;
            result.global_ids[i] = recv[offset + 1];
            result.patch_ids[i] = static_cast<PatchId>(recv[offset + 2]);
            std::copy_n(recv.begin() + static_cast<std::ptrdiff_t>(offset + 4), 4,
                        result.elements.node_ids.begin() + static_cast<std::ptrdiff_t>(4 * i));
            ++filled;
        }
        first += n;
    }
    assert(filled == result.size());
    // Compact the four-node slots in place to the final TRI/QUAD CSR layout.
    std::size_t offset = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result.elements.offsets[i] = offset;
        const std::size_t arity = result.elements.types[i] == CellType::TRI ? 3 : 4;
        for (std::size_t j = 0; j < arity; ++j)
            result.elements.node_ids[offset + j] = result.elements.node_ids[4 * i + j];
        offset += arity;
    }
    result.elements.offsets.back() = offset;
    result.elements.node_ids.resize(offset);
    return result;
}

/// @brief Converts a validated import snapshot into one 3D extrusion layer.
[[nodiscard]] ExtrusionInfo extrude(RawMesh& mesh, mpi::CommView comm, const ExtrusionOptions& options)
{
    ensure(mesh.cell_dimension == 2 && (mesh.physical_dimension == 2 || mesh.physical_dimension == 3),
           "Extrusion requires a 2D mesh");
    ensure(std::isfinite(options.thickness_fraction) && options.thickness_fraction > 0 &&
               std::isfinite(options.planarity_tolerance) && options.planarity_tolerance >= 0 &&
               options.planarity_tolerance < 1, "Invalid extrusion thickness/planarity tolerance");
    ensure(!options.bottom_patch.empty() && !options.top_patch.empty() && options.bottom_patch != options.top_patch,
           "Invalid extrusion cap names");
    ensure(mesh.patches.size() <= static_cast<std::size_t>(std::numeric_limits<PatchId>::max()) - 1,
           "Extrusion patch IDs overflow");
    for (const auto& patch : mesh.patches)
        ensure(patch.name != options.bottom_patch && patch.name != options.top_patch,
               "Extrusion cap name collides with an existing patch");
    const auto& dist = mesh.distribution;
    assert(dist.rank == comm.rank() && dist.nodes.parts() == comm.size() &&
           dist.cells.parts() == comm.size() && dist.boundary_ordinals.parts() == comm.size());
    const auto nodes = dist.nodes.global_size(), cells = dist.cells.global_size();
    const auto boundaries = dist.boundary_ordinals.global_size();
    const auto maximum = std::numeric_limits<GlobalIndex>::max();
    ensure(nodes > 0 && cells > 0 && nodes <= maximum / 2 && cells <= (maximum - boundaries) / 2,
           "Extrusion global counts are empty or overflow GlobalIndex");
    assert(mesh.coordinates.size() == size(dist.nodes.local_size(comm.rank())) &&
           mesh.coordinates.y.size() == mesh.coordinates.size() && mesh.coordinates.z.size() == mesh.coordinates.size());
    assert(mesh.cells.size() == size(dist.cells.local_size(comm.rank())) &&
           mesh.boundary.size() == size(dist.boundary_ordinals.local_size(comm.rank())) &&
           mesh.boundary.global_ids.size() == mesh.boundary.size() && mesh.boundary.patch_ids.size() == mesh.boundary.size());
#ifndef NDEBUG
    for (const auto type : mesh.cells.types) assert(type == CellType::TRI || type == CellType::QUAD);
    for (const auto type : mesh.boundary.elements.types) assert(type == CellType::BAR);
#endif
    std::array<double, 3> low{}, high{};
    low.fill(std::numeric_limits<double>::infinity());
    high.fill(-std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < mesh.coordinates.size(); ++i) {
        const std::array<double, 3> point{mesh.coordinates.x[i], mesh.coordinates.y[i], mesh.coordinates.z[i]};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            ensure(std::isfinite(point[axis]), "Non-finite extrusion coordinate");
            low[axis] = std::min(low[axis], point[axis]);
            high[axis] = std::max(high[axis], point[axis]);
        }
    }
    // Geometry reductions are required to choose one thickness on every rank.
    mpi::allreduce_in_place<mpi::Reduction::min>(comm, std::span<double>{low});
    mpi::allreduce_in_place<mpi::Reduction::max>(comm, std::span<double>{high});
    double length = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto span = high[axis] - low[axis];
        ensure(std::isfinite(span), "Extrusion bbox span overflows");
        length = std::max(length, span);
    }
    ensure(length > 0, "Degenerate extrusion bbox");
    GlobalIndex maximum_id = cells - 1;
    for (const auto id : mesh.boundary.global_ids) maximum_id = std::max(maximum_id, id);
    maximum_id = mpi::allreduce<mpi::Reduction::max>(comm, maximum_id);
    ensure(maximum_id <= maximum - 2 * cells, "Extrusion cap IDs overflow GlobalIndex");
    const Block output_nodes{2 * nodes, comm.size()}, output_boundary{boundaries + 2 * cells, comm.size()};
    ensure(std::in_range<LocalIndex>(output_nodes.local_size(comm.rank())), "Extrusion owned nodes exceed LocalIndex");
    auto cache = fetch_nodes(mesh, output_nodes, comm, options.batch_elements);
    ExtrusionInfo info;
    info.normal = plane_normal(mesh, cache, comm, length, options);
    info.normal_axis = static_cast<int>(std::max_element(info.normal.begin(), info.normal.end(),
        [](double left, double right) { return std::abs(left) < std::abs(right); }) - info.normal.begin());
    info.normal_sign = info.normal[static_cast<std::size_t>(info.normal_axis)] < 0 ? -1 : 1;
    const auto h = options.thickness_fraction * length;
    ensure(std::isfinite(h) && h > 0 && h <= std::numeric_limits<GeometryReal>::max(),
           "Extrusion thickness exceeds GeometryReal");
    info.thickness = static_cast<GeometryReal>(h);
    ensure(info.thickness > 0, "Extrusion thickness underflows GeometryReal");
    std::vector<bool> reverse(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i)
        reverse[i] = reverse_cell(cache, mesh.cells.nodes(i), info.normal, length);
    Coordinates coordinates;
    const auto count = size(output_nodes.local_size(comm.rank()));
    coordinates.x.resize(count);
    coordinates.y.resize(count);
    coordinates.z.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto id = output_nodes.begin(comm.rank()) + static_cast<GlobalIndex>(i);
        auto point = cache.get(id < nodes ? id : id - nodes);
        if (id >= nodes) {
            const auto original = point;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                point[axis] = static_cast<GeometryReal>(static_cast<double>(point[axis]) + info.thickness * info.normal[axis]);
                ensure(std::isfinite(point[axis]), "Extrusion coordinate overflows GeometryReal");
            }
            ensure(point != original, "Extrusion layer collapses at GeometryReal precision");
        }
        coordinates.x[i] = point[0];
        coordinates.y[i] = point[1];
        coordinates.z[i] = point[2];
    }
    // Release cached coordinates before expanding connectivity and boundary storage.
    cache = {};
    const auto bottom = static_cast<PatchId>(mesh.patches.size()), top = static_cast<PatchId>(bottom + 1);
    const auto old_size = mesh.cells.node_ids.size();
    mesh.cells.node_ids.resize(product(old_size, 2));
    // Expand backwards so source connectivity survives without a second full array.
    for (std::size_t i = mesh.cells.size(); i-- > 0;) {
        const auto start = mesh.cells.offsets[i];
        const std::size_t arity = mesh.cells.types[i] == CellType::TRI ? 3 : 4;
        std::array<GlobalIndex, 4> ids{};
        std::copy_n(mesh.cells.node_ids.begin() + static_cast<std::ptrdiff_t>(start), arity, ids.begin());
        if (reverse[i]) std::reverse(ids.begin() + 1, ids.begin() + static_cast<std::ptrdiff_t>(arity));
        for (std::size_t j = 0; j < arity; ++j) {
            mesh.cells.node_ids[2 * start + j] = ids[j];
            mesh.cells.node_ids[2 * start + arity + j] = ids[j] + nodes;
        }
        mesh.cells.types[i] = arity == 3 ? CellType::PRISM : CellType::HEXA;
        mesh.cells.offsets[i] = 2 * start;
    }
    mesh.cells.offsets.back() = 2 * old_size;
    mesh.patches.push_back({options.bottom_patch, {}, "ExtrusionCap"});
    mesh.patches.push_back({options.top_patch, {}, "ExtrusionCap"});
    mesh.coordinates = std::move(coordinates);
    mesh.distribution.nodes = output_nodes;
    mesh.boundary = build_boundary(mesh, output_boundary, maximum_id + 1, bottom, top, comm, options.batch_elements);
    mesh.distribution.boundary_ordinals = output_boundary;
    mesh.cell_dimension = mesh.physical_dimension = 3;
    return info;
}
} // namespace

/// @brief Collectively constructs a 3D layer without per-stage status collectives.
ExtrusionInfo extrude_one_layer(RawMesh& mesh, mpi::CommView comm, const ExtrusionOptions& options)
{
    ensure(!comm.is_null() && mpi::is_active(), "Extrusion requires active MPI and a live communicator");
    try {
        return extrude(mesh, comm, options);
    } catch (const std::exception& error) {
        // Allocation failures must not leave other ranks waiting in the next exchange.
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unknown exception during mesh extrusion");
    }
}
} // namespace krona::mesh