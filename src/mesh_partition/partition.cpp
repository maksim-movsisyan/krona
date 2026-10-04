#include <krona/mesh_partition/partition.hpp>

#include <krona/mpi/collectives.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <dkaminpar.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace krona::mesh_partition {
namespace {

/// @brief Builds a bijective placement, optionally grouping shared-memory ranks.
void make_rank_mapping(mpi::CommView comm, bool group, PartitionResult& result)
{
    const auto peers = static_cast<std::size_t>(comm.size());
    result.part2rank.resize(peers);
    result.rank2part.resize(peers);
    std::iota(result.part2rank.begin(), result.part2rank.end(), 0);

    if (group && comm.size() > 1) {
        // The smallest parent rank identifies each shared-memory group.
        std::vector<int> leaders(peers);
        MPI_Comm handle = MPI_COMM_NULL;
        mpi::check(MPI_Comm_split_type(comm.native(), MPI_COMM_TYPE_SHARED,
                                      comm.rank(), MPI_INFO_NULL, &handle),
                   "MPI_Comm_split_type(shared memory)");
        auto shared = mpi::Comm::adopt(handle);
        int leader = comm.rank();
        mpi::broadcast(shared.view(), leader, 0);
        shared.close();
        mpi::allgather(comm, std::span<const int>{&leader, 1}, std::span<int>{leaders});

        // Local MPI order is not a physical core ID; no CPU affinity is changed.
        std::sort(result.part2rank.begin(), result.part2rank.end(), [&](int a, int b) {
            const auto left = leaders[static_cast<std::size_t>(a)];
            const auto right = leaders[static_cast<std::size_t>(b)];
            return left != right ? left < right : a < b;
        });
        int previous = -1;
        for (const int rank : result.part2rank) {
            const auto current = leaders[static_cast<std::size_t>(rank)];
            if (current != previous) ++result.shared_memory_groups;
            previous = current;
        }
    } else if (group) {
        result.shared_memory_groups = 1;
    }

    // A single block belongs to each destination rank.
    for (std::size_t block = 0; block < peers; ++block)
        result.rank2part[static_cast<std::size_t>(result.part2rank[block])] =
            static_cast<PartitionBlockId>(block);
}

/// @brief Borrows matching index types or converts them into temporary import storage.
template<std::integral Target, std::integral Source>
[[nodiscard]] std::span<const Target> index_view(
    std::span<const Source> source, std::vector<Target>& storage)
{
    if constexpr (std::same_as<Target, Source>) {
        // Matching types need no allocation, cast or aliasing workaround.
        return source;
    } else {
        storage.reserve(source.size());
        for (const auto value : source) storage.push_back(static_cast<Target>(value));
        return storage;
    }
}

/// @brief Imports distributed CSR and releases conversion buffers before solving.
void import_graph(kaminpar::dKaMinPar& partitioner, const mesh::DualGraph& graph,
                  std::span<const GlobalIndex> distribution)
{
    std::vector<kaminpar::dist::GlobalNodeID> distribution_storage, adjacency_storage;
    std::vector<kaminpar::dist::GlobalEdgeID> offset_storage;
    const auto vtxdist = index_view(distribution, distribution_storage);
    const auto xadj = index_view(std::span<const std::size_t>{graph.offsets}, offset_storage);
    const auto adjncy = index_view(std::span<const GlobalIndex>{graph.adjacency}, adjacency_storage);

    // dKaMinPar owns its imported graph after this call; input remains untouched.
    partitioner.copy_graph(vtxdist, xadj, adjncy);
}

/// @brief Checks ownership and CSR dimensions once; deep graph validity is a contract.
[[nodiscard]] std::size_t check_input(
    const mesh::DualGraph& graph, std::span<const GlobalIndex> distribution,
    mpi::CommView comm, const DKaMinParOptions& options)
{
    const auto peers = static_cast<std::size_t>(comm.size());
    if (distribution.size() != peers + 1 || distribution.front() != 0)
        mpi::fatal("Partitioning requires P+1 contiguous ownership endpoints starting at zero");
    for (std::size_t rank = 0; rank < peers; ++rank)
        if (distribution[rank + 1] < distribution[rank])
            mpi::fatal("Cell ownership endpoints must be nondecreasing");

    const auto rank = static_cast<std::size_t>(comm.rank());
    const auto local = distribution[rank + 1] - distribution[rank];
    if (!std::in_range<std::size_t>(local)) mpi::fatal("Local graph size exceeds size_t");
    const auto cells = static_cast<std::size_t>(local);
    // Protect narrowing inside the library's configured local ID and weight types.
    if (cells >= std::numeric_limits<std::size_t>::max() ||
        cells >= std::numeric_limits<kaminpar::dist::NodeID>::max() ||
        graph.adjacency.size() >= std::numeric_limits<kaminpar::dist::EdgeID>::max() ||
        !std::in_range<kaminpar::dist::NodeWeight>(cells) ||
        !std::in_range<kaminpar::dist::EdgeWeight>(graph.adjacency.size()))
        mpi::fatal("Local graph exceeds dKaMinPar ID/weight widths; rebuild with wider types");
    if (graph.offsets.size() != cells + 1 || graph.offsets.front() != 0 ||
        graph.offsets.back() != graph.adjacency.size())
        mpi::fatal("Dual graph CSR dimensions do not match cell ownership");
    if (!(options.imbalance_tolerance > 0) || !std::isfinite(options.imbalance_tolerance) ||
        options.threads_per_rank < 1)
        mpi::fatal("Invalid dKaMinPar imbalance tolerance or worker-thread count");
    if (options.threads_per_rank > 1 && mpi::thread_level() < MPI_THREAD_FUNNELED)
        mpi::fatal("Multithreaded partitioning requires MPI_THREAD_FUNNELED or stronger");

#ifndef NDEBUG
    // No full connectivity scan remains in a release build.
    for (std::size_t row = 0; row < cells; ++row)
        assert(graph.offsets[row] <= graph.offsets[row + 1]);
    for (const auto neighbour : graph.adjacency)
        assert(neighbour >= 0 && neighbour < distribution.back());
#endif
    return cells;
}

/// @brief Computes destinations and planned loads without gathering the global graph.
[[nodiscard]] PartitionResult partition(
    const mesh::DualGraph& graph, std::span<const GlobalIndex> distribution,
    mpi::CommView comm, const DKaMinParOptions& options)
{
    const auto cells = check_input(graph, distribution, comm, options);
    PartitionResult result;
    make_rank_mapping(comm, options.group_by_shared_memory, result);
    result.cell_target_rank.resize(cells, 0);
    result.cells_per_rank.resize(static_cast<std::size_t>(comm.size()), 0);

    // A single destination or an empty global graph needs no solver call.
    if (comm.size() == 1 || distribution.back() == 0) {
        result.cells_per_rank[0] = static_cast<std::uint64_t>(cells);
        return result;
    }

    {
        kaminpar::dKaMinPar::reseed(options.seed);
        kaminpar::dKaMinPar partitioner(
            comm.native(), options.threads_per_rank, kaminpar::dist::create_default_context());
        partitioner.set_output_level(options.quiet ? kaminpar::OutputLevel::QUIET
                                                  : kaminpar::OutputLevel::PROGRESS);
        import_graph(partitioner, graph, distribution);
        std::vector<kaminpar::dist::BlockID> blocks(cells);

        // This overload is available in the supplied older API and in dKaMinPar 3.7.
        const auto cut = partitioner.compute_partition(
            static_cast<kaminpar::dist::BlockID>(comm.size()),
            options.imbalance_tolerance, std::span{blocks});
        if (cut < 0) mpi::fatal("dKaMinPar returned a negative edge cut");
        result.global_edge_cut = static_cast<std::uint64_t>(cut);
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const auto block = static_cast<std::size_t>(blocks[cell]);
            if (block >= result.part2rank.size()) mpi::fatal("dKaMinPar returned an invalid block ID");
            const int target = result.part2rank[block];
            result.cell_target_rank[cell] = target;
            ++result.cells_per_rank[static_cast<std::size_t>(target)];
        }
        // Destroy solver graphs and raw block storage before reducing diagnostics.
    }
    mpi::allreduce_in_place<mpi::Reduction::sum>(comm, std::span{result.cells_per_rank});
    return result;
}

} // namespace

/// @brief Partitions arbitrary contiguous rank ranges collectively on the MPI main thread.
PartitionResult partition_cells(
    const mesh::DualGraph& graph, std::span<const GlobalIndex> cell_distribution,
    mpi::CommView comm, const DKaMinParOptions& options)
{
    if (comm.is_null() || !mpi::is_active()) mpi::fatal("Partitioning requires active MPI");
    mpi::require_main_thread();
    try {
        // Fatal errors abort WORLD immediately instead of leaving peers in the solver.
        return partition(graph, cell_distribution, comm, options);
    } catch (const std::exception& error) {
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unknown exception during distributed partitioning");
    }
}

/// @brief Converts balanced ownership to small replicated endpoints and partitions the graph.
PartitionResult partition_cells(
    const mesh::DualGraph& graph,
    const distribution::BlockDistribution<GlobalIndex>& ownership,
    mpi::CommView comm, const DKaMinParOptions& options)
{
    if (ownership.parts() != comm.size()) mpi::fatal("Cell ownership does not match the communicator");
    try {
        std::vector<GlobalIndex> endpoints(static_cast<std::size_t>(comm.size()) + 1);
        for (int rank = 0; rank < comm.size(); ++rank)
            endpoints[static_cast<std::size_t>(rank)] = ownership.begin(rank);
        endpoints.back() = ownership.global_size();
        // Only O(P) ownership metadata is synthesized; graph arrays are borrowed.
        return partition_cells(graph, std::span<const GlobalIndex>{endpoints}, comm, options);
    } catch (const std::exception& error) {
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unknown exception while preparing cell ownership");
    }
}

} // namespace krona::mesh_partition
