#pragma once

#include <krona/distribution/block.hpp>
#include <krona/mesh/faces.hpp>
#include <krona/mpi/communicator.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace krona::mesh_partition {

using PartitionBlockId = std::uint32_t;

/// @brief Configures unweighted dKaMinPar partitioning into one block per MPI rank.
struct DKaMinParOptions {
    double imbalance_tolerance = 0.03;
    int threads_per_rank = 1;
    int seed = 0;
    bool quiet = true;
    bool group_by_shared_memory = true;
};

/// @brief Owns migration destinations, block placement and replicated global statistics.
struct PartitionResult {
    std::vector<int> cell_target_rank; // One destination for each input CSR row.
    std::vector<int> part2rank;
    std::vector<PartitionBlockId> rank2part;
    std::vector<std::uint64_t> cells_per_rank; // Planned ownership after migration.
    std::uint64_t global_edge_cut = 0; // Undirected cut edges, not directed CSR arcs.
    int shared_memory_groups = 0; // Zero when discovery was disabled.
};

/// @brief Collectively partitions CSR rows with explicit contiguous ownership ranges.
/// cell_distribution has P+1 entries; row i has ID cell_distribution[rank]+i.
[[nodiscard]] PartitionResult partition_cells(
    const mesh::DualGraph& graph, std::span<const GlobalIndex> cell_distribution,
    mpi::CommView comm, const DKaMinParOptions& options = {});

/// @brief Partitions a graph using the balanced ownership produced by mesh import.
[[nodiscard]] PartitionResult partition_cells(
    const mesh::DualGraph& graph,
    const distribution::BlockDistribution<GlobalIndex>& ownership,
    mpi::CommView comm, const DKaMinParOptions& options = {});

// Every rank participates with identical options and ownership metadata.
// Input is a valid symmetric CSR graph with zero-based global neighbour IDs.
// Self-loops are absent; each row's neighbours are unique. Empty ranks are allowed.
// cell_distribution begins at zero, is nondecreasing and covers all graph vertices.
// Partitioning changes neither graph arrays nor the current mesh ownership.
// Rank grouping is a placement heuristic, not CPU binding or network-cut optimization.
// No inter-node cut is reported without actually measuring it.
// Calls run on the MPI main thread; do not overlap other dKaMinPar calls in a process.
// Multiple worker threads require at least MPI_THREAD_FUNNELED support.
// Adapted/migrated inputs need contiguous per-rank vertex IDs; remap adjacency accordingly.

} // namespace krona::mesh_partition
