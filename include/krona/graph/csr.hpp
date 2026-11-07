#pragma once

#include <krona/config.hpp>
#include <cstddef>
#include <span>
#include <vector>

namespace krona::graph {

/// @brief Borrows distributed CSR rows; adjacency entries are global vertex IDs.
/// A valid graph has n_local+1 nondecreasing offsets, starting at zero and
/// ending at adjacency.size(). An empty rank supplies one zero offset.
/// Per-rank vertex ownership is supplied separately to distributed algorithms.
struct CsrGraphView {
    std::span<const std::size_t> offsets;
    std::span<const GlobalIndex> adjacency;
};

/// @brief Owns local CSR rows without mesh, MPI or partitioner dependencies.
struct CsrGraph {
    std::vector<std::size_t> offsets;
    std::vector<GlobalIndex> adjacency;

    /// @brief Borrows arrays without copying; the owner must outlive the view.
    [[nodiscard]] CsrGraphView view() const & noexcept { return {offsets, adjacency}; }
    CsrGraphView view() const && = delete;
};

} // namespace krona::graph
