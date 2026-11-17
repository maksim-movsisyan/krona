#pragma once

#include <krona/linalg/csr_pattern.hpp>

namespace krona::linalg {

/// @brief Interprets a CSR graph as a graph of fixed-size dense blocks.
/// @tparam Exec Kokkos execution space used by both the block and scalar maps.
/// @details A graph entry stands for one dense block. The expanded map gives every component
/// its own scalar ID for vectors and solvers. Copies share the graph, entry map, and maps.
template<class Exec = execution::ExecutionSpace>
class BsrPattern {
public:
    using execution_space = Exec;
    using memory_space = typename CsrPattern<Exec>::memory_space;
    using graph_type = typename CsrPattern<Exec>::graph_type;

    /// @brief Builds a block pattern and collectively expands its owned distribution.
    /// @param block_graph One row and column ID per block row or column.
    /// @param block_size Number of rows and columns in each dense block, equal on every rank.
    BsrPattern(const CsrPattern<Exec>& block_graph, LocalOrdinal block_size)
        : graph_(block_graph), distribution_(block_graph.distribution().expanded(checked_size(block_size))), block_size_(block_size) {
        const std::size_t per_block = static_cast<std::size_t>(block_size_)*static_cast<std::size_t>(block_size_);
        mpi::require(graph_.local_nonzeros() <= std::numeric_limits<std::size_t>::max()/per_block,
                     "Linalg: BSR coefficient count overflows");
        mpi::require(graph_.native()->getColMap()->getLocalNumElements() <=
                     static_cast<std::size_t>(std::numeric_limits<LocalOrdinal>::max())/static_cast<std::size_t>(block_size_),
                     "Linalg: expanded column count exceeds Tpetra int ordinals");
    }

    [[nodiscard]] const CsrPattern<Exec>& block_graph() const noexcept { return graph_; }
    /// @brief Returns the owned map with one local position per block row.
    [[nodiscard]] const Distribution<Exec>& block_distribution() const noexcept { return graph_.distribution(); }
    /// @brief Returns the owned map used by vectors, with all components of each block row together.
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return distribution_; }
    [[nodiscard]] LocalOrdinal block_size() const noexcept { return block_size_; }
    /// @brief Counts stored dense blocks, including any whose coefficients are zero.
    [[nodiscard]] std::size_t local_nonzeros() const noexcept { return graph_.local_nonzeros(); }
    [[nodiscard]] std::size_t local_values() const noexcept
    { return local_nonzeros()*static_cast<std::size_t>(block_size_)*static_cast<std::size_t>(block_size_); }
    [[nodiscard]] const Teuchos::RCP<const graph_type>& native() const noexcept { return graph_.native(); }
    [[nodiscard]] bool has_input_mapping() const noexcept { return graph_.has_input_mapping(); }
    [[nodiscard]] auto input_to_entry() const { return graph_.input_to_entry(); }
    [[nodiscard]] std::span<const std::size_t> host_input_to_entry() const { return graph_.host_input_to_entry(); }

private:
    static LocalOrdinal checked_size(LocalOrdinal size) {
        mpi::require(size > 0, "Linalg: nonpositive component count");
        mpi::require(size <= std::numeric_limits<LocalOrdinal>::max()/size,
                     "Linalg: squared block size exceeds Tpetra int ordinals");
        return size;
    }
    CsrPattern<Exec> graph_;
    Distribution<Exec> distribution_;
    LocalOrdinal block_size_;
};

} // namespace krona::linalg
