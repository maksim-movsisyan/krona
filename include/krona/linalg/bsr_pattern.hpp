#pragma once

#include <krona/linalg/csr_pattern.hpp>

namespace krona::linalg {

/// @brief Immutable block graph, fixed block size and expanded scalar-DOF distribution.
/// Copies share graph, input permutation and maps. Reuse one pattern across block matrices.
template<class Exec = execution::ExecutionSpace>
class BsrPattern {
public:
    using execution_space = Exec;
    using memory_space = typename CsrPattern<Exec>::memory_space;
    using graph_type = typename CsrPattern<Exec>::graph_type;

    BsrPattern(const CsrPattern<Exec>& block_graph, LocalOrdinal block_size)
        : graph_(block_graph), distribution_(block_graph.distribution().expanded(checked_size(block_size))), block_size_(block_size)
    {
        const std::size_t per_block = static_cast<std::size_t>(block_size_)*static_cast<std::size_t>(block_size_);
        mpi::require(graph_.local_nonzeros() <= std::numeric_limits<std::size_t>::max()/per_block,
                     "Linalg: BSR coefficient count overflows");
        mpi::require(graph_.native()->getColMap()->getLocalNumElements() <=
                     static_cast<std::size_t>(std::numeric_limits<LocalOrdinal>::max())/static_cast<std::size_t>(block_size_),
                     "Linalg: expanded column count exceeds Tpetra int ordinals");
    }

    [[nodiscard]] const CsrPattern<Exec>& block_graph() const noexcept { return graph_; }
    [[nodiscard]] const Distribution<Exec>& block_distribution() const noexcept { return graph_.distribution(); }
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return distribution_; }
    [[nodiscard]] LocalOrdinal block_size() const noexcept { return block_size_; }
    [[nodiscard]] std::size_t local_nonzeros() const noexcept { return graph_.local_nonzeros(); }
    [[nodiscard]] std::size_t local_values() const noexcept
    { return local_nonzeros()*static_cast<std::size_t>(block_size_)*static_cast<std::size_t>(block_size_); }
    [[nodiscard]] const Teuchos::RCP<const graph_type>& native() const noexcept { return graph_.native(); }
    [[nodiscard]] bool has_input_mapping() const noexcept { return graph_.has_input_mapping(); }
    [[nodiscard]] auto input_to_entry() const { return graph_.input_to_entry(); }
    [[nodiscard]] std::span<const std::size_t> host_input_to_entry() const { return graph_.host_input_to_entry(); }

private:
    static LocalOrdinal checked_size(LocalOrdinal size)
    {
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
