#pragma once

#include <krona/linalg/distribution.hpp>

#include <algorithm>
#include <span>
#include <string>
#include <vector>

namespace krona::linalg {

/// @brief Host CSR input in owned-row order, with global column IDs.
struct CsrInput {
    std::vector<std::size_t> row_offsets;
    std::vector<GlobalOrdinal> column_gids;
};

/// @brief Immutable distributed sparsity and input-to-native coefficient mapping.
/// Construction is collective; copies share the graph and device permutation.
template<class Exec = execution::ExecutionSpace>
class CsrPattern {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using graph_type = typename IndexTypes<Exec>::graph_type;
    using permutation_type = Kokkos::View<const std::size_t*, typename Exec::memory_space>;

    CsrPattern(const Distribution<Exec>& distribution, std::span<const std::size_t> row_offsets,
               std::span<const GlobalOrdinal> column_gids, const Exec& exec = {}, bool keep_input_mapping = true)
        : distribution_(distribution), has_input_mapping_(keep_input_mapping)
    {
        mpi::guarded([&] {
            const auto rows = distribution_.local_size();
            mpi::require(row_offsets.size() == rows+1 && row_offsets.front() == 0 &&
                         row_offsets.back() == column_gids.size(), "Linalg: invalid CSR offsets/extents");
            std::vector<std::size_t> counts(rows);
            std::vector<GlobalOrdinal> sorted;
            for (std::size_t row = 0; row < rows; ++row) {
                mpi::require(row_offsets[row] <= row_offsets[row+1] && row_offsets[row+1] <= column_gids.size(),
                             "Linalg: nonmonotone/out-of-range CSR offsets");
                counts[row] = row_offsets[row+1]-row_offsets[row];
                static_cast<void>(detail::local_count(counts[row]));
                sorted.assign(column_gids.begin()+static_cast<std::ptrdiff_t>(row_offsets[row]),
                              column_gids.begin()+static_cast<std::ptrdiff_t>(row_offsets[row+1]));
                std::sort(sorted.begin(), sorted.end());
                mpi::require(sorted.empty() || (sorted.front() >= 0 && sorted.back() < std::numeric_limits<GlobalOrdinal>::max()),
                             "Linalg: invalid column GID");
                mpi::require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
                             "Linalg: duplicate column in a CSR row");
            }

            {
                std::vector<GlobalOrdinal> remote;
                for (const auto gid : column_gids)
                    if (distribution_.native()->getLocalElement(gid) == Teuchos::OrdinalTraits<LocalOrdinal>::invalid())
                        remote.push_back(gid);
                std::sort(remote.begin(), remote.end());
                remote.erase(std::unique(remote.begin(), remote.end()), remote.end());
                const auto count = detail::local_count(remote.size());
                std::vector<int> owners(remote.size());
                const auto status = distribution_.native()->getRemoteIndexList(
                    Teuchos::ArrayView<const GlobalOrdinal>(remote.data(), count),
                    Teuchos::ArrayView<int>(owners.data(), count));
                mpi::require(status == Tpetra::AllIDsPresent, "Linalg: column has no owner in the domain map");
            }

            auto graph = Teuchos::rcp(new graph_type(distribution_.native(),
                Teuchos::ArrayView<const std::size_t>(counts.data(), detail::local_count(rows))));
            for (std::size_t row = 0; row < rows; ++row) {
                if (counts[row] == 0) continue;
                graph->insertGlobalIndices(distribution_.global_id(static_cast<LocalOrdinal>(row)),
                    Teuchos::ArrayView<const GlobalOrdinal>(column_gids.data()+row_offsets[row],
                                                            detail::local_count(counts[row])));
            }
            graph->fillComplete(distribution_.native(), distribution_.native());
            static_cast<void>(detail::local_count(graph->getColMap()->getLocalNumElements()));
            graph_ = graph;
            nonzeros_ = graph_->getLocalNumEntries();
            if (!has_input_mapping_) {
                Kokkos::fence("Linalg: native CSR preparation complete");
                return;
            }

            const auto native_graph = graph_->getLocalGraphHost();
            std::vector<std::size_t> permutation(column_gids.size());
            for (std::size_t row = 0; row < rows; ++row) {
                const auto begin = native_graph.row_map(row), end = native_graph.row_map(row+1);
                for (std::size_t input = row_offsets[row]; input < row_offsets[row+1]; ++input) {
                    const auto local = graph_->getColMap()->getLocalElement(column_gids[input]);
                    mpi::require(local != Teuchos::OrdinalTraits<LocalOrdinal>::invalid(),
                                 "Linalg: column is absent from the finalized column map");
                    auto pos = begin, upper = end;
                    while (pos < upper) {
                        const auto middle = pos+(upper-pos)/2;
                        if (native_graph.entries(middle) < local) pos = middle+1;
                        else upper = middle;
                    }
                    mpi::require(pos < end && native_graph.entries(pos) == local,
                                 "Linalg: CSR entry disappeared during graph finalization");
                    permutation[input] = pos;
                }
            }
            Kokkos::View<std::size_t*, typename Exec::memory_space> device(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string("linalg/input_to_entry")), permutation.size());
            auto host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, device);
            for (std::size_t i = 0; i < permutation.size(); ++i) host(i) = permutation[i];
            Kokkos::deep_copy(exec, device, host);
            input_to_entry_ = device;
            host_input_to_entry_ = host;
            Kokkos::fence("Linalg: CSR preparation complete");
        });
    }

    CsrPattern(const Distribution<Exec>& distribution, const CsrInput& input, const Exec& exec = {}, bool keep_input_mapping = true)
        : CsrPattern(distribution, input.row_offsets, input.column_gids, exec, keep_input_mapping) {}

    [[nodiscard]] std::size_t local_nonzeros() const noexcept { return nonzeros_; }
    [[nodiscard]] bool has_input_mapping() const noexcept { return has_input_mapping_; }
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return distribution_; }
    [[nodiscard]] const Teuchos::RCP<const graph_type>& native() const noexcept { return graph_; }
    [[nodiscard]] permutation_type input_to_entry() const
    {
        mpi::require(has_input_mapping_, "Linalg: original CSR input mapping was disabled");
        return input_to_entry_;
    }
    [[nodiscard]] std::span<const std::size_t> host_input_to_entry() const
    {
        mpi::require(has_input_mapping_, "Linalg: original CSR input mapping was disabled");
        return {host_input_to_entry_.data(), host_input_to_entry_.extent(0)};
    }

private:
    Distribution<Exec> distribution_;
    Teuchos::RCP<const graph_type> graph_;
    permutation_type input_to_entry_;
    Kokkos::View<const std::size_t*, Kokkos::HostSpace> host_input_to_entry_;
    std::size_t nonzeros_ = 0;
    bool has_input_mapping_ = true;
};

} // namespace krona::linalg
