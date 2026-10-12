#pragma once

#include <krona/linalg/types.hpp>
#include <krona/mpi/communicator.hpp>

#include <span>
#include <algorithm>
#include <vector>

namespace krona::linalg {

/// @brief Square-system owned-row distribution, independent of mesh and physics.
/// Construction is collective; the borrowed MPI communicator must outlive its users.
template<class Exec = execution::ExecutionSpace>
class Distribution {
public:
    using map_type = typename Types<double, Exec>::map_type;

    /// @brief Preserves the supplied local row order; GIDs must be globally unique.
    Distribution(std::span<const GlobalOrdinal> owned_gids, mpi::CommView comm)
    {
        mpi::guarded([&] {
            mpi::require(Tpetra::isInitialized(), "Linalg: initialize Tpetra before constructing distributions");
            mpi::require(!comm.is_null(), "Linalg: null communicator");
            const auto count = detail::local_count(owned_gids.size());
            for (const auto gid : owned_gids)
                mpi::require(gid >= 0 && gid < std::numeric_limits<GlobalOrdinal>::max(), "Linalg: invalid owned GID");
            if (!std::is_sorted(owned_gids.begin(), owned_gids.end())) {
                std::vector<GlobalOrdinal> sorted(owned_gids.begin(), owned_gids.end());
                std::sort(sorted.begin(), sorted.end());
                mpi::require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "Linalg: duplicate local owned GID");
            } else mpi::require(std::adjacent_find(owned_gids.begin(), owned_gids.end()) == owned_gids.end(),
                                   "Linalg: duplicate local owned GID");
            auto mpi_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(comm.native())));
            map_ = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                Teuchos::ArrayView<const GlobalOrdinal>(owned_gids.data(), count), GlobalOrdinal{0}, mpi_comm));
            const bool one_to_one = map_->isOneToOne();
            mpi::require(map_->getGlobalNumElements() == 0 || one_to_one, "Linalg: owned GIDs overlap between ranks");
        });
    }

    [[nodiscard]] std::size_t local_size() const noexcept { return map_->getLocalNumElements(); }
    [[nodiscard]] Tpetra::global_size_t global_size() const noexcept { return map_->getGlobalNumElements(); }
    [[nodiscard]] GlobalOrdinal global_id(LocalOrdinal row) const
    {
        mpi::require(row >= 0 && static_cast<std::size_t>(row) < local_size(), "Linalg: invalid owned row");
        return map_->getGlobalElement(row);
    }
    [[nodiscard]] const Teuchos::RCP<const map_type>& native() const noexcept { return map_; }

private:
    Teuchos::RCP<const map_type> map_;
};

} // namespace krona::linalg
