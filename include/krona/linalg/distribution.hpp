#pragma once

#include <krona/linalg/types.hpp>
#include <krona/mpi/communicator.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <span>
#include <algorithm>
#include <vector>
#include <utility>

namespace krona::linalg {

/// @brief Square-system owned-row distribution, independent of mesh and physics.
/// Construction is collective; the borrowed MPI communicator must outlive its users.
template<class Exec = execution::ExecutionSpace>
class Distribution {
public:
    using execution_space = Exec;
    using map_type = typename IndexTypes<Exec>::map_type;
    using memory_space = typename IndexTypes<Exec>::memory_space;

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

    /// @brief Collective cell-major expansion: point_gid = block_gid*components + component.
    /// Preserves local block order; one component shares the original map.
    [[nodiscard]] Distribution expanded(LocalOrdinal components) const
    {
        return mpi::guarded([&] {
            mpi::require(components > 0, "Linalg: nonpositive component count");
            LocalOrdinal minimum = 0, maximum = 0;
            Teuchos::reduceAll(*map_->getComm(), Teuchos::REDUCE_MIN, 1, &components, &minimum);
            Teuchos::reduceAll(*map_->getComm(), Teuchos::REDUCE_MAX, 1, &components, &maximum);
            mpi::require(minimum == maximum, "Linalg: component count differs between ranks");
            if (components == 1) return *this;
            const std::size_t width = static_cast<std::size_t>(components);
            mpi::require(local_size() <= static_cast<std::size_t>(std::numeric_limits<LocalOrdinal>::max())/width,
                         "Linalg: expanded local count exceeds Tpetra int ordinals");
            mpi::require(global_size() <= (std::numeric_limits<Tpetra::global_size_t>::max()-1)/width,
                         "Linalg: expanded global count overflows");
            std::vector<GlobalOrdinal> gids(local_size()*width);
            for (std::size_t row = 0; row < local_size(); ++row) {
                const auto gid = map_->getGlobalElement(static_cast<LocalOrdinal>(row));
                mpi::require(gid <= (std::numeric_limits<GlobalOrdinal>::max()-components)/components,
                             "Linalg: expanded GID overflows");
                for (LocalOrdinal c = 0; c < components; ++c)
                    gids[row*width+static_cast<std::size_t>(c)] = gid*components+c;
            }
            auto map = Teuchos::rcp(new map_type(global_size()*width,
                Teuchos::ArrayView<const GlobalOrdinal>(gids.data(), detail::local_count(gids.size())),
                GlobalOrdinal{0}, map_->getComm()));
            return Distribution(std::move(map));
        });
    }

private:
    explicit Distribution(Teuchos::RCP<const map_type> map) : map_(std::move(map)) {}
    Teuchos::RCP<const map_type> map_;
};

} // namespace krona::linalg