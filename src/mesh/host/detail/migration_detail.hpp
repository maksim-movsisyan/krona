#pragma once

#include <krona/mesh/host/migrate.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>

namespace krona::mesh::host::migration_detail {

/// @brief Releases a temporary vector's capacity at its last use.
template<class T>
void release(std::vector<T>& values)
{
    // clear() would retain the backing allocation.
    std::vector<T>{}.swap(values);
}

/// @brief Narrows an entity count once before local indices are generated.
[[nodiscard]] inline LocalIndex local_count(std::size_t count)
{
    if (!std::in_range<LocalIndex>(count)) mpi::fatal("Migrated mesh exceeds LocalIndex");
    return static_cast<LocalIndex>(count);
}

/// @brief Packed record counts and displacements for one MPI exchange.
struct Layout {
    std::vector<int> counts, offsets;
    std::size_t size = 0;
};

/// @brief Creates a packed MPI 3.1 layout, checking overflow before narrowing.
[[nodiscard]] inline Layout layout(std::span<const std::size_t> counts)
{
    Layout result;
    result.counts.resize(counts.size());
    result.offsets.resize(counts.size());
    for (std::size_t peer = 0; peer < counts.size(); ++peer) {
        // Keep counts and the whole packed displacement range representable by int.
        if (counts[peer] > static_cast<std::size_t>(std::numeric_limits<int>::max()) - result.size)
            mpi::fatal("Migration buffer exceeds MPI 3.1 INT_MAX; use more ranks or batching");
        result.counts[peer] = static_cast<int>(counts[peer]);
        result.offsets[peer] = static_cast<int>(result.size);
        result.size += counts[peer];
    }
    return result;
}

/// @brief Exchanges a consumed packed vector and retains its request layout for replies.
template<class T>
[[nodiscard]] std::vector<T> exchange(
    mpi::CommView comm, const Layout& plan, std::vector<T>&& send,
    mpi::AlltoallvWorkspace& work)
{
    if (comm.size() == 1) {
        // A local transfer moves storage and performs no MPI calls.
        work.send_counts = work.recv_counts = plan.counts;
        work.send_displacements = work.recv_displacements = plan.offsets;
        return std::move(send);
    }
    std::vector<T> recv;
    if constexpr (mpi::RawRecord<T>) {
        mpi::RecordDatatype<T> type;
        mpi::alltoallv_packed(comm, plan.counts, std::span{send}, recv, work, type);
    } else {
        mpi::alltoallv_packed(comm, plan.counts, std::span{send}, recv, work);
    }
    release(send);
    return recv;
}

/// @brief Returns one response per request using the known inverse layout.
template<class T>
[[nodiscard]] std::vector<T> reply(
    mpi::CommView comm, std::vector<T>&& send, const mpi::AlltoallvWorkspace& work)
{
    if (comm.size() == 1) return std::move(send);
    std::size_t size = 0;
    for (const auto count : work.send_counts) size += static_cast<std::size_t>(count);
    std::vector<T> recv(size);
    // Request counts already determine response counts: no second counts exchange.
    if constexpr (mpi::RawRecord<T>) {
        mpi::RecordDatatype<T> type;
        mpi::alltoallv(comm, std::span{send}, work.recv_counts, work.recv_displacements,
                      std::span{recv}, work.send_counts, work.send_displacements, type);
    } else {
        mpi::alltoallv(comm, std::span{send}, work.recv_counts, work.recv_displacements,
                      std::span{recv}, work.send_counts, work.send_displacements);
    }
    release(send);
    return recv;
}

/// @brief Maps nonnegative GIDs using dense storage or a contiguous linear-probing table.
class GidIndex {
public:
    /// @brief Chooses dense storage for compact ranges and reserves a sparse table otherwise.
    GidIndex(GlobalIndex lo, GlobalIndex hi, std::size_t expected) : lo_(lo)
    {
        const auto span = hi >= lo ? static_cast<std::uint64_t>(hi) -
                                    static_cast<std::uint64_t>(lo) + 1 : 0;
        dense_ = span / 4 + (span % 4 != 0) <= std::max<std::size_t>(expected, 1);
        if (dense_) values_.assign(static_cast<std::size_t>(span), invalid_local_index);
        else {
            std::size_t capacity = 8;
            while (capacity - capacity / 4 < expected) capacity *= 2;
            keys_.resize(capacity);
            values_.assign(capacity, invalid_local_index);
        }
    }

    /// @brief Inserts an unseen ID; node markers and local indices are nonnegative.
    bool insert(GlobalIndex gid, LocalIndex value)
    {
        auto slot = find_slot(gid);
        assert(slot < values_.size());
        if (values_[slot] != invalid_local_index) return false;
        if (!dense_ && size_ == values_.size() - values_.size() / 4) {
            grow();
            slot = find_slot(gid);
        }
        if (!dense_) keys_[slot] = gid;
        values_[slot] = value;
        ++size_;
        return true;
    }

    /// @brief Resolves an existing ID, rejecting a broken ownership or connectivity contract.
    [[nodiscard]] LocalIndex at(GlobalIndex gid) const
    {
        const auto slot = find_slot(gid);
        if (slot == values_.size() || values_[slot] == invalid_local_index)
            throw std::runtime_error("Missing GID in migrated mesh lookup");
        return values_[slot];
    }

    /// @brief Replaces a node marker with its final local index.
    void assign(GlobalIndex gid, LocalIndex value)
    {
        const auto slot = find_slot(gid);
        assert(slot < values_.size() && values_[slot] != invalid_local_index);
        values_[slot] = value;
    }

    /// @brief Returns the number of distinct IDs stored in this lookup.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// @brief Reports whether ordered node IDs can be emitted without sorting.
    [[nodiscard]] bool dense() const noexcept { return dense_; }

    /// @brief Emits owned-used marker 0 before ghost-only marker 1, each in GID order.
    void write_nodes(std::span<GlobalIndex> ids, std::size_t owned) const
    {
        assert(dense_ && ids.size() == size_);
        std::size_t first = 0, second = owned;
        for (std::size_t i = 0; i < values_.size(); ++i) {
            if (values_[i] == 0) ids[first++] = lo_ + static_cast<GlobalIndex>(i);
            else if (values_[i] == 1) ids[second++] = lo_ + static_cast<GlobalIndex>(i);
        }
        assert(first == owned && second == ids.size());
    }

private:
    /// @brief Mixes structured mesh IDs before sparse table probing.
    [[nodiscard]] static std::uint64_t hash(GlobalIndex gid) noexcept
    {
        auto value = static_cast<std::uint64_t>(gid);
        value ^= value >> 30; value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27; value *= UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    }

    /// @brief Finds an occupied or empty slot; out-of-range dense IDs return a sentinel.
    [[nodiscard]] std::size_t find_slot(GlobalIndex gid) const noexcept
    {
        if (dense_) {
            if (gid < lo_ || static_cast<std::uint64_t>(gid - lo_) >= values_.size())
                return values_.size();
            return static_cast<std::size_t>(gid - lo_);
        }
        const auto mask = values_.size() - 1;
        auto slot = static_cast<std::size_t>(hash(gid)) & mask;
        while (values_[slot] != invalid_local_index && keys_[slot] != gid)
            slot = (slot + 1) & mask;
        return slot;
    }

    /// @brief Grows sparse storage geometrically without per-entry allocations.
    void grow()
    {
        auto keys = std::move(keys_);
        auto values = std::move(values_);
        if (values.size() > values.max_size() / 2)
            throw std::length_error("Mesh GID lookup capacity overflow");
        keys_.resize(values.size() * 2);
        values_.assign(keys_.size(), invalid_local_index);
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i] == invalid_local_index) continue;
            const auto slot = find_slot(keys[i]);
            keys_[slot] = keys[i];
            values_[slot] = values[i];
        }
    }

    GlobalIndex lo_;
    bool dense_ = false;
    std::size_t size_ = 0;
    std::vector<GlobalIndex> keys_;
    std::vector<LocalIndex> values_;
};

/// @brief Builds a fixed cell lookup from final ordered GIDs.
[[nodiscard]] inline GidIndex cell_index(std::span<const GlobalIndex> ids)
{
    const auto [lo, hi] = std::minmax_element(ids.begin(), ids.end());
    GidIndex result(ids.empty() ? 0 : *lo, ids.empty() ? -1 : *hi, ids.size());
    static_cast<void>(local_count(ids.size()));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        [[maybe_unused]] const bool inserted = result.insert(ids[i], static_cast<LocalIndex>(i));
        assert(inserted);
    }
    return result;
}

} // namespace krona::mesh::host::migration_detail
