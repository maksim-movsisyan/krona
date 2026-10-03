#pragma once

#include <cassert>
#include <concepts>
#include <stdexcept>
#include <utility>

namespace krona::distribution {

/// @brief Balanced contiguous ownership of [0, global_size), including empty ranks.
template<std::signed_integral Index>
class BlockDistribution final {
public:
    /// @brief Creates an empty single-part distribution.
    constexpr BlockDistribution() noexcept = default;

    /// @brief Divides a nonnegative global size into a positive number of parts.
    constexpr BlockDistribution(Index global_size, int parts) : size_(global_size), parts_(parts)
    {
        // Validate construction once; rank access below is a caller contract.
        if (global_size < 0 || parts <= 0 || !std::in_range<Index>(parts))
            throw std::invalid_argument("Invalid block distribution");
    }

    /// @brief Returns the number of globally owned entities.
    [[nodiscard]] constexpr Index global_size() const noexcept { return size_; }

    /// @brief Returns the number of ownership blocks.
    [[nodiscard]] constexpr int parts() const noexcept { return parts_; }

    /// @brief Returns the first global ID owned by a valid rank.
    [[nodiscard]] constexpr Index begin(int rank) const noexcept
    {
        assert(rank >= 0 && rank < parts_);
        const auto p = static_cast<Index>(parts_), r = static_cast<Index>(rank);
        const auto remainder = size_ % p;
        // The first remainder ranks own one additional entity.
        return (size_ / p) * r + (r < remainder ? r : remainder);
    }

    /// @brief Returns the exclusive end of a valid rank's block.
    [[nodiscard]] constexpr Index end(int rank) const noexcept
    {
        return begin(rank) + local_size(rank);
    }

    /// @brief Returns the number of entities owned by a valid rank.
    [[nodiscard]] constexpr Index local_size(int rank) const noexcept
    {
        assert(rank >= 0 && rank < parts_);
        const auto p = static_cast<Index>(parts_);
        return size_ / p + (static_cast<Index>(rank) < size_ % p ? 1 : 0);
    }

    /// @brief Returns an ID's owner in constant time, or -1 for an invalid ID.
    [[nodiscard]] constexpr int owner(Index id) const noexcept
    {
        if (id < 0 || id >= size_) return -1;
        const auto p = static_cast<Index>(parts_);
        const Index q = size_ / p, remainder = size_ % p;
        // Avoid q + 1 overflowing when size == max(Index) and parts == 1.
        if (remainder == 0) return static_cast<int>(id / q);
        const Index large_end = (q + 1) * remainder;
        if (id < large_end) return static_cast<int>(id / (q + 1));
        // For q == 0 every valid ID was handled by the larger-block branch.
        return static_cast<int>(remainder + (id - large_end) / q);
    }

private:
    Index size_ = 0;
    int parts_ = 1;
};

} // namespace krona::distribution