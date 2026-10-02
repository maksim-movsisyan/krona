#pragma once

#include <concepts>
#include <stdexcept>
#include <utility>

namespace krona::distribution
{

// Balanced contiguous ownership of [0, global_size), including empty ranks.
template <std::signed_integral Index> 
class BlockDistribution final
{
  public:
    constexpr BlockDistribution() noexcept = default;
    constexpr BlockDistribution(Index global_size, int parts) : size_(global_size), parts_(parts)
    {
        if (global_size < 0 || parts <= 0 || !std::in_range<Index>(parts))
        {
            throw std::invalid_argument("Invalid block distribution");
        }
    }

    [[nodiscard]] constexpr Index global_size() const noexcept
    {
        return size_;
    }
    [[nodiscard]] constexpr int parts() const noexcept
    {
        return parts_;
    }
    [[nodiscard]] constexpr Index begin(int rank) const
    {
        check_rank(rank);
        const auto p = static_cast<Index>(parts_);
        const auto r = static_cast<Index>(rank);
        const Index remainder = size_ % p;
        return (size_ / p) * r + (r < remainder ? r : remainder);
    }
    [[nodiscard]] constexpr Index end(int rank) const
    {
        check_rank(rank);
        return rank == parts_ - 1 ? size_ : begin(rank + 1);
    }
    [[nodiscard]] constexpr Index local_size(int rank) const
    {
        return end(rank) - begin(rank);
    }

    // -1 denotes an ID outside this distribution.
    [[nodiscard]] constexpr int owner(Index id) const noexcept
    {
        if (id < 0 || id >= size_)
        {
            return -1;
        }
        const auto p = static_cast<Index>(parts_);
        const Index q = size_ / p;
        const Index remainder = size_ % p;
        // Avoid q + 1 overflowing when size == max(Index) and parts == 1.
        if (remainder == 0)
        {
            return static_cast<int>(id / q);
        }
        const Index large_end = (q + 1) * remainder;
        if (id < large_end)
        {
            return static_cast<int>(id / (q + 1));
        }
        return static_cast<int>(remainder + (id - large_end) / q);
    }

  private:
    constexpr void check_rank(int rank) const
    {
        if (rank < 0 || rank >= parts_)
        {
            throw std::out_of_range("Invalid block rank");
        }
    }
    Index size_ = 0;
    int parts_ = 1;
};

} // namespace krona::distribution
