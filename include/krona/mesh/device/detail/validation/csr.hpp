#pragma once

#include <krona/mesh/device/mesh_view.hpp>

#include <cstddef>

namespace krona::mesh::device::validation {

/// @brief Describes how IDs are ordered within a row.
/// Nondecreasing allows repeats. Strict requires each ID to be greater than the previous one.
enum class RowOrder { Unordered, Nondecreasing, Strict };

/// @brief Checks the offsets and local IDs of a compressed adjacency table (CSR).
/// @tparam Order Whether IDs can appear in any order, must be sorted, or must also be unique.
/// @tparam RejectSelf Reject an ID equal to the row index, for example a cell listed as its own neighbour.
/// @tparam UniqueUnordered Check for duplicates by scanning earlier entries, even in an unsorted row.
/// @tparam Offsets Kokkos view containing the start of every row and one final offset.
/// @tparam Values Kokkos view containing the local IDs for all rows.
/// Run over rows + 1 after checking the view shapes. For example, offsets {0, 2, 3}
/// describe rows [0, 2) and [2, 3) in the values array.
template <RowOrder Order, bool RejectSelf, bool UniqueUnordered, class Offsets, class Values>
struct ValidateCsr {
    Offsets offsets;
    Values values;
    std::size_t rows, upper_bound;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t row, std::size_t& bad) const {
        // The extra iteration checks the final offset, including when there are no rows.
        if (row == rows) {
            if (offsets(row) != values.extent(0) || (rows == 0 && offsets(row) != 0))
                ++bad;
            return;
        }

        // Check this row's bounds before reading any IDs.
        const auto begin = offsets(row), end = offsets(row + 1);
        if ((row == 0 && begin != 0) || begin > end || end > values.extent(0)) {
            ++bad;
            return;
        }

        bool valid = true;
        for (std::size_t j = begin; j < end; ++j) {
            const auto id = values(j);
            valid &= id >= 0 && static_cast<std::size_t>(id) < upper_bound;
            if constexpr (RejectSelf)
                valid &= id < 0 || static_cast<std::size_t>(id) != row;

            // Sorted rows can check ordering and detect repeats using adjacent entries.
            if constexpr (Order != RowOrder::Unordered) {
                if (j != begin) {
                    if constexpr (Order == RowOrder::Strict)
                        valid &= values(j - 1) < id;
                    else
                        valid &= values(j - 1) <= id;
                }
            }
            if constexpr (UniqueUnordered)
                for (std::size_t k = begin; k < j; ++k)
                    valid &= values(k) != id;
        }

        if (!valid)
            ++bad;
    }
};

} // namespace krona::mesh::device::validation
