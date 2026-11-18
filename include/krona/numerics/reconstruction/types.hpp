#pragma once

#include <cstddef>
#include <limits>

namespace krona::numerics::reconstruction {

/// @brief Selects the owned cells in the half-open interval [begin, end).
/// The default end means the number of owned rows advertised by the stencil.
struct CellRange {
    std::size_t begin = 0;
    std::size_t end = std::numeric_limits<std::size_t>::max();
};

/// @brief Computes a separate factor for each variable.
struct Componentwise {};

/// @brief Reuses the smallest factor in the cell for all its variables.
struct SharedMinimum {};

/// @brief Values used to limit an increment at one point.
/// @tparam Real Arithmetic type used to compare the increment with the bounds.
template <class Real>
struct LimiterContext {
    std::size_t cell = 0;
    std::size_t component = 0;
    Real value = 0;
    Real minimum = 0;
    Real maximum = 0;
    Real increment = 0;
};

} // namespace krona::numerics::reconstruction