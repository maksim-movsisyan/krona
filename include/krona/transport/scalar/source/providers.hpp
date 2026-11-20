#pragma once

#include <krona/transport/scalar/detail/views.hpp>

namespace krona::transport::scalar {

/// @brief Supplies the same source per unit volume to every owned cell.
/// @tparam Real Floating-point type used to store the source density.
template<class Real = ComputeReal>
struct UniformSource {
    Real value = 0; ///< Source density. Positive values increase the right-hand side.

    /// @brief Returns the constant for any local cell index.
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};

/// @brief Reads a source per unit volume from an existing cell field.
/// @tparam View Rank-one Kokkos view indexed by local cell ID, with at least one entry per owned cell.
template<class View>
struct CellSource {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values; ///< Source densities accessible from the caller's execution space.

    /// @brief Reads the source density at the given owned cell.
    /// @param cell Local index of the owned cell.
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell) const { return values(cell); }
};

namespace detail {

/// @brief Leaves storage checks for custom callable sources to their callers.
/// @tparam Exec Kokkos execution space that evaluates the source.
/// @tparam Source Custom provider that can be called with an owned cell index.
template<class Exec, class Source>
void validate_source(const Source&, std::size_t) {}

/// @brief Checks that a field-backed source is accessible and covers the owned cells.
/// @tparam Exec Kokkos execution space that reads the source field.
/// @tparam View Rank-one source view indexed by owned cell ID.
/// @param source Provider with the field view to check.
/// @param count Number of owned cells on this rank.
template<class Exec, class View>
void validate_source(CellSource<View> source, std::size_t count) { validate_view<Exec>(source.values, count); }

} // namespace detail
} // namespace krona::transport::scalar
