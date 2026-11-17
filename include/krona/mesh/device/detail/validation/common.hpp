#pragma once

#include <krona/mesh/device/mesh_view.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace krona::mesh::device::validation {

/// @brief Checks that a selected mask contains every requested flag.
/// @tparam Mask The flag enum used by both arguments, such as MeshData or AuxConnType.
template <class Mask>
KOKKOS_INLINE_FUNCTION constexpr bool has(Mask mask, Mask flags) noexcept {
    return (static_cast<std::uint64_t>(mask) & static_cast<std::uint64_t>(flags)) ==
           static_cast<std::uint64_t>(flags);
}

/// @brief Sets the tolerances for face geometry checks.
struct Tolerance {
    /// Relative tolerance for metric comparisons. It also bounds the difference between weights in [0, 1].
    GeometryReal relative = GeometryReal{64} * std::numeric_limits<GeometryReal>::epsilon();
    /// Maximum allowed difference between the squared length of a normal and one.
    GeometryReal normal_absolute = GeometryReal{64} * std::numeric_limits<GeometryReal>::epsilon();
};

namespace detail {

/// @brief Accepts finite, strictly positive stored geometry values.
KOKKOS_INLINE_FUNCTION bool positive(GeometryReal x) { return Kokkos::isfinite(x) && x > 0; }

/// @brief Checks all three stored vector components for finiteness.
KOKKOS_INLINE_FUNCTION bool finite3(GeometryReal x, GeometryReal y, GeometryReal z) {
    return Kokkos::isfinite(x) && Kokkos::isfinite(y) && Kokkos::isfinite(z);
}

/// @brief Returns the larger scalar without host-only library calls.
KOKKOS_INLINE_FUNCTION GeometryReal max(GeometryReal a, GeometryReal b) { return a > b ? a : b; }

/// @brief Compares finite stored metrics with a relative tolerance.
KOKKOS_INLINE_FUNCTION bool close(GeometryReal a, GeometryReal b, GeometryReal tolerance) {
    return Kokkos::isfinite(a) && Kokkos::isfinite(b) &&
           Kokkos::abs(a - b) <= tolerance * max(Kokkos::abs(a), Kokkos::abs(b));
}

/// @brief Accepts a finite vector with at least one nonzero component.
KOKKOS_INLINE_FUNCTION bool nonzero3(GeometryReal x, GeometryReal y, GeometryReal z) {
    return finite3(x, y, z) && max(Kokkos::abs(x), max(Kokkos::abs(y), Kokkos::abs(z))) > 0;
}

/// @brief Checks local count relationships before any device array indexing.
inline bool valid_sizes(const MeshSizes& s) {
    return s.n_own >= 0 && s.n_cells >= s.n_own && s.n_nodes_own >= 0 && s.n_nodes >= s.n_nodes_own &&
           s.n_faces >= 0 &&
           (!s.faces_grouped || (s.n_internal_faces >= 0 && s.n_inner_faces >= s.n_internal_faces &&
                                 s.n_faces >= s.n_inner_faces));
}

/// @brief Rejects nonfinite or negative validation tolerances.
inline bool valid_tolerance(Tolerance t) {
    return Kokkos::isfinite(t.relative) && t.relative >= 0 && Kokkos::isfinite(t.normal_absolute) &&
           t.normal_absolute >= 0;
}

/// @brief Runs a check and returns its completed count on the host.
/// @tparam Exec Kokkos execution space used for the reduction.
/// @tparam Kernel The Kokkos functor that adds failures to the count.
template <class Exec, class Kernel>
std::size_t reduce(const char* label, const Exec& exec, std::size_t count, Kernel kernel) {
    std::size_t bad = 0;
    if (count != 0)
        Kokkos::parallel_reduce(
            label, Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>(exec, 0, count), kernel, bad);
    return bad;
}
} // namespace detail

} // namespace krona::mesh::device::validation
