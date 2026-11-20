#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <type_traits>

namespace krona::transport::scalar::detail {

/// @brief Checks that a rank-one view covers the requested prefix and can be accessed by Exec.
/// A zero-length prefix also accepts an empty view.
/// @tparam Exec Kokkos execution space that will read or write the entries.
/// @tparam View Kokkos view of inputs or outputs, including strided components.
/// @param view Storage descriptor to check.
/// @param count Number of entries the operation will access from index zero.
template<class Exec, class View>
void validate_view(View view, std::size_t count) {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
    mpi::require(view.extent(0) >= count && (count == 0 || (view.data() != nullptr && view.stride(0) > 0)),
                 "Scalar transport: invalid component extent or storage");
}

/// @brief Checks storage for a writable floating-point output prefix.
/// @tparam Exec Kokkos execution space that will write the entries.
/// @tparam View Kokkos view whose value type must be mutable and floating-point.
/// @param view Output storage descriptor to check.
/// @param count Number of entries the operation may update from index zero.
template<class Exec, class View>
void validate_output(View view, std::size_t count) {
    static_assert(!std::is_const_v<typename View::value_type>);
    static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
    validate_view<Exec>(view, count);
}
} // namespace krona::transport::scalar::detail
