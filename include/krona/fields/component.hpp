#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <cstddef>

namespace krona::fields {

/// @brief Borrows one component across entities using a pointer and a stride.
/// @tparam T Type of one value, including const for a read-only component.
/// @tparam Space Kokkos memory space where data points.
/// The caller supplies valid indices and keeps the allocation alive.
template<class T, class Space>
struct ComponentDescriptor {
    using value_type = T;
    using memory_space = Space;

    T* data = nullptr;
    /// Distance from one entity's value to the next, measured in elements of type T.
    std::size_t stride = 1;

    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t i) const { return data[i*stride]; }
};

/// @brief Extracts the pointer and stride from a rank-one view or component subview.
/// @tparam View Type of the rank-one Kokkos view. Its value type and memory space are preserved.
/// @param view Source view. A nonempty view must have a data pointer and a positive stride.
/// @return A descriptor that borrows the values for use in a kernel.
template<class View>
[[nodiscard]] auto descriptor_from_view(View view) {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1,
                  "Fields: a packed Kokkos view must have rank one");
    mpi::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0),
                 "Fields: invalid packed component storage");
    return ComponentDescriptor<typename View::value_type, typename View::memory_space>{view.data(), view.stride(0)};
}

} // namespace krona::fields
