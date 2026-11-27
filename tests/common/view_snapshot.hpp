#pragma once

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace krona::test {

/// @brief Returns an independent host copy after the work on exec completes.
template<class Exec, class View>
auto host_snapshot(const Exec& exec, const View& view)
{
    if constexpr (View::rank == 1 && std::is_same_v<typename View::array_layout, Kokkos::LayoutStride>) {
        // Gather the selected entries before crossing memory spaces.
        using Value = typename View::non_const_value_type;
        Kokkos::View<Value*, typename View::memory_space> packed(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/packed_snapshot"), view.extent(0));
        auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, packed);
        Kokkos::deep_copy(exec, packed, view);
        Kokkos::deep_copy(exec, host, packed);
        exec.fence("Test: packed host snapshot ready");
        return host;
    } else {
        auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
        Kokkos::deep_copy(exec, host, view);
        exec.fence("Test: host snapshot ready");
        return host;
    }
}

} // namespace krona::test
