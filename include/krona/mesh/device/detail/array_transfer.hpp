#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mesh/device/mesh_view.hpp>
#include <krona/mpi/error.hpp>

#include <string>
#include <type_traits>
#include <vector>

namespace krona::mesh::device::detail {

/// @brief Shares the same borrow-or-copy path between base and auxiliary storage.
/// @tparam Exec Kokkos execution space for allocations and copies. If its memory
/// space is Kokkos::HostSpace, the view borrows the host vector directly.
template <class Exec>
struct ArrayTransfer {
    using memory_space = typename Exec::memory_space;
    static constexpr bool borrows_host = std::is_same_v<memory_space, Kokkos::HostSpace>;

    /// @brief Borrows a HostSpace source or enqueues a copy into an owned device buffer.
    /// @tparam T Element type of the host vector and the two Kokkos views.
    template <class T>
    static void attach(MeshArray<T, memory_space, false>& view, MeshArray<T, memory_space, true>& buffer,
                       const std::vector<T>& source, const char* label, const Exec& exec) {
        if constexpr (borrows_host) {
            view = MeshArray<T, memory_space, false>(source.data(), source.size());
        } else {
            if (source.empty())
                return;
            buffer = MeshArray<T, memory_space, true>(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), source.size());
            const MeshArray<T, Kokkos::HostSpace, false> host(source.data(), source.size());
            Kokkos::deep_copy(exec, buffer, host);
            view = buffer;
        }
    }

    /// @brief Completes queued device uploads; host borrowing requires no transfer fence.
    static void finish(const Exec& exec) {
        if constexpr (!borrows_host) exec.fence("KRONA mesh upload complete");
    }
};

} // namespace krona::mesh::device::detail
