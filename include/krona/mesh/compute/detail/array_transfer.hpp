#pragma once

#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/mpi/error.hpp>

#include <cstdio>
#include <source_location>
#include <string>
#include <type_traits>
#include <vector>

namespace krona::mesh::compute::detail {

template<class T>
void require_size(const std::vector<T>& source, std::size_t expected, const char* name,
                  std::source_location where = std::source_location::current()) noexcept
{
    if (source.size() != expected) [[unlikely]] {
        // Format the diagnostic without allocating memory on the error path.
        char message[512]{};
        std::snprintf(message, sizeof(message),
                      "Mesh compute: invalid size of %s (expected %zu, got %zu)",
                      name, expected, source.size());
        krona::mpi::fatal(message, where);
    }
}

inline void require_csr(const std::vector<std::size_t>& offsets,
                        const std::vector<LocalIndex>& values, std::size_t rows,
                        const char* name,
                        std::source_location where = std::source_location::current()) noexcept
{
    require_size(offsets, rows + 1, name, where);
    mpi::require(offsets.front() == 0 && offsets.back() == values.size(), "Mesh compute: invalid CSR bounds", where);
}

inline void require_kokkos(
    std::source_location where = std::source_location::current()) noexcept
{
    mpi::require(Kokkos::is_initialized() && !Kokkos::is_finalized(), "Mesh compute requires active Kokkos", where);
}

// Shared transfer mechanism; no host cache is built or modified here.
template<class Exec>
struct ArrayTransfer {
    using memory_space = typename Exec::memory_space;
    static constexpr bool borrows_host = std::is_same_v<memory_space, Kokkos::HostSpace>;

    template<class T>
    static void attach(MeshArray<T, memory_space, false>& view,
                       MeshArray<T, memory_space, true>& buffer,
                       const std::vector<T>& source, const char* label, const Exec& exec)
    {
        if constexpr (borrows_host) {
            view = MeshArray<T, memory_space, false>(source.data(), source.size());
        } else {
            if (source.empty()) return;
            buffer = MeshArray<T, memory_space, true>(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), source.size());
            const MeshArray<T, Kokkos::HostSpace, false> host(source.data(), source.size());
            Kokkos::deep_copy(exec, buffer, host);
            view = buffer;
        }
    }

    static void finish(const Exec& exec)
    {
        if constexpr (!borrows_host) exec.fence("KRONA mesh upload complete");
    }
};

} // namespace krona::mesh::compute::detail