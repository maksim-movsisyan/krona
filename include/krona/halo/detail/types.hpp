#pragma once

#include <krona/mpi/datatype.hpp>
#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <limits>
#include <utility>

namespace krona::halo::detail {

inline std::size_t product(std::size_t a, std::size_t b)
{
    mpi::require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
            "Halo: payload size overflows size_t");
    return a * b;
}

// Setup-only copies use temporary mirrors. Drain work while those mirrors are still alive.
template<class Exec, class Submit>
void complete(const Exec& exec, Submit&& submit)
{
    try {
        std::forward<Submit>(submit)();
        exec.fence("KRONA halo: setup copies complete");
    } catch (...) {
        exec.fence("KRONA halo: drain failed setup copies");
        throw;
    }
}

template<class> inline constexpr bool unsupported = false;

template<class T>
MPI_Datatype datatype() { return mpi::datatype<T>(); }

// Select pinned staging for the actual execution space, not Kokkos' default backend.
template<class Exec> struct StagingSpace { using type = Kokkos::HostSpace; };
#ifdef KOKKOS_ENABLE_CUDA
template<> struct StagingSpace<Kokkos::Cuda> { using type = Kokkos::CudaHostPinnedSpace; };
#endif
#ifdef KOKKOS_ENABLE_HIP
template<> struct StagingSpace<Kokkos::HIP> { using type = Kokkos::HIPHostPinnedSpace; };
#endif

template<class T>
struct Component {
    T* data = nullptr;
    std::size_t stride = 1;
};

struct Segment {
    std::size_t offset = 0, count = 0;
};

} // namespace krona::halo::detail
