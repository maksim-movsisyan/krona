#pragma once

#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>

namespace krona::halo::detail {

template<class> inline constexpr bool unsupported = false;

// Select pinned staging for the actual execution space, not Kokkos' default backend.
template<class Exec> struct StagingSpace { using type = Kokkos::HostSpace; };
#ifdef KOKKOS_ENABLE_CUDA
template<> struct StagingSpace<Kokkos::Cuda> { using type = Kokkos::CudaHostPinnedSpace; };
#endif
#ifdef KOKKOS_ENABLE_HIP
template<> struct StagingSpace<Kokkos::HIP> { using type = Kokkos::HIPHostPinnedSpace; };
#endif

struct Segment { 
    std::size_t offset = 0, count = 0;
};

} // namespace krona::halo::detail
