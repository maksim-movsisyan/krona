#pragma once

#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>

namespace krona::halo::detail {

template<class> inline constexpr bool unsupported = false;

/// @brief Chooses the host buffer space used for MPI staging.
/// @tparam Exec Kokkos execution space that copies the staging buffers to or from device memory.
/// CUDA and HIP use pinned host memory. The general case uses ordinary HostSpace.
template<class Exec> struct StagingSpace { using type = Kokkos::HostSpace; };
#ifdef KOKKOS_ENABLE_CUDA
template<> struct StagingSpace<Kokkos::Cuda> { using type = Kokkos::CudaHostPinnedSpace; };
#endif
#ifdef KOKKOS_ENABLE_HIP
template<> struct StagingSpace<Kokkos::HIP> { using type = Kokkos::HIPHostPinnedSpace; };
#endif

/// @brief One peer's range in the send or receive cell map.
struct Segment {
    std::size_t offset = 0; ///< First slot in the map, counted in cells.
    std::size_t count = 0;  ///< Number of cells in that peer's range.
};

} // namespace krona::halo::detail