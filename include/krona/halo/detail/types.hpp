#pragma once

#include <krona/mpi/datatype.hpp>
#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>

namespace krona::halo::detail {

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
