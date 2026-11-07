#pragma once

#include <krona/config.hpp>
#include <Kokkos_Core.hpp>

#if (defined(KRONA_EXECUTION_SPACE_SERIAL) + defined(KRONA_EXECUTION_SPACE_OPENMP) + \
     defined(KRONA_EXECUTION_SPACE_CUDA) + defined(KRONA_EXECUTION_SPACE_HIP)) != 1
#error "Select exactly one KRONA execution space in the generated config.hpp"
#endif

namespace krona::execution {
 
#if defined(KRONA_EXECUTION_SPACE_SERIAL)
using ExecutionSpace = Kokkos::Serial;
#elif defined(KRONA_EXECUTION_SPACE_OPENMP)
using ExecutionSpace = Kokkos::OpenMP;
#elif defined(KRONA_EXECUTION_SPACE_CUDA)
using ExecutionSpace = Kokkos::Cuda;
#elif defined(KRONA_EXECUTION_SPACE_HIP)
using ExecutionSpace = Kokkos::HIP;
#endif

using MemorySpace = typename ExecutionSpace::memory_space;
using Device = Kokkos::Device<ExecutionSpace, MemorySpace>;

} // namespace krona::execution
