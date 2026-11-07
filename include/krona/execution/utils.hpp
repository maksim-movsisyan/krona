#pragma once

#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>

#include <utility>
#include <source_location>


namespace krona::execution {

inline void require_kokkos(std::source_location where = std::source_location::current()) noexcept
{
    mpi::require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
                 "Operation require active Kokkos", where);
}
 
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

} //namespace krona::execution