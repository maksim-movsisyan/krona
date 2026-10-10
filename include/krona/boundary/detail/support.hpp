#pragma once

#include <krona/mpi/error.hpp>

#include <exception>
#include <source_location>
#include <utility>

namespace krona::boundary::detail {

inline void require(bool condition, const char* message,
                    std::source_location where = std::source_location::current()) noexcept
{
    if (!condition) [[unlikely]] mpi::fatal(message, where);
}

template<class Function>
decltype(auto) guarded(Function&& function)
{
    try { return std::forward<Function>(function)(); }
    catch (const std::exception& error) { mpi::fatal(error.what()); }
    catch (...) { mpi::fatal("Scalar boundary: unknown exception"); }
}

} // namespace krona::boundary::detail
