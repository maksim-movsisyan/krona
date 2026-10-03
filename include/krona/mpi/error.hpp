#pragma once

#include <mpi.h>

#include <source_location>
#include <string_view>

namespace krona::mpi {

namespace detail {

/// @brief Reports an MPI error and terminates the job.
[[noreturn]] void mpi_failure(
    int error_code,
    std::string_view operation,
    std::source_location where) noexcept;

} // namespace detail

/// @brief Checks an MPI return code and terminates the job on failure.
/// Use MPI_ERRORS_RETURN on the MPI object to receive errors here.
inline void check(
    int error_code,
    std::string_view operation,
    std::source_location where = std::source_location::current()) noexcept
{
    // Keep the successful path to a single comparison.
    if (error_code != MPI_SUCCESS) [[unlikely]] {
        detail::mpi_failure(error_code, operation, where);
    }
}

/// @brief Reports a fatal error and aborts MPI_COMM_WORLD while MPI is active.
/// Otherwise exits locally; no stack unwinding or MPI_Finalize is performed.
[[noreturn]] void fatal(
    std::string_view message,
    std::source_location where = std::source_location::current()) noexcept;

} // namespace krona::mpi
