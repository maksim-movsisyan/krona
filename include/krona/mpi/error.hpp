#pragma once

#include <mpi.h>

#include <source_location>
#include <string_view>

namespace krona::mpi {

namespace detail {

// Out-of-line failure path; not part of the public interface.
[[noreturn]] void mpi_failure(
    int error_code,
    MPI_Comm abort_comm,
    std::string_view operation,
    std::source_location where) noexcept;

} // namespace detail

// Verifies MPI call status. Fast path (MPI_SUCCESS) is a zero-overhead scalar check.
// Non-zero errors are treated as fatal (requires MPI_ERRORS_RETURN on the object).
//
// abort_comm: Target communicator to abort on failure (falls back to MPI_COMM_WORLD 
// if MPI_COMM_NULL). Must be valid if MPI is active.
//
// Preconditions:
// - Caller guarantees thread safety (no synchronization under MPI_THREAD_SERIALIZED).
// - MPI_Finalize must not race with this call.
// - 'operation' storage must remain valid for the call duration (null-terminator not required).
inline void check(
    int error_code,
    MPI_Comm abort_comm,
    std::string_view operation,
    std::source_location where = std::source_location::current()) noexcept {
    if (error_code != MPI_SUCCESS) [[unlikely]] {
        detail::mpi_failure(error_code, abort_comm, operation, where);
    }
}

// Non-collective, immediate termination without stack unwinding or MPI_Finalize.
// Calls MPI_Abort(abort_comm) if MPI is active (MPI_COMM_WORLD if MPI_COMM_NULL),
// otherwise terminates the local process via std::_Exit.
//
// Diagnostics are crash-safe (bounded stack formatting, zero C++ heap allocations).
// Inherits the same communicator, threading, and string lifetime preconditions as check().
[[noreturn]] void fatal(
    MPI_Comm abort_comm,
    std::string_view message,
    std::source_location where = std::source_location::current()) noexcept;

} // namespace krona::mpi
