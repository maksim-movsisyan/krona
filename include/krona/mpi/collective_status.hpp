#pragma once

#include <krona/mpi/communicator.hpp>

#include <source_location>
#include <string_view>

namespace krona::mpi {

/// @brief Stores the first local failure without allocating memory.
/// Text is copied and limited to 511 bytes; use a fresh object for each stage.
struct LocalError final
{
    char message[512]{};
    std::source_location where{};
    bool failed = false;
};

/// @brief Records the first local error and its location; makes no MPI calls.
void set_local_error(
    LocalError& local,
    std::string_view message,
    std::source_location where = std::source_location::current()) noexcept;

/// @brief Returns if all ranks succeeded; otherwise prints one error and aborts WORLD.
/// Collective on a live communicator: one Allreduce, plus a barrier on failure.
void require_collective_success(
    CommView comm,
    const LocalError& local,
    std::string_view stage = "Local preparation",
    std::source_location where = std::source_location::current()) noexcept;

// All ranks, including failed ones, must call the check in matching order.
// Local preparation and its exception unwinding must not enter MPI collectives.
// Callers keep the communicator live and obey MPI's thread rules.

} // namespace krona::mpi
