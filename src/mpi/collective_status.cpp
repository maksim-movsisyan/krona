#include <krona/mpi/collective_status.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace krona::mpi {

/// @brief Copies the first failure message into the local error record.
void set_local_error(
    LocalError& local,
    std::string_view message,
    std::source_location where) noexcept
{
    // Preserve the original failure if several checks fail on this rank.
    if (local.failed) {
        return;
    }
    if (message.empty()) {
        message = "Unspecified local preparation failure";
    }

    // Own the text so temporary strings and exception.what() remain safe.
    const auto length = std::min(message.size(), sizeof(local.message) - 1);
    std::memcpy(local.message, message.data(), length);
    local.message[length] = '\0';
    local.where = where;
    local.failed = true;
}

/// @brief Agrees on local success and reports the lowest failed rank once.
void require_collective_success(
    CommView comm,
    const LocalError& local,
    std::string_view stage,
    std::source_location where) noexcept
{
    if (comm.is_null()) {
        fatal("Collective status checking requires a live communicator", where);
    }

    // The communicator size is greater than every valid rank.
    const int no_failure = comm.size();
    const int candidate = local.failed ? comm.rank() : no_failure;
    int first_failed_rank = no_failure;
    check(MPI_Allreduce(&candidate, &first_failed_rank, 1, MPI_INT, MPI_MIN,
                        comm.native()), "MPI_Allreduce(status)", where);
    if (first_failed_rank == no_failure) {
        return;
    }

    // The selected rank already owns the diagnostic; no text broadcast is needed.
    if (comm.rank() == first_failed_rank) {
        std::fprintf(stderr,
                     "[KRONA collective failure] communicator rank=%d\n  stage: ",
                     first_failed_rank);
        if (!stage.empty()) {
            std::fwrite(stage.data(), 1, stage.size(), stderr);
        }
        std::fprintf(stderr, "\n  message: %s\n  at %s:%lu (%s)\n",
                     local.message, local.where.file_name(),
                     static_cast<unsigned long>(local.where.line()),
                     local.where.function_name());
        std::fflush(stderr);
    }

    // Wait for the selected rank to flush its diagnostic before anyone aborts.
    check(MPI_Barrier(comm.native()), "MPI_Barrier(status diagnostic)", where);

    // The error is already printed; terminate without duplicate fatal messages.
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::_Exit(EXIT_FAILURE);
}

} // namespace krona::mpi
