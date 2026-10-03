#pragma once

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <source_location>

namespace krona::mpi {

/// @brief Returns whether MPI initialization has completed.
/// Remains true after MPI_Finalize; returns false if the query fails.
[[nodiscard]] inline bool is_initialized() noexcept
{
    // This query is valid before initialization and after finalization.
    int initialized = 0;
    return MPI_Initialized(&initialized) == MPI_SUCCESS && initialized != 0;
}

/// @brief Returns whether MPI finalization has completed.
/// Returns false if the query fails.
[[nodiscard]] inline bool is_finalized() noexcept
{
    // This query is valid even outside MPI's active lifetime.
    int finalized = 0;
    return MPI_Finalized(&finalized) == MPI_SUCCESS && finalized != 0;
}

/// @brief Returns whether MPI is initialized and not finalized.
/// Returns false if either state query fails.
[[nodiscard]] inline bool is_active() noexcept
{
    // Do not treat a failed state query as an active MPI context.
    if (!is_initialized()) {
        return false;
    }

    int finalized = 0;
    return MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0;
}

/// @brief Returns the provided MPI_THREAD_* level; MPI must be active.
[[nodiscard]] inline int thread_level() noexcept
{
    // Query the actual support level, which may differ from the requested one.
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    return provided;
}

/// @brief Returns whether this thread initialized MPI; MPI must be active.
[[nodiscard]] inline bool is_main_thread() noexcept
{
    // Ask MPI rather than tracking a separate C++ thread identifier.
    int main_thread = 0;
    MPI_Is_thread_main(&main_thread);
    return main_thread != 0;
}

/// @brief Requires the thread that initialized MPI; MPI must be active.
/// A wrong thread exits locally without making further MPI calls.
inline void require_main_thread(std::source_location where = std::source_location::current()) noexcept
{
    if (!is_main_thread()) {
        // Do not call MPI_Abort or query MPI from a forbidden thread.
        std::fprintf(stderr,
                     "[KRONA fatal] This operation requires the MPI main thread\n"
                     "  at %s:%lu (%s)\n",
                     where.file_name(), static_cast<unsigned long>(where.line()),
                     where.function_name());
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }
}

} // namespace krona::mpi