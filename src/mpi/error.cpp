#include <krona/mpi/error.hpp>
#include <krona/mpi/context.hpp>

#include <cstdio>
#include <cstdlib>

namespace krona::mpi {

namespace {

/// @brief Prints the message and its call site without allocating a C++ string.
void report(
    std::string_view message,
    std::source_location where,
    int rank = -1) noexcept
{
    std::fprintf(stderr, "[KRONA fatal] world rank=%d\n  ", rank);

    // string_view does not require a trailing null terminator.
    if (!message.empty()) {
        std::fwrite(message.data(), 1, message.size(), stderr);
    }
    std::fprintf(stderr, "\n  at %s:%lu (%s)\n",
                 where.file_name(), static_cast<unsigned long>(where.line()),
                 where.function_name());
}


/// @brief Prints diagnostics and terminates the job or the local process.
[[noreturn]] void terminate_failure(
    std::string_view message,
    int error_code,
    std::source_location where) noexcept
{
    // Check if MPI is still alive before querying the rank or aborting.
    const bool active = is_active();
    int rank = -1;
    if (active) {
        // Diagnostic calls are best effort; do not recursively check them.
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    report(message, where, rank);

    if (error_code != MPI_SUCCESS) {
        std::fprintf(stderr, "  MPI error %d", error_code);
        if (active) {
            char text[MPI_MAX_ERROR_STRING]{};
            int length = 0;
            if (MPI_Error_string(error_code, text, &length) == MPI_SUCCESS) {
                std::fputs(": ", stderr);
                std::fwrite(text, 1, static_cast<std::size_t>(length), stderr);
            }
        }
        std::fputc('\n', stderr);
    }
    std::fflush(stderr);

    // A fatal error terminates the entire world, including other subcommunicators.
    if (active) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Covers inactive MPI and an MPI_Abort implementation that returns.
    std::_Exit(EXIT_FAILURE);
}

} // namespace

namespace detail {

/// @brief Routes a failed MPI call to the common termination path.
[[noreturn]] void mpi_failure(
    int error_code,
    std::string_view operation,
    std::source_location where) noexcept
{
    // Preserve the caller's location rather than this wrapper's location.
    terminate_failure(operation, error_code, where);
}

} // namespace detail

/// @brief Routes an application error to the common termination path.
[[noreturn]] void fatal(
    std::string_view message,
    std::source_location where) noexcept
{
    // MPI_SUCCESS indicates that there is no MPI error code to report.
    terminate_failure(message, MPI_SUCCESS, where);
}

} // namespace krona::mpi
