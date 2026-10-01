#include <krona/mpi/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace krona::mpi {
namespace {

// These state queries are permitted before MPI_Init and after MPI_Finalize.
[[nodiscard]] bool mpi_is_active() noexcept {
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
        return false;
    }

    int finalized = 0;
    return MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0;
}

struct Text {
    int size;
    const char* data;
    const char* suffix;
};

// Bound each field separately, so a long message cannot hide its location.
// Precision-limited %s accepts a non-null-terminated string_view.
[[nodiscard]] Text bounded(std::string_view value, std::size_t limit) noexcept {
    return {
        static_cast<int>(std::min(value.size(), limit)),
        value.empty() ? "" : value.data(),
        value.size() > limit ? "... [truncated]" : ""
    };
}

[[noreturn]] void terminate_failure(
    MPI_Comm abort_comm,
    std::string_view context,
    int error_code,
    bool is_mpi_error,
    std::source_location where) noexcept {
    const bool active = mpi_is_active();
    MPI_Comm scope = MPI_COMM_NULL;
    int rank = -1;

    char mpi_text[MPI_MAX_ERROR_STRING]{};
    std::string_view description = "unavailable";

    if (active) {
        scope = abort_comm == MPI_COMM_NULL ? MPI_COMM_WORLD : abort_comm;

        // Diagnostics after an MPI error are best effort. Do not recursively
        // check failures here and never introduce a collective operation.
        int reported_rank = -1;
        if (MPI_Comm_rank(scope, &reported_rank) == MPI_SUCCESS) {
            rank = reported_rank;
        }

        if (is_mpi_error) {
            int length = 0;
            if (MPI_Error_string(error_code, mpi_text, &length) == MPI_SUCCESS
                && length >= 0 && length <= MPI_MAX_ERROR_STRING) {
                description = std::string_view{
                    mpi_text, static_cast<std::size_t>(length)};
            }
        }
    }

    const Text text = bounded(context, 2048);
    const Text file = bounded(where.file_name(), 512);
    const Text function = bounded(where.function_name(), 512);
    const Text error = bounded(description, MPI_MAX_ERROR_STRING);

    // One bounded write reduces interleaving, but cross-process stderr
    // ordering / atomicity depends on the launcher and is not guaranteed.
    char diagnostic[4096 + MPI_MAX_ERROR_STRING]{};
    int length = 0;
    if (is_mpi_error) {
        length = std::snprintf(
            diagnostic, sizeof(diagnostic),
            "[KRONA MPI fatal] abort-comm rank=%d, MPI=%s\n"
            "  operation: %.*s%s\n"
            "  MPI error %d: %.*s%s\n"
            "  at %.*s%s:%lu:%lu (%.*s%s)\n",
            rank, active ? "active" : "inactive",
            text.size, text.data, text.suffix,
            error_code, error.size, error.data, error.suffix,
            file.size, file.data, file.suffix,
            static_cast<unsigned long>(where.line()),
            static_cast<unsigned long>(where.column()),
            function.size, function.data, function.suffix);
    } else {
        length = std::snprintf(
            diagnostic, sizeof(diagnostic),
            "[KRONA fatal] abort-comm rank=%d, MPI=%s\n"
            "  message: %.*s%s\n"
            "  at %.*s%s:%lu:%lu (%.*s%s)\n",
            rank, active ? "active" : "inactive",
            text.size, text.data, text.suffix,
            file.size, file.data, file.suffix,
            static_cast<unsigned long>(where.line()),
            static_cast<unsigned long>(where.column()),
            function.size, function.data, function.suffix);
    }

    if (length > 0) {
        const auto bytes = std::min(
            static_cast<std::size_t>(length), sizeof(diagnostic) - 1);
        static_cast<void>(std::fwrite(diagnostic, 1, bytes, stderr));
    } else {
        static_cast<void>(std::fputs("[KRONA fatal] diagnostic unavailable\n", stderr));
    }
    static_cast<void>(std::fflush(stderr));

    if (active) {
        // Use a portable nonzero exit status. The original MPI error code
        // is diagnostic data and need not be a suitable process exit code.
        static_cast<void>(MPI_Abort(scope, EXIT_FAILURE));
    }

    // Covers inactive MPI and implementations where MPI_Abort returns.
    // Do not run destructors / atexit handlers that could enter collectives.
    std::_Exit(EXIT_FAILURE);
}

} // namespace

namespace detail {

[[noreturn]] void mpi_failure(
    int error_code,
    MPI_Comm abort_comm,
    std::string_view operation,
    std::source_location where) noexcept {
    terminate_failure(abort_comm, operation, error_code, true, where);
}

} // namespace detail

[[noreturn]] void fatal(
    MPI_Comm abort_comm,
    std::string_view message,
    std::source_location where) noexcept {
    terminate_failure(abort_comm, message, MPI_SUCCESS, false, where);
}

} // namespace krona::mpi
