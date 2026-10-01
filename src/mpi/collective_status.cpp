#include <krona/mpi/collective_status.hpp>
#include <krona/mpi/datatype.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace krona::mpi {
namespace {

void require_status_comm(CommView comm, std::source_location where) noexcept {
    if (comm.is_null() || comm.is_intercommunicator()) {
        fatal(comm.abort_scope(), "Status agreement requires a live intracommunicator", where);
    }
    int initialized = 0;
    int finalized = 0;
    check(MPI_Initialized(&initialized), MPI_COMM_NULL, "MPI_Initialized", where);
    check(MPI_Finalized(&finalized), MPI_COMM_NULL, "MPI_Finalized", where);
    if (initialized == 0 || finalized != 0) {
        fatal(MPI_COMM_NULL, "Status agreement requires active MPI", where);
    }
}

struct Text {
    int size;
    const char* data;
    const char* suffix;
};

[[nodiscard]] Text bounded(std::string_view text, std::size_t limit) noexcept {
    return {static_cast<int>(std::min(text.size(), limit)),
            text.empty() ? "" : text.data(),
            text.size() > limit ? "... [truncated]" : ""};
}

} // namespace

void LocalStatus::fail(std::string_view message, std::source_location where) noexcept {
    if (failed_) {
        return;
    }
    failed_ = true;
    where_ = where;
    if (message.empty()) {
        message = "Unspecified local preparation failure";
    }

    constexpr std::string_view suffix = "... [truncated]";
    constexpr std::size_t limit = message_capacity - 1;
    if (message.size() > limit) {
        message_size_ = limit;
        const auto prefix = limit - suffix.size();
        std::memcpy(message_.data(), message.data(), prefix);
        std::memcpy(message_.data() + prefix, suffix.data(), suffix.size());
    } else {
        message_size_ = message.size();
        std::memcpy(message_.data(), message.data(), message_size_);
    }
    message_[message_size_] = '\0';
}

LocalStatus LocalStatus::failure(std::string_view message, std::source_location where) noexcept {
    LocalStatus result;
    result.fail(message, where);
    return result;
}

CollectiveStatus synchronize_status(
    CommView comm,
    const LocalStatus& local,
    std::source_location where) noexcept {
    require_status_comm(comm, where);
    constexpr int no_failure = std::numeric_limits<int>::max();
    const int candidate = local.ok() ? no_failure : comm.rank();
    int first_failed_rank = no_failure;
    check(MPI_Allreduce(&candidate, &first_failed_rank, 1, datatype<int>(), MPI_MIN,
                        comm.native()), comm.abort_scope(), "MPI_Allreduce(status)", where);

    CollectiveStatus result;
    result.comm_ = comm;
    if (first_failed_rank == no_failure) {
        return result;
    }
    result.first_failed_rank_ = first_failed_rank;

    if (comm.rank() == first_failed_rank) {
        const auto location = local.location();
        const Text message = bounded(local.message(), LocalStatus::message_capacity - 1);
        const Text file = bounded(location.file_name(), 256);
        const Text function = bounded(location.function_name(), 128);
        static_cast<void>(std::snprintf(
            result.diagnostic_.data(), result.diagnostic_.size(),
            "communicator rank=%d: %.*s%s\n  at %.*s%s:%lu:%lu (%.*s%s)",
            first_failed_rank, message.size, message.data, message.suffix,
            file.size, file.data, file.suffix,
            static_cast<unsigned long>(location.line()),
            static_cast<unsigned long>(location.column()),
            function.size, function.data, function.suffix));
    }

    // Send diagnostic text, never source_location pointers or C++ object layout.
    check(MPI_Bcast(result.diagnostic_.data(),
                    static_cast<int>(result.diagnostic_.size()), MPI_CHAR,
                    first_failed_rank, comm.native()),
          comm.abort_scope(), "MPI_Bcast(status diagnostic)", where);
    return result;
}

void CollectiveStatus::require_success(
    std::string_view stage,
    std::source_location where) const noexcept {
    if (ok()) {
        return;
    }
    require_status_comm(comm_, where);
    if (comm_.rank() == first_failed_rank_) {
        const Text label = bounded(stage, 256);
        static_cast<void>(std::fprintf(
            stderr, "[KRONA collective failure] %.*s%s\n  %s\n",
            label.size, label.data, label.suffix, diagnostic_.data()));
        static_cast<void>(std::fflush(stderr));
    }

    // Root flushes its diagnostic before peers are allowed to abort.
    int diagnostic_ready = 1;
    check(MPI_Bcast(&diagnostic_ready, 1, datatype<int>(), first_failed_rank_, comm_.native()),
          comm_.abort_scope(), "MPI_Bcast(status abort acknowledgement)", where);
    static_cast<void>(MPI_Abort(comm_.abort_scope(), EXIT_FAILURE));
    std::_Exit(EXIT_FAILURE);
}

} // namespace krona::mpi
