#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace krona::mpi {
namespace {

[[nodiscard]] constexpr bool valid(ThreadLevel level) noexcept {
    switch (level) {
    case ThreadLevel::single:
    case ThreadLevel::funneled:
    case ThreadLevel::serialized:
    case ThreadLevel::multiple:
        return true;
    }
    return false;
}

void require_main_thread(std::source_location where) noexcept {
    int main_thread = 0;
    check(MPI_Is_thread_main(&main_thread), MPI_COMM_WORLD,
          "MPI_Is_thread_main", where);
    if (main_thread == 0) {
        // Do not enter MPI again from a potentially forbidden thread.
        static_cast<void>(std::fprintf(
            stderr, "[KRONA fatal] Runtime requires the MPI main thread\n"
            "  at %s:%lu\n", where.file_name(),
            static_cast<unsigned long>(where.line())));
        static_cast<void>(std::fflush(stderr));
        std::_Exit(EXIT_FAILURE);
    }
}

} // namespace

Runtime::Runtime(
    int& argc,
    char**& argv,
    ThreadLevel required,
    std::source_location where) noexcept
    : Runtime(&argc, &argv, required, where) {}

Runtime::Runtime(ThreadLevel required, std::source_location where) noexcept
    : Runtime(nullptr, nullptr, required, where) {}

Runtime::Runtime(
    int* argc,
    char*** argv,
    ThreadLevel required,
    std::source_location where) noexcept
    : construction_thread_(std::this_thread::get_id()), origin_(where) {
    if (!valid(required)) {
        fatal(MPI_COMM_NULL, "Invalid requested MPI thread level", where);
    }

    int finalized = 0;
    check(MPI_Finalized(&finalized), MPI_COMM_NULL, "MPI_Finalized", where);
    if (finalized != 0) {
        fatal(MPI_COMM_NULL, "MPI cannot be reinitialized after finalization", where);
    }

    int initialized = 0;
    check(MPI_Initialized(&initialized), MPI_COMM_NULL, "MPI_Initialized", where);

    int provided = MPI_THREAD_SINGLE;
    if (initialized != 0) {
        require_main_thread(where);
        check(MPI_Query_thread(&provided), MPI_COMM_WORLD, "MPI_Query_thread", where);
    } else {
        check(MPI_Init_thread(argc, argv, static_cast<int>(required), &provided),
              MPI_COMM_NULL, "MPI_Init_thread", where);
        owns_mpi_ = true;
    }

    provided_ = static_cast<ThreadLevel>(provided);
    if (!valid(provided_)) {
        fatal(MPI_COMM_WORLD, "MPI returned an invalid thread level", where);
    }
    if (provided < static_cast<int>(required)) {
        char message[256]{};
        static_cast<void>(std::snprintf(
            message, sizeof(message),
            "Insufficient MPI thread support: required=%s, provided=%s",
            to_string(required).data(), to_string(provided_).data()));
        fatal(MPI_COMM_WORLD, message, where);
    }
}

Runtime::~Runtime() noexcept {
    if (closed_ || !owns_mpi_) {
        return;
    }
    require_construction_thread(origin_);
    if (std::uncaught_exceptions() != 0) {
        fatal(MPI_COMM_WORLD, "Exception unwinding with an owned MPI runtime", origin_);
    }
    finalize(origin_);
}

void Runtime::require_construction_thread(std::source_location where) const noexcept {
    if (std::this_thread::get_id() != construction_thread_) {
        // MPI calls are unsafe here under FUNNELED; terminate locally.
        static_cast<void>(std::fprintf(
            stderr, "[KRONA fatal] Runtime lifecycle used from another thread\n"
            "  at %s:%lu\n", where.file_name(),
            static_cast<unsigned long>(where.line())));
        static_cast<void>(std::fflush(stderr));
        std::_Exit(EXIT_FAILURE);
    }
}

void Runtime::finalize(std::source_location where) noexcept {
    require_construction_thread(where);
    if (closed_) {
        return;
    }
    if (owns_mpi_) {
        int finalized = 0;
        check(MPI_Finalized(&finalized), MPI_COMM_NULL, "MPI_Finalized", where);
        if (finalized != 0) {
            fatal(MPI_COMM_NULL, "Owned MPI runtime was finalized externally", where);
        }
        if (std::uncaught_exceptions() != 0) {
            fatal(MPI_COMM_WORLD, "MPI finalization during exception unwinding", where);
        }
        check(MPI_Finalize(), MPI_COMM_NULL, "MPI_Finalize", where);
        owns_mpi_ = false;
    }
    closed_ = true;
}

} // namespace krona::mpi
