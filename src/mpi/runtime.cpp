#include <krona/mpi/runtime.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace krona::mpi {

namespace {

/// @brief Returns whether a value names a supported MPI thread level.
[[nodiscard]] constexpr bool valid(ThreadLevel level) noexcept
{
    // A cast to an enum may still produce an invalid value.
    switch (level) {
    case ThreadLevel::single:
    case ThreadLevel::funneled:
    case ThreadLevel::serialized:
    case ThreadLevel::multiple:
        return true;
    default:
        return false;
    }
}

} // namespace

/// @brief Delegates initialization with command-line arguments.
Runtime::Runtime(
    int& argc,
    char**& argv,
    ThreadLevel required,
    std::source_location where) noexcept : Runtime(&argc, &argv, required, where) {}

/// @brief Delegates initialization without command-line arguments.
Runtime::Runtime(ThreadLevel required, std::source_location where) noexcept : Runtime(nullptr, nullptr, required, where) {}

/// @brief Initializes MPI if needed and checks the provided thread support.
Runtime::Runtime(
    int* argc,
    char*** argv,
    ThreadLevel required,
    std::source_location where) noexcept : construction_thread_(std::this_thread::get_id()), origin_(where)
{
    // MPI cannot be initialized again after finalization.
    require(valid(required), "Invalid requested MPI thread level", where);
    require(!is_finalized(), "MPI cannot be reinitialized after finalization", where);

    int provided = MPI_THREAD_SINGLE;
    if (is_initialized()) {
        // Borrow the existing runtime without changing its ownership.
        require_main_thread(where);
        provided = krona::mpi::thread_level();
    } else {
        // Only the guard that initializes MPI may finalize it.
        check(MPI_Init_thread(argc, argv, static_cast<int>(required), &provided),
              "MPI_Init_thread", where);
        owns_mpi_ = true;
    }

    // The actual support level may be lower than requested.
    provided_ = static_cast<ThreadLevel>(provided);
    require(valid(required), "MPI returned an invalid thread level", where);
    require(provided >= static_cast<int>(required), "Insufficient MPI thread support", where);
}

/// @brief Finalizes an owned runtime through the common lifecycle path.
Runtime::~Runtime() noexcept
{
    // Borrowed guards do not finalize MPI or make lifecycle MPI calls.
    if (owns_mpi_) {
        finalize(origin_);
    }
}

/// @brief Rejects lifecycle calls from a different C++ thread.
void Runtime::require_construction_thread(std::source_location where) const noexcept
{
    if (std::this_thread::get_id() != construction_thread_) {
        // MPI calls may be forbidden here; print and exit locally.
        std::fprintf(stderr,
                     "[KRONA fatal] Runtime lifecycle used from another thread\n"
                     "  at %s:%lu (%s)\n",
                     where.file_name(), static_cast<unsigned long>(where.line()),
                     where.function_name());
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }
}

/// @brief Finalizes owned MPI once or detaches a borrowed guard.
void Runtime::finalize(std::source_location where) noexcept
{
    // Check the C++ thread before making any MPI calls.
    require_construction_thread(where);
    if (closed_) {
        return;
    }

    if (owns_mpi_) {
        require(is_active(), "Owned MPI runtime is no longer active; it must not be finalized externally", where);

        // Finalizing during unwinding may mismatch other ranks' collective calls.
        require(std::uncaught_exceptions() == 0, "MPI finalization during exception unwinding", where);

        check(MPI_Finalize(), "MPI_Finalize", where);
        owns_mpi_ = false;
    }

    // Borrowed guards only close themselves; repeated calls become a no-op.
    closed_ = true;
}

} // namespace krona::mpi