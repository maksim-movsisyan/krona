#pragma once

#include <mpi.h>

#include <source_location>
#include <string_view>
#include <thread>

namespace krona::mpi {

/// @brief Available MPI thread support levels.
enum class ThreadLevel : int
{
    single = MPI_THREAD_SINGLE,
    funneled = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple = MPI_THREAD_MULTIPLE
};

/// @brief Returns the MPI name of a thread support level.
[[nodiscard]] constexpr std::string_view to_string(ThreadLevel level) noexcept
{
    // String literals remain valid for the lifetime of the program.
    switch (level) {
    case ThreadLevel::single:     return "MPI_THREAD_SINGLE";
    case ThreadLevel::funneled:   return "MPI_THREAD_FUNNELED";
    case ThreadLevel::serialized: return "MPI_THREAD_SERIALIZED";
    case ThreadLevel::multiple:   return "MPI_THREAD_MULTIPLE";
    }
    return "invalid MPI thread level";
}

/// @brief Initializes MPI or borrows an existing runtime on the MPI main thread.
/// Owned MPI is finalized on normal exit; all MPI resources must be released first.
class Runtime final
{
public:
    /// @brief Initializes or borrows MPI, passing command-line arguments if needed.
    Runtime(
        int& argc,
        char**& argv,
        ThreadLevel required = ThreadLevel::funneled,
        std::source_location where = std::source_location::current()) noexcept;

    /// @brief Initializes or borrows MPI without passing command-line arguments.
    explicit Runtime(
        ThreadLevel required = ThreadLevel::funneled,
        std::source_location where = std::source_location::current()) noexcept;

    /// @brief Finalizes owned MPI on normal exit; aborts during exception unwinding.
    ~Runtime() noexcept;

    /// @brief Runtime ownership cannot be copied.
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    /// @brief Runtime guards cannot be moved to another object or thread.
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    /// @brief Returns whether this guard owns MPI initialization.
    [[nodiscard]] bool owns_mpi() const noexcept { return owns_mpi_; }

    /// @brief Returns the cached provided thread support level.
    [[nodiscard]] ThreadLevel thread_level() const noexcept { return provided_; }

    /// @brief Finalizes owned MPI collectively; a borrowed guard only detaches.
    /// Repeated calls are harmless; call from the construction thread.
    void finalize(std::source_location where = std::source_location::current()) noexcept;

private:
    /// @brief Implements the shared initialization and borrowing path.
    Runtime(
        int* argc,
        char*** argv,
        ThreadLevel required,
        std::source_location where) noexcept;

    /// @brief Checks the lifecycle thread without making MPI calls.
    void require_construction_thread(std::source_location where) const noexcept;

    std::thread::id construction_thread_;
    std::source_location origin_;
    ThreadLevel provided_ = ThreadLevel::single;
    bool owns_mpi_ = false;
    bool closed_ = false;
};

} // namespace krona::mpi