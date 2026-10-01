#pragma once

#include <mpi.h>

#include <source_location>
#include <string_view>
#include <thread>

namespace krona::mpi {

enum class ThreadLevel : int {
    single = MPI_THREAD_SINGLE,
    funneled = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple = MPI_THREAD_MULTIPLE
};

[[nodiscard]] constexpr std::string_view to_string(ThreadLevel level) noexcept {
    switch (level) {
    case ThreadLevel::single:     return "MPI_THREAD_SINGLE";
    case ThreadLevel::funneled:   return "MPI_THREAD_FUNNELED";
    case ThreadLevel::serialized: return "MPI_THREAD_SERIALIZED";
    case ThreadLevel::multiple:   return "MPI_THREAD_MULTIPLE";
    }
    return "invalid MPI thread level";
}

// Lifecycle calls belong to the MPI main thread and must not race.
// Release MPI resources and complete communication before owned finalization.
class Runtime final {
public:
    // Initialize MPI if needed; otherwise borrow the existing runtime.
    Runtime(
        int& argc,
        char**& argv,
        ThreadLevel required = ThreadLevel::funneled,
        std::source_location where = std::source_location::current()) noexcept;

    // Same lifecycle, with no command-line arguments passed to MPI.
    explicit Runtime(
        ThreadLevel required = ThreadLevel::funneled,
        std::source_location where = std::source_location::current()) noexcept;

    // Finalize owned MPI on normal exit; abort during exception unwinding.
    ~Runtime() noexcept;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    [[nodiscard]] bool owns_mpi() const noexcept { return owns_mpi_; }
    [[nodiscard]] ThreadLevel thread_level() const noexcept { return provided_; }

    // Finalize owned MPI collectively; borrowed guards simply detach.
    // Repeated calls are harmless. Use on the construction thread.
    void finalize(
        std::source_location where = std::source_location::current()) noexcept;

private:
    Runtime(
        int* argc,
        char*** argv,
        ThreadLevel required,
        std::source_location where) noexcept;

    void require_construction_thread(std::source_location where) const noexcept;

    std::thread::id construction_thread_;
    std::source_location origin_;
    ThreadLevel provided_ = ThreadLevel::single;
    bool owns_mpi_ = false;
    bool closed_ = false;
};

} // namespace krona::mpi
