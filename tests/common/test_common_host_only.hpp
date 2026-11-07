#pragma once

#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/runtime.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <source_location>
#include <span>
#include <string_view>

namespace krona::test::host {
namespace detail {

// Each process runs one suite on the MPI main thread; the name is borrowed by run().
inline std::string_view suite_name;

} // namespace detail

/// @brief Checks a condition in every build mode and aborts MPI on failure.
inline void check(bool condition, std::string_view message,
                  std::source_location where = std::source_location::current()) {
    if (condition)
        return;
    std::cerr << where.file_name() << ':' << where.line() << ": " << message << '\n';
    if (!detail::suite_name.empty())
        std::cerr << "FAIL " << detail::suite_name << '\n';
    // Abort before unwinding test resources or leaving peers inside a collective.
    mpi::fatal("Test check failed", where);
}

/// @brief Reports expected and actual values before failing an exact comparison.
template <class Actual, class Expected>
void equal(const Actual &actual, const Expected &expected, std::string_view message,
           std::source_location where = std::source_location::current()) {
    if (actual == expected)
        return;
    std::cerr << "  expected=" << expected << " actual=" << actual << '\n';
    check(false, message, where);
}

/// @brief Compares finite reference values using an absolute tolerance.
template <class T>
void near(T actual, T expected, T tolerance, std::string_view message,
          std::source_location where = std::source_location::current()) {
    if (std::isfinite(actual) && std::abs(actual - expected) <= tolerance)
        return;
    std::cerr << "  expected=" << expected << " actual=" << actual << " tolerance=" << tolerance << '\n';
    check(false, message, where);
}

/// @brief Names a host test that all ranks execute with the same communicator.
struct Case {
    std::string_view name;
    void (*function)(mpi::CommView);
};

/// @brief Runs one named case or the suite with MPI and rank-zero RUN/PASS output.
inline int run(int argc, char **argv, std::string_view suite, std::span<const Case> cases) {
    mpi::Runtime runtime{argc, argv, mpi::ThreadLevel::funneled};
    const auto comm = mpi::CommView::world();
    detail::suite_name = suite;

    return mpi::guarded([&] {
        check(argc <= 2, "Expected at most one test-case name");
        const std::string_view selected = argc == 2 ? argv[1] : "";
        bool found = false;
        for (const auto &entry : cases) {
            if (!selected.empty() && selected != entry.name)
                continue;
            found = true;
            if (comm.rank() == 0)
                std::cout << "RUN  " << suite << '.' << entry.name << " [Host]\n" << std::flush;
            entry.function(comm);
            // Report success only after every rank has finished the case.
            mpi::barrier(comm);
            if (comm.rank() == 0)
                std::cout << "PASS " << suite << '.' << entry.name << '\n' << std::flush;
        }
        check(found, "Unknown test-case name");
        detail::suite_name = {};
        return EXIT_SUCCESS;
    });
}

} // namespace krona::test::host
