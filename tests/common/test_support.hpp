#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/runtime.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <source_location>
#include <span> 
#include <string_view>
#include <utility>

namespace krona::test {

using Exec = execution::ExecutionSpace;
using Index = LocalIndex;
using Real = StorageReal;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<Index>>;

class Failure final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "Test check failed"; }
};

/// @brief Checks a condition in every build configuration, including Release.
inline void check(bool condition, std::string_view message,
                  std::source_location where = std::source_location::current())
{
    if (condition) return;
    std::cerr << where.file_name() << ':' << where.line() << ": " << message << '\n';
    throw Failure{};
}

/// @brief Reports expected/actual values before failing an exact comparison.
template<class Actual, class Expected>
void equal(const Actual& actual, const Expected& expected, std::string_view message,
           std::source_location where = std::source_location::current())
{
    if (actual == expected) return;
    std::cerr << "  expected=" << expected << " actual=" << actual << '\n';
    check(false, message, where);
}

/// @brief Compares small floating-point reference results with an absolute tolerance.
template<class T>
void near(T actual, T expected, T tolerance, std::string_view message,
          std::source_location where = std::source_location::current())
{
    if (std::isfinite(actual) && std::abs(actual - expected) <= tolerance) return;
    std::cerr << "  expected=" << expected << " actual=" << actual
              << " tolerance=" << tolerance << '\n';
    check(false, message, where);
}

/// @brief Submits work and drains it before returning or propagating an exception.
/// Allocate referenced buffers outside submit so they outlive its stack unwinding.
template<class Submit>
void complete(const Exec& exec, Submit&& submit)
{
    try {
        std::forward<Submit>(submit)();
        exec.fence("KRONA test: work complete");
    } catch (...) {
        exec.fence("KRONA test: drain failed work");
        throw;
    }
}

struct Case {
    std::string_view name;
    void (*function)(const Exec&);
};

/// @brief Runs one named case or the whole suite in a single MPI process.
inline int run(int argc, char** argv, std::string_view suite, std::span<const Case> cases)
{
    std::cerr << "[test] Before MPI initialization\n" << std::flush;
    mpi::Runtime mpi_runtime{argc, argv, mpi::ThreadLevel::funneled};
    std::cerr << "[test] After MPI initialization\n" << std::flush;

    try {
        std::cerr << "[test] Before Kokkos initialization\n" << std::flush;
        Kokkos::ScopeGuard kokkos{argc, argv};
        std::cerr << "[test] After Kokkos initialization\n" << std::flush;
        check(argc <= 2, "Expected at most one test-case name");
        const std::string_view selected = argc == 2 ? argv[1] : "";
        Exec exec{};
        bool found = false;
        for (const auto& entry : cases) {
            if (!selected.empty() && selected != entry.name) continue;
            found = true;
            std::cout << "RUN  " << suite << '.' << entry.name << " [" << Exec::name() << "]\n";
            entry.function(exec);
            std::cout << "PASS " << suite << '.' << entry.name << '\n';
        }
        check(found, "Unknown test-case name");
        return EXIT_SUCCESS;
    } catch (const Failure&) {
        // Cases drain submitted work before this normal shutdown path.
        std::cerr << "FAIL " << suite << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unhandled test exception");
    }
}

} // namespace krona::test
