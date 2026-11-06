#pragma once

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <source_location>
#include <string_view>
#include <algorithm>

namespace krona::mpi {

namespace detail {

/// @brief Reports an MPI error and terminates the job.
[[noreturn]] void mpi_failure(
    int error_code,
    std::string_view operation,
    std::source_location where) noexcept;

} // namespace detail

/// @brief Checks an MPI return code and terminates the job on failure.
/// Use MPI_ERRORS_RETURN on the MPI object to receive errors here.
inline void check(
    int error_code,
    std::string_view operation,
    std::source_location where = std::source_location::current()) noexcept
{
    // Keep the successful path to a single comparison.
    if (error_code != MPI_SUCCESS) [[unlikely]] {
        detail::mpi_failure(error_code, operation, where);
    }
}

/// @brief Reports a fatal error and aborts MPI_COMM_WORLD while MPI is active.
/// Otherwise exits locally; no stack unwinding or MPI_Finalize is performed.
[[noreturn]] void fatal(
    std::string_view message,
    std::source_location where = std::source_location::current()) noexcept;

/// @brief Terminates the job when a required condition is not satisfied.
inline void require(
    bool condition,
    std::string_view message,
    std::source_location where = std::source_location::current()) noexcept
{
    if (!condition) [[unlikely]] {
        fatal(message, where);
    }
}

/// @brief Converts escaping exceptions into a fatal job error.
template<class Function>
decltype(auto) guarded(
    Function&& function,
    std::source_location where = std::source_location::current()) noexcept
{
    try {
        return std::forward<Function>(function)();
    } catch (const std::exception& error) {
        fatal(error.what(), where);
    } catch (...) {
        fatal("Unhandled non-standard exception", where);
    }
}

/// @brief Requires an exact number of entries.
template<class Array>
inline void require_size(
    const Array& array,
    std::size_t expected,
    const char* message = "Unexpected array size",
    std::source_location where = std::source_location::current())
{
    require(array.size() == expected, message, where);
}

/// @brief Requires storage for at least the requested number of entries.
template<class Array>
inline void require_min_size(
    const Array& array,
    std::size_t minimum,
    const char* message = "Insufficient array size",
    std::source_location where = std::source_location::current())
{
    require(array.size() >= minimum, message, where);
}

/// @brief Checks host CSR offsets; does not validate stored indices.
template<class Offsets, class Values, bool CheckMonotonicity = false>
inline void require_csr(
    const Offsets& offsets,
    const Values& values,
    std::size_t rows,
    const char* message = "Invalid CSR offsets",
    std::source_location where = std::source_location::current())
{
    require(rows < std::numeric_limits<std::size_t>::max(), message, where);
    require(offsets.size() == rows + 1, message, where);

    // Size is checked before reading either endpoint.
    require(offsets[0] == 0 && offsets[rows] == values.size(), message, where);

    if constexpr (CheckMonotonicity)
        for (std::size_t row = 0; row < rows; ++row)
            require(offsets[row] <= offsets[row + 1], message, where);
}

/// @brief Checks host offsets;
template<class Offsets, bool CheckMonotonicity = false>
inline void require_offsets(
    const Offsets& offsets,
    std::size_t peers, 
    std::size_t entries,
    const char* message = "Invalid CSR offsets",
    std::source_location where = std::source_location::current())
{
    if (peers == 0 && offsets.empty()) {
        mpi::require(entries == 0, "Halo: map entries without peers");
        return;
    }
    mpi::require(offsets.size() == peers + 1 && offsets.front() == 0 &&
                    offsets.back() == entries, message, where);

    if constexpr (CheckMonotonicity)
        mpi::require(std::is_sorted(offsets.begin(), offsets.end()), "Offsets monotonisity failed", where);
}

/// @brief Returns a product after checking size_t overflow.
[[nodiscard]] inline std::size_t checked_product(
    std::size_t a,
    std::size_t b,
    const char* message = "Size product overflows size_t",
    std::source_location where = std::source_location::current())
{
    require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b, message, where);
    return a * b;
}


} // namespace krona::mpi
