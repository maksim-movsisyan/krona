#pragma once

#include <krona/mpi/communicator.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <new>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

namespace krona::mpi {

class LocalStatus final {
public:
    static constexpr std::size_t message_capacity = 512;

    LocalStatus() noexcept = default;

    [[nodiscard]] bool ok() const noexcept { return !failed_; }
    [[nodiscard]] std::string_view message() const noexcept
    {
        return {message_.data(), message_size_};
    }
    [[nodiscard]] std::source_location location() const noexcept { return where_; }

    // Record the first failure; copy and truncate its text without allocation.
    void fail(
        std::string_view message,
        std::source_location where = std::source_location::current()) noexcept;

    [[nodiscard]] static LocalStatus failure(
        std::string_view message,
        std::source_location where = std::source_location::current()) noexcept;

    // Capture local work returning void or LocalStatus; no MPI, including destructors.
    template<class Work>
        requires std::invocable<Work> &&
            (std::same_as<std::invoke_result_t<Work>, void> ||
             std::same_as<std::remove_cvref_t<std::invoke_result_t<Work>>, LocalStatus>)
    [[nodiscard]] static LocalStatus capture(
        Work&& work,
        std::source_location where = std::source_location::current()) noexcept
    {
        try {
            if constexpr (std::same_as<std::invoke_result_t<Work>, void>) {
                std::invoke(std::forward<Work>(work));
                return {};
            } else {
                return std::invoke(std::forward<Work>(work));
            }
        } catch (const std::bad_alloc&) {
            return failure("Memory allocation failed", where);
        } catch (const std::exception& error) {
            return failure(error.what(), where);
        } catch (...) {
            return failure("Unknown exception during local preparation", where);
        }
    }

private:
    std::array<char, message_capacity> message_{};
    std::source_location where_{};
    std::size_t message_size_ = 0;
    bool failed_ = false;
};

// Consensus result; its borrowed communicator must outlive require_success().
class CollectiveStatus final {
public:
    static constexpr std::size_t diagnostic_capacity = 1024;

    [[nodiscard]] bool ok() const noexcept { return first_failed_rank_ < 0; }
    [[nodiscard]] int first_failed_rank() const noexcept { return first_failed_rank_; }
    [[nodiscard]] std::string_view diagnostic() const noexcept
    {
        return std::string_view{diagnostic_.data()};
    }

    // Call on all participants: return on success; otherwise print once and abort.
    void require_success(
        std::string_view stage = "Local preparation",
        std::source_location where = std::source_location::current()) const noexcept;

private:
    friend CollectiveStatus synchronize_status(
        CommView, const LocalStatus&, std::source_location) noexcept;

    CollectiveStatus() noexcept = default;

    CommView comm_;
    std::array<char, diagnostic_capacity> diagnostic_{};
    int first_failed_rank_ = -1;
};

// Collective on an intracommunicator: one Allreduce, plus a broadcast on failure.
// Every rank must reach this call in matching order, including failed ranks.
[[nodiscard]] CollectiveStatus synchronize_status(
    CommView comm,
    const LocalStatus& local,
    std::source_location where = std::source_location::current()) noexcept;

// Agree collectively, then enforce success. Failed preparation is not rolled back.
inline void require_collective_success(
    CommView comm,
    const LocalStatus& local,
    std::string_view stage = "Local preparation",
    std::source_location where = std::source_location::current()) noexcept
{
    synchronize_status(comm, local, where).require_success(stage, where);
}

} // namespace krona::mpi
