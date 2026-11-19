#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>

namespace krona::boundary::scalar {

namespace detail {

/// @brief Device flag slots read by Status after a batch.
enum Error : int {
    BadRelation, ///< A sampled parameter, correction or relation coefficient is invalid.
    BadOutput,   ///< A computed face value is nonfinite or overflows the destination type.
    ErrorCount   ///< Number of allocated flag slots.
};

/// @brief Marks an error category once or more, safely across concurrent boundary faces.
/// @tparam Flags Device view indexed by Error.
/// @param flags Shared error flags for the current batch.
/// @param error Category to mark as present.
template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, Error error)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(error)), 1); }

} // namespace detail

/// @brief Collects local device errors across one or more boundary submissions.
/// @tparam Exec Kokkos execution space that runs the submissions and transfers the flags.
/// @details Call begin_async(), submit work on the same execution instance, then call check().
/// Multiple managers can share a batch. The status records which error categories occurred.
template<class Exec = execution::ExecutionSpace>
class Status {
public:
    /// @brief Device view copied into each kernel that reports an error.
    using flags_type = Kokkos::View<int*, typename Exec::memory_space>;

    /// @brief Allocates the reusable diagnostic buffers.
    /// @param exec Execution instance used by the manager and by the diagnostic copies.
    explicit Status(Exec exec = {}) : exec_(exec) {
        execution::require_kokkos();
        mpi::guarded([&] {
            flags_ = flags_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                                  "boundary/scalar/status"), detail::ErrorCount);
            host_ = Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_);
        });
    }
    
    // One object owns the diagnostic buffers and the state of their current batch.
    Status(const Status&) = delete;
    Status& operator=(const Status&) = delete;

    /// @brief Waits for an open batch before releasing its buffers. Use check() to read its errors.
    ~Status() {
        if (pending_) mpi::guarded([&] { exec_.fence("Boundary: pending status work completed"); });
    }
    /// @brief Opens a batch and queues the flag reset on its execution instance.
    void begin_async() {
        execution::require_kokkos();
        mpi::require(!pending_, "Scalar boundary: status pending; call check()");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0); });
    }
    /// @brief Returns the writable flag view for a submission in the open batch.
    /// @param exec Execution instance on which the submission will run.
    [[nodiscard]] flags_type device_flags(const Exec& exec) const {
        mpi::require(pending_, "Scalar boundary: call status.begin_async() first");
        mpi::require(exec == exec_, "Scalar boundary: status execution instance mismatch");
        return flags_;
    }
    /// @brief Reports whether begin_async() opened a batch that check() has yet to close.
    [[nodiscard]] bool pending() const noexcept { return pending_; }

    /// @brief Waits on the execution instance, checks the local flags and reports failures via mpi::require.
    void check() {
        mpi::require(pending_, "Scalar boundary: no pending status batch");
        mpi::guarded([&] {
            Kokkos::deep_copy(exec_, host_, flags_);
            exec_.fence("Boundary: scalar diagnostics ready");
        });
        pending_ = false;
        mpi::require(host_(detail::BadRelation) == 0,
                     "Scalar boundary: nonfinite parameter, correction or affine relation");
        mpi::require(host_(detail::BadOutput) == 0,
                     "Scalar boundary: nonfinite or unrepresentable boundary value");
    }
private:
    Exec exec_;
    flags_type flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::boundary::scalar
