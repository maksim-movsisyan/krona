#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>

namespace krona::boundary::physical {

/// @brief Failure categories recorded independently across all faces in a batch.
enum class Error : int {
    Parameters,      ///< An invalid sampled condition parameter or direction.
    Interior,        ///< Invalid primitive input supplied by the reader.
    State,           ///< Invalid face or exterior state produced by the closure.
    Relation,        ///< Invalid explicit correction or affine temperature relation.
    Output,          ///< A state value outside the destination field's representable range.
    Characteristics, ///< Incompatible inlet, outlet or backflow data.
    Unsupported,     ///< A missing capability or law needed by the assigned condition.
    Count            ///< Number of device flag slots.
};

namespace detail {
    
/// @brief Default error handler for evaluating a condition directly inside a kernel.
struct AbortErrors {
    KOKKOS_INLINE_FUNCTION void fail(Error, const char* message) const { Kokkos::abort(message); }
    KOKKOS_INLINE_FUNCTION bool failed() const { return false; }
};

} // namespace detail

/// @brief Records a failure in both the calling thread and the batch shared by all faces.
/// @tparam Flags Writable device view indexed by Error.
template<class Flags>
struct DeviceErrors {
    Flags flags;
    bool* local_failed; ///< Address of a local bool that lives for the current face evaluation.

    /// @brief Signals that this face failed and atomically records the category for the host.
    /// @param error Category to mark for the diagnostic batch.
    KOKKOS_INLINE_FUNCTION void fail(Error error, const char*) const {
        *local_failed = true;
        Kokkos::atomic_exchange(&flags(static_cast<int>(error)), 1);
    }
    /// @brief Reports whether this face has failed so far.
    KOKKOS_INLINE_FUNCTION bool failed() const { return *local_failed; }
};

/// @brief Reusable local diagnostics for one or more physical-boundary launches.
/// @tparam Exec Kokkos execution space that runs the launches and transfers their flags.
/// @details Call begin_async(), submit work on the same execution instance, then call check().
template<class Exec = execution::ExecutionSpace>
class Status {
public:
    /// @brief View copied into kernels to collect error flags.
    using flags_type = Kokkos::View<int*, typename Exec::memory_space>;

    /// @brief Allocates the device flags and host mirror for reusing across batches.
    /// @param exec Execution instance used by all submissions in the batch.
    explicit Status(Exec exec = {}) : exec_(exec) {
        execution::require_kokkos();
        mpi::guarded([&] {
            flags_ = flags_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                "boundary/physical/status"), static_cast<int>(Error::Count));
            host_ = Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_);
        });
    }
    Status(const Status&) = delete;
    Status& operator=(const Status&) = delete;
    /// @brief Waits for an open batch before releasing its buffers. Use check() to report failures.
    ~Status() { if (pending_) mpi::guarded([&] { exec_.fence("Physical boundary: pending status work completed"); }); }

    /// @brief Opens a batch and queues the reset of its device flags.
    void begin_async() {
        execution::require_kokkos();
        mpi::require(!pending_, "Physical boundary: status pending; call check()");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0); });
    }
    /// @brief Returns the flag view for a kernel in the current batch.
    /// @param exec Execution instance on which the kernel will run.
    [[nodiscard]] flags_type device_flags(const Exec& exec) const {
        mpi::require(pending_, "Physical boundary: call status.begin_async() first");
        mpi::require(exec == exec_, "Physical boundary: status execution instance mismatch");
        return flags_;
    }
    /// @brief Reports whether begin_async() opened a batch that check() has yet to close.
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    /// @brief Local blocking completion and fatal diagnostics; no MPI collective.
    void check() {
        mpi::require(pending_, "Physical boundary: no pending status batch");
        mpi::guarded([&] {
            Kokkos::deep_copy(exec_, host_, flags_);
            exec_.fence("Physical boundary: diagnostics ready");
        });
        pending_ = false;
        using E = Error;
        require_clear(E::Parameters, "Physical boundary: invalid sampled parameter or direction");
        require_clear(E::Interior, "Physical boundary: invalid interior primitive state");
        require_clear(E::Characteristics, "Physical boundary: incompatible characteristic or backflow data");
        require_clear(E::Unsupported, "Physical boundary: unsupported condition/model combination");
        require_clear(E::Relation, "Physical boundary: invalid temperature relation or correction");
        require_clear(E::State, "Physical boundary: invalid face or exterior primitive state");
        require_clear(E::Output, "Physical boundary: unrepresentable primitive output");
    }
private:
    /// @brief Reports a failure if any face in the completed batch marked this category.
    void require_clear(Error e, const char* message) const
    { mpi::require(host_(static_cast<int>(e)) == 0, message); }

    Exec exec_;
    flags_type flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::boundary::physical
