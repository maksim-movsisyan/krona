#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>

namespace krona::boundary::physical {
enum class Error : int { Parameters, Interior, State, Relation, Output, Characteristics, Unsupported, Count };
namespace detail {
struct AbortErrors {
    KOKKOS_INLINE_FUNCTION void fail(Error, const char* message) const { Kokkos::abort(message); }
    KOKKOS_INLINE_FUNCTION bool failed() const { return false; }
};
} // namespace detail

/// @brief Construct inside a device thread; local_failed points to that thread's live local bool.
template<class Flags>
struct DeviceErrors {
    Flags flags;
    bool* local_failed;
    KOKKOS_INLINE_FUNCTION void fail(Error error, const char*) const {
        *local_failed = true;
        Kokkos::atomic_exchange(&flags(static_cast<int>(error)), 1);
    }
    KOKKOS_INLINE_FUNCTION bool failed() const { return *local_failed; }
};

/// @brief Reusable local diagnostics; begin/check bracket a batch on one execution instance.
template<class Exec = execution::ExecutionSpace>
class Status {
public:
    using flags_type = Kokkos::View<int*, typename Exec::memory_space>;
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
    ~Status() { if (pending_) mpi::guarded([&] { exec_.fence("Physical boundary: pending status work completed"); }); }
    void begin_async() {
        execution::require_kokkos();
        mpi::require(!pending_, "Physical boundary: status pending; call check()");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0); });
    }
    [[nodiscard]] flags_type device_flags(const Exec& exec) const {
        mpi::require(pending_, "Physical boundary: call status.begin_async() first");
        mpi::require(exec == exec_, "Physical boundary: status execution instance mismatch");
        return flags_;
    }
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
    void require_clear(Error e, const char* message) const
    { mpi::require(host_(static_cast<int>(e)) == 0, message); }
    Exec exec_;
    flags_type flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::boundary::physical
