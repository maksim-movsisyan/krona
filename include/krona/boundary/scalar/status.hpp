#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>

namespace krona::boundary::scalar {
namespace detail {
enum Error : int { BadRelation, BadOutput, ErrorCount };
template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, Error error)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(error)), 1); }
} // namespace detail

/// @brief Reusable local diagnostics for a batch on one execution instance. No MPI collectives.
template<class Exec = execution::ExecutionSpace>
class Status {
public:
    using flags_type = Kokkos::View<int*, typename Exec::memory_space>;
    explicit Status(Exec exec = {}) : exec_(exec)
    {
        execution::require_kokkos();
        mpi::guarded([&] {
            flags_ = flags_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                                  "boundary/scalar/status"), detail::ErrorCount);
            host_ = Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_);
        });
    }
    Status(const Status&) = delete;
    Status& operator=(const Status&) = delete;
    ~Status()
    {
        if (pending_) mpi::guarded([&] { exec_.fence("Boundary: pending status work completed"); });
    }
    void begin_async()
    {
        execution::require_kokkos();
        mpi::require(!pending_, "Scalar boundary: status pending; call check()");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0); });
    }
    [[nodiscard]] flags_type device_flags(const Exec& exec) const
    {
        mpi::require(pending_, "Scalar boundary: call status.begin_async() first");
        mpi::require(exec == exec_, "Scalar boundary: status execution instance mismatch");
        return flags_;
    }
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    /// @brief Blocks on this execution instance; any local failure terminates via mpi::require.
    void check()
    {
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
