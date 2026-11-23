#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>

namespace krona::transport::viscous {

namespace detail {

enum class Error { InnerFace, BoundaryFace, Residual, Jacobian, Count };

template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, Error error) {
    Kokkos::atomic_exchange(&flags(static_cast<int>(error)), 1);
}

} // namespace detail

/// @brief Records errors from a group of viscous submissions on the same execution instance.
/// @tparam Exec Execution space used for submissions and diagnostic copies.
template<class Exec = execution::ExecutionSpace>
class AssemblyStatus {
public:
    using Flags = Kokkos::View<int*, typename Exec::memory_space>;

    explicit AssemblyStatus(Exec exec = {}) : exec_(exec) {
        execution::require_kokkos();
        mpi::guarded([&] {
            flags_ = Flags(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "viscous/status"),
                           static_cast<int>(detail::Error::Count));
            host_ = Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_);
        });
    }

    AssemblyStatus(const AssemblyStatus&) = delete;
    AssemblyStatus& operator=(const AssemblyStatus&) = delete;

    ~AssemblyStatus() {
        if (pending_) mpi::guarded([&] { exec_.fence("Viscous: pending diagnostics complete"); });
    }

    /// @brief Queues a diagnostic reset and opens the batch.
    void begin_async() {
        mpi::require(!pending_, "Viscous: diagnostics already pending");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0); });
    }

    [[nodiscard]] Flags device_flags(const Exec& exec) const {
        mpi::require(pending_ && exec == exec_, "Viscous: expected an open batch on the operator execution instance");
        return flags_;
    }

    /// @brief Waits for this rank's batch and reports invalid contributions through MPI error handling.
    void check() {
        mpi::require(pending_, "Viscous: no pending diagnostics");
        mpi::guarded([&] {
            Kokkos::deep_copy(exec_, host_, flags_);
            exec_.fence("Viscous: diagnostics ready");
        });
        pending_ = false;
        mpi::require(!host_(static_cast<int>(detail::Error::InnerFace)), "Viscous: invalid inner face input or flux");
        mpi::require(!host_(static_cast<int>(detail::Error::BoundaryFace)), "Viscous: invalid boundary input or flux");
        mpi::require(!host_(static_cast<int>(detail::Error::Residual)), "Viscous: residual overflow");
        mpi::require(!host_(static_cast<int>(detail::Error::Jacobian)), "Viscous: invalid Jacobian or matrix coefficient");
    }

private:
    Exec exec_;
    Flags flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::transport::viscous
