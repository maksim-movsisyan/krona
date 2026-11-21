#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>

namespace krona::transport::euler {

namespace detail {

/// @brief Entries written by the Euler kernels during an assembly batch.
enum Diagnostic : int {
    BadState,                ///< An input cell has an inadmissible thermodynamic state.
    BadBoundary,             ///< The boundary closure produced an inadmissible exterior state.
    BadFlux,                 ///< The Riemann solver could not provide a usable flux.
    BadOutput,               ///< A residual contribution or sum cannot be stored.
    BadJacobian,             ///< A derivative, native block index or coefficient is invalid.
    ReconstructionFallbacks, ///< Number of reconstructed sides replaced by their cell state.
    RiemannFallbacks,        ///< Number of flux evaluations accepted with the solver's fallback.
    DiagnosticCount
};

/// @brief Sets an error flag that can be shared by all work items in the batch.
/// @tparam Flags Writable device view indexed by Diagnostic.
template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, Diagnostic index) {
    Kokkos::atomic_exchange(&flags(static_cast<int>(index)), 1ULL);
}

/// @brief Adds one fallback evaluation to the batch counter.
/// @tparam Flags Writable device view indexed by Diagnostic.
template<class Flags>
KOKKOS_INLINE_FUNCTION void count(Flags flags, Diagnostic index) {
    Kokkos::atomic_add(&flags(static_cast<int>(index)), 1ULL);
}

} // namespace detail

/// @brief Counts fallback evaluations on this rank since begin_async().
/// An inner face between two owned cells can be evaluated once for each row.
struct AssemblyReport {
    unsigned long long reconstruction_fallbacks = 0;
    unsigned long long riemann_fallbacks = 0;
};

/// @brief Collects Euler errors and fallback counts for work queued on one execution instance.
/// @tparam Exec Kokkos execution space used by the operator and for diagnostic copies.
template<class Exec = execution::ExecutionSpace>
class AssemblyStatus {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using flags_type = Kokkos::View<unsigned long long*, memory_space>;

    /// @brief Allocates the diagnostics and the host mirror used to read them.
    /// @param exec Execution instance on which the Euler kernels will run.
    explicit AssemblyStatus(Exec exec = {}) : exec_(exec) {
        execution::require_kokkos();
        mpi::guarded([&] {
            flags_ = flags_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "euler/status"),
                                detail::DiagnosticCount);
            host_ = Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_);
        });
    }

    AssemblyStatus(const AssemblyStatus&) = delete;
    AssemblyStatus& operator=(const AssemblyStatus&) = delete;

    /// @brief Waits for a pending batch before releasing its buffers. Call check() to report errors.
    ~AssemblyStatus() {
        if (pending_)
            mpi::guarded([&] { exec_.fence("Euler: pending diagnostics complete"); });
    }

    /// @brief Opens a batch and queues the diagnostic reset.
    void begin_async() {
        execution::require_kokkos();
        mpi::require(!pending_, "Euler: diagnostics already pending");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0ULL); });
    }

    /// @brief Returns the writable diagnostics for a launch in the open batch.
    /// @param exec Instance on which the launch will run.
    [[nodiscard]] flags_type device_flags(const Exec& exec) const {
        mpi::require(pending_, "Euler: call begin_async before contributions");
        mpi::require(exec == exec_, "Euler: execution instance mismatch");
        return flags_;
    }

    /// @brief Waits for this rank's batch, checks its flags and returns its fallback counts.
    /// The caller can reduce the counts explicitly when a global report is needed.
    /// An error is fatal to the MPI job through mpi::require.
    [[nodiscard]] AssemblyReport check() {
        mpi::require(pending_, "Euler: no pending diagnostics");
        mpi::guarded([&] {
            Kokkos::deep_copy(exec_, host_, flags_);
            exec_.fence("Euler: diagnostics ready");
        });
        pending_ = false;
        mpi::require(host_(detail::BadState) == 0, "Euler: invalid cell thermodynamic state");
        mpi::require(host_(detail::BadBoundary) == 0, "Euler: invalid exterior boundary state");
        mpi::require(host_(detail::BadFlux) == 0, "Euler: Riemann solver rejected a face state or fan");
        mpi::require(host_(detail::BadOutput) == 0, "Euler: nonfinite or unrepresentable residual");
        mpi::require(host_(detail::BadJacobian) == 0, "Euler: invalid Jacobian derivative, block entry or coefficient");
        return {host_(detail::ReconstructionFallbacks), host_(detail::RiemannFallbacks)};
    }

    /// @brief Reports whether a batch is still open.
    [[nodiscard]] bool pending() const noexcept { return pending_; }

    /// @brief Returns the instance used for resets, kernel submissions and diagnostic copies.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }

private:
    Exec exec_;
    flags_type flags_;
    Kokkos::View<unsigned long long*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::transport::euler
