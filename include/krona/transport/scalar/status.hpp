#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>

namespace krona::transport::scalar {
namespace detail {

/// @brief Slots set by kernels during a scalar assembly batch.
enum AssemblyFlag : int {
    BadSource,         ///< A source density is nonfinite.
    BadDiffusion,      ///< A diffusion coefficient or inner face conductance is unusable.
    BadBoundary,       ///< A diffusive boundary flux relation is invalid.
    BadCorrection,     ///< A correction is invalid or changes a boundary matrix slope.
    HasAnchor,         ///< A diffusive boundary flux has a positive owner coefficient.
    BadSystem,         ///< A contribution or target entry is invalid or unrepresentable.
    BadMassFlux,       ///< A convective face mass flux is nonfinite.
    BadInflow,         ///< A required boundary inflow value is missing or nonfinite.
    BadReconstruction, ///< A reconstructed increment is invalid.
    BadTemporalMass,   ///< A cell mass used in a time term is nonfinite or nonpositive.
    BadTemporalState,  ///< A time-layer value is nonfinite.
    StatusCount        ///< Number of flag slots to allocate.
};

/// @brief Marks a category without racing with other threads reporting it.
/// @tparam Flags Writable device view indexed by AssemblyFlag.
/// @param flags Shared flags for the open batch.
/// @param index Category to mark.
template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, AssemblyFlag index)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(index)), 1); }

/// @brief Adds an increment to an entry owned by this thread.
/// An invalid increment, conversion or sum sets BadSystem and leaves the entry as it was.
/// @tparam View Writable rank-one view of output entries.
/// @tparam Real Arithmetic type of the incoming increment.
/// @tparam Flags Writable device view of assembly flags.
/// @param view Output storage.
/// @param i Entry to update.
/// @param increment Contribution to add.
/// @param flags Shared flags for the open batch.
template<class View, class Real, class Flags>
KOKKOS_INLINE_FUNCTION void add_checked(View view, std::size_t i, Real increment, Flags flags) {
    using Value = typename View::non_const_value_type;
    const Value value = static_cast<Value>(increment);
    const Value result = view(i) + value;
    if (!Kokkos::isfinite(increment) || !Kokkos::isfinite(value) || !Kokkos::isfinite(result)) {
        flag(flags, BadSystem);
        return;
    }
    view(i) = result;
}

/// @brief Adds an increment atomically when several threads can update the same entry.
/// The returned old value lets us check the exact addition performed by this thread.
/// A nonfinite sum sets BadSystem after the atomic update.
/// @tparam View Writable rank-one view of output entries.
/// @tparam Real Arithmetic type of the incoming increment.
/// @tparam Flags Writable device view of assembly flags.
/// @param view Output storage.
/// @param i Entry to update.
/// @param increment Contribution to add.
/// @param flags Shared flags for the open batch.
template<class View, class Real, class Flags>
KOKKOS_INLINE_FUNCTION void atomic_add_checked(View view, std::size_t i, Real increment, Flags flags) {
    using Value = typename View::non_const_value_type;
    const Value value = static_cast<Value>(increment);
    if (!Kokkos::isfinite(increment) || !Kokkos::isfinite(value)) {
        flag(flags, BadSystem);
        return;
    }
    const Value previous = Kokkos::atomic_fetch_add(&view(i), value);
    if (!Kokkos::isfinite(previous + value)) flag(flags, BadSystem);
}

} // namespace detail

/// @brief Information from the completed batch on this rank.
struct AssemblyReport {
    bool has_local_diffusion_anchor = false; ///< At least one local diffusive boundary flux had a positive owner coefficient.
};

/// @brief Collects device diagnostics across one or more contributions on this rank.
/// Call begin_async(), submit all contributions on the same instance, then call check().
/// @tparam Exec Kokkos execution space used to assemble and copy the flags.
template<class Exec = execution::ExecutionSpace>
class AssemblyStatus {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using flags_type = Kokkos::View<int*, memory_space>; ///< Writable flag buffer passed to kernels.

    /// @brief Allocates reusable flag storage and its host mirror.
    /// @param exec Execution instance shared with the operators that submit to this batch.
    explicit AssemblyStatus(Exec exec = {}) : exec_(exec) {
        execution::require_kokkos();
        mpi::guarded([&] {
            flags_ = flags_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "transport/status"),
                                detail::StatusCount);
            host_ = Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_);
        });
    }

    AssemblyStatus(const AssemblyStatus&) = delete;
    AssemblyStatus& operator=(const AssemblyStatus&) = delete;

    /// @brief Waits for a pending batch so queued kernels can finish using its flag storage.
    ~AssemblyStatus() {
        if (pending_) mpi::guarded([&] { exec_.fence("Transport: pending assembly completed"); });
    }

    /// @brief Opens a batch and queues a reset of the flags on its execution instance.
    void begin_async() {
        execution::require_kokkos();
        mpi::require(!pending_, "Scalar transport: assembly pending; call check()");
        pending_ = true;
        mpi::guarded([&] { Kokkos::deep_copy(exec_, flags_, 0); });
    }

    /// @brief Returns the writable flags for a submission to the open batch.
    /// @param exec Execution instance on which the submission will run.
    [[nodiscard]] flags_type device_flags(const Exec& exec) const {
        mpi::require(pending_, "Scalar transport: call begin_async() before adding contributions");
        mpi::require(exec == exec_, "Scalar transport: assembly execution instance mismatch");
        return flags_;
    }

    /// @brief Returns the execution instance used for resets and diagnostic copies.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }

    /// @brief Reports whether there is a batch waiting for check().
    [[nodiscard]] bool pending() const noexcept { return pending_; }

    /// @brief Waits for the local batch and reports kernel errors through mpi::require.
    /// @return Local diffusion information. The caller can reduce it explicitly for a global check.
    [[nodiscard]] AssemblyReport check() {
        mpi::require(pending_, "Scalar transport: no pending assembly");
        mpi::guarded([&] {
            Kokkos::deep_copy(exec_, host_, flags_);
            exec_.fence("Transport: assembly diagnostics ready");
        });
        pending_ = false;

        mpi::require(host_(detail::BadSource) == 0, "Scalar transport: source density must be finite");
        mpi::require(host_(detail::BadDiffusion) == 0, "Scalar diffusion: coefficient must be finite and positive");
        mpi::require(host_(detail::BadBoundary) == 0, "Scalar diffusion: invalid affine boundary flux");
        mpi::require(host_(detail::BadCorrection) == 0,
            "Scalar diffusion: invalid correction data or correction-dependent boundary slope");
        mpi::require(host_(detail::BadMassFlux) == 0, "Scalar convection: mass flux must be finite");
        mpi::require(host_(detail::BadInflow) == 0, "Scalar convection: missing or nonfinite inflow value");
        mpi::require(host_(detail::BadReconstruction) == 0, "Scalar convection: invalid reconstructed increment");
        mpi::require(host_(detail::BadTemporalMass) == 0, "Scalar transient: cell mass must be finite and positive");
        mpi::require(host_(detail::BadTemporalState) == 0, "Scalar transient: time-layer values must be finite");
        mpi::require(host_(detail::BadSystem) == 0, "Scalar transport: nonfinite or unrepresentable contribution/target");
        return {host_(detail::HasAnchor) != 0};
    }

private:
    Exec exec_;
    flags_type flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::transport::scalar
