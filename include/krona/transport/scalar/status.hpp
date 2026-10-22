#pragma once

#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>

#include <array>

namespace krona::transport::scalar {
namespace detail {
enum AssemblyFlag : int { BadSource, BadDiffusion, BadBoundary, BadCorrection, HasAnchor, BadSystem, StatusCount };
template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, AssemblyFlag index)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(index)), 1); }

template<class View, class Real, class Flags>
KOKKOS_INLINE_FUNCTION void add_checked(View view, std::size_t i, Real increment, Flags flags)
{
    using Value = typename View::non_const_value_type;
    const Value value = static_cast<Value>(increment);
    const Value result = view(i)+value;
    if (!Kokkos::isfinite(increment) || !Kokkos::isfinite(value) || !Kokkos::isfinite(result)) {
        flag(flags, BadSystem); return;
    }
    view(i) = result;
} 
template<class View, class Real, class Flags>
KOKKOS_INLINE_FUNCTION void atomic_add_checked(View view, std::size_t i, Real increment, Flags flags)
{
    using Value = typename View::non_const_value_type;
    const Value value = static_cast<Value>(increment);
    if (!Kokkos::isfinite(increment) || !Kokkos::isfinite(value)) { flag(flags, BadSystem); return; }
    const Value previous = Kokkos::atomic_fetch_add(&view(i), value);
    if (!Kokkos::isfinite(previous+value)) flag(flags, BadSystem);
}
} // namespace detail

struct AssemblyReport { bool has_diffusion_anchor = false; };

/// @brief One reusable diagnostic buffer for an entire additive assembly batch.
/// begin_async/check bracket contributions on one execution instance. check() does not commit algebra wrappers.
template<class Exec = execution::ExecutionSpace>
class AssemblyStatus {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using flags_type = Kokkos::View<int*, memory_space>;

    explicit AssemblyStatus(Exec exec = {}) : exec_(exec),
        flags_(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "transport/status"), detail::StatusCount),
        host_(Kokkos::create_mirror(Kokkos::WithoutInitializing, flags_)) {}
    AssemblyStatus(const AssemblyStatus&) = delete;
    AssemblyStatus& operator=(const AssemblyStatus&) = delete;
    ~AssemblyStatus() { if (pending_) exec_.fence("Transport: pending assembly completed"); }

    void begin_async()
    {
        execution::require_kokkos();
        mpi::require(!pending_, "Scalar transport: assembly pending; call check()");
        pending_ = true;
        Kokkos::deep_copy(exec_, flags_, 0);
    }
    [[nodiscard]] flags_type device_flags(const Exec& exec) const
    {
        mpi::require(pending_, "Scalar transport: call begin_async() before adding contributions");
        mpi::require(exec == exec_, "Scalar transport: assembly execution instance mismatch");
        return flags_;
    }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
    [[nodiscard]] bool pending() const noexcept { return pending_; }

    /// @brief Explicit blocking collective diagnostics; no anchor/whole-system diagonal requirement here.
    [[nodiscard]] AssemblyReport check(mpi::CommView comm)
    {
        mpi::require(pending_, "Scalar transport: no pending assembly");
        Kokkos::deep_copy(exec_, host_, flags_);
        exec_.fence("Transport: assembly diagnostics ready");
        pending_ = false;
        std::array<int, detail::StatusCount> global{};
        mpi::check(MPI_Allreduce(host_.data(), global.data(), detail::StatusCount, MPI_INT, MPI_MAX, comm.native()),
                   "MPI_Allreduce(transport status)");
        mpi::require(global[detail::BadSource] == 0, "Scalar transport: source density must be finite");
        mpi::require(global[detail::BadDiffusion] == 0, "Scalar diffusion: coefficient must be finite and positive");
        mpi::require(global[detail::BadBoundary] == 0, "Scalar diffusion: invalid affine boundary flux");
        mpi::require(global[detail::BadCorrection] == 0,
            "Scalar diffusion: invalid correction data or correction-dependent boundary slope");
        mpi::require(global[detail::BadSystem] == 0, "Scalar transport: nonfinite or unrepresentable contribution/target");
        return {global[detail::HasAnchor] != 0};
    }
private:
    Exec exec_;
    flags_type flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_;
    bool pending_ = false;
};

} // namespace krona::transport::scalar
