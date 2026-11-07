#pragma once
#include <krona/execution/types.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <array>

namespace krona::transport::euler {
namespace detail {
enum Diagnostic : int { BadState,BadBoundary,BadFlux,BadOutput,ReconstructionFallbacks,RiemannFallbacks,DiagnosticCount };
template<class Flags>KOKKOS_INLINE_FUNCTION void flag(Flags flags,Diagnostic index)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(index)),1ULL); }
template<class Flags>KOKKOS_INLINE_FUNCTION void count(Flags flags,Diagnostic index)
{ Kokkos::atomic_add(&flags(static_cast<int>(index)),1ULL); }
}
struct AssemblyReport { unsigned long long reconstruction_fallbacks=0,riemann_fallbacks=0; };

/// @brief begin_async/check bracket a batch on one exec. Counters count evaluations, not unique faces.
template<class Exec=execution::ExecutionSpace>
class AssemblyStatus {
public:
    using execution_space=Exec;
    using memory_space=typename Exec::memory_space;
    using flags_type=Kokkos::View<unsigned long long*,memory_space>;
    explicit AssemblyStatus(Exec exec={}):exec_(exec),
        flags_(Kokkos::view_alloc(exec,Kokkos::WithoutInitializing,"euler/status"),detail::DiagnosticCount),
        host_(Kokkos::create_mirror(Kokkos::WithoutInitializing,flags_)) {}
    AssemblyStatus(const AssemblyStatus&)=delete;
    AssemblyStatus& operator=(const AssemblyStatus&)=delete;
    ~AssemblyStatus(){if(pending_)exec_.fence("Euler: pending diagnostics complete");}
    void begin_async()
    {
        execution::require_kokkos();
        mpi::require(!pending_,"Euler: diagnostics already pending");
        pending_=true;Kokkos::deep_copy(exec_,flags_,0ULL);
    }
    flags_type device_flags(const Exec& exec) const
    {
        mpi::require(pending_,"Euler: call begin_async before contributions");
        mpi::require(exec==exec_,"Euler: execution instance mismatch");
        return flags_;
    }
    [[nodiscard]] AssemblyReport check(mpi::CommView comm)
    {
        mpi::require(pending_,"Euler: no pending diagnostics");
        Kokkos::deep_copy(exec_,host_,flags_);exec_.fence("Euler: diagnostics ready");pending_=false;
        std::array<unsigned long long,detail::DiagnosticCount> global{};
        mpi::check(MPI_Allreduce(host_.data(),global.data(),detail::DiagnosticCount,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm.native()),"MPI_Allreduce(Euler status)");
        mpi::require(global[detail::BadState]==0,"Euler: invalid cell thermodynamic state");
        mpi::require(global[detail::BadBoundary]==0,"Euler: invalid exterior boundary state");
        mpi::require(global[detail::BadFlux]==0,"Euler: Riemann solver rejected a face state or fan");
        mpi::require(global[detail::BadOutput]==0,"Euler: nonfinite or unrepresentable residual");
        return {global[detail::ReconstructionFallbacks],global[detail::RiemannFallbacks]};
    }
    [[nodiscard]] bool pending()const noexcept{return pending_;}
    [[nodiscard]] const Exec& execution_space_instance()const noexcept{return exec_;}
private:
    Exec exec_;
    flags_type flags_;
    Kokkos::View<unsigned long long*,Kokkos::HostSpace> host_;
    bool pending_=false;
};
} // namespace krona::transport::euler
