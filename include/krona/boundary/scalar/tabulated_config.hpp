#pragma once

#include <krona/boundary/detail/tabulated_values.hpp>
#include <krona/boundary/scalar/manager.hpp>

namespace krona::boundary::scalar {

/// @brief Scalar manager with providers for all three tabulated condition kinds.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
using TabulatedManager = Manager<Exec, Real,
    FixedValue<Real, boundary::detail::TabulatedPatchValues<Real, typename Exec::memory_space>>,
    NormalGradient<Real, boundary::detail::TabulatedPatchValues<Real, typename Exec::memory_space>>,
    FixedFlux<Real, boundary::detail::TabulatedPatchValues<Real, typename Exec::memory_space>>,
    CustomConditions...>;

/// @brief Collectively configures one scalar, sampling CSV profiles on the local face centroids.
/// Pass identical configurations on all ranks and the host mesh used to build the layout.
/// Relative CSV paths are resolved from rank 0's working directory. Copies complete before return.
/// Complete prior boundary kernels before rebinding, and finalize the manager before dispatch.
template<class Exec, class Real, class... CustomConditions>
void configure_tabulated(Manager<Exec, Real, CustomConditions...>& manager, const Configuration& config,
                         const mesh::host::MeshPart& host, mpi::CommView comm) {
    mpi::require(!comm.is_null() && host.rank == comm.rank() && host.nprocs == comm.size(),
                 "Scalar boundary: CSV configuration requires the mesh communicator");
    const auto& exec = manager.execution_space();
    manager.configure(config, [&](const ConditionConfig& spec, PatchRange range, const auto& assign) {
        const auto& path = std::get<std::filesystem::path>(spec.parameter);
        const auto provider = boundary::detail::load_patch_values<Real>(
                path, range, host, comm, exec, detail::checked_parameter<Real>);
        switch (spec.kind) {
            case Kind::FixedValue: assign(fixed_value<Real>(provider)); return;
            case Kind::NormalGradient: assign(normal_gradient<Real>(provider)); return;
            case Kind::FixedFlux: assign(fixed_flux<Real>(provider)); return;
        }
        mpi::fatal("Scalar boundary: unknown condition kind");
    });
}

} // namespace krona::boundary::scalar