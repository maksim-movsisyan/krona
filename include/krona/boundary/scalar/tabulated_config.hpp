#pragma once

#include <krona/boundary/detail/tabulated_values.hpp>
#include <krona/boundary/scalar/detail/condition_binding.hpp>
#include <krona/boundary/scalar/manager.hpp>

namespace krona::boundary::scalar {

/// @brief Scalar manager with CSV-backed providers for fixed values, gradients and fluxes.
/// @tparam Exec Kokkos execution space whose memory stores the sampled profile values.
/// @tparam Real Floating-point type used for the sampled data and boundary calculation.
/// @tparam CustomConditions Additional device-callable conditions to register with the manager.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
using TabulatedManager = Manager<Exec, Real,
    FixedValue<Real, boundary::detail::TabulatedPatchValues<Real, typename Exec::memory_space>>,
    NormalGradient<Real, boundary::detail::TabulatedPatchValues<Real, typename Exec::memory_space>>,
    FixedFlux<Real, boundary::detail::TabulatedPatchValues<Real, typename Exec::memory_space>>,
    CustomConditions...>;

/// @brief Binds one scalar's constants and samples any CSV profiles at local face centroids.
/// @tparam Exec Kokkos execution space used to upload the samples.
/// @tparam Real Floating-point type used for the sampled data and boundary calculation.
/// @tparam CustomConditions Manager's extra condition types, including all three tabulated kinds.
/// @param manager Manager built from this host mesh's layout, typically a TabulatedManager.
/// @param config Same configuration, in the same patch order, on every rank.
/// @param host Host mesh used to build the layout. Its local face centroids are the sampling points.
/// @param comm Mesh communicator used to load the CSV data collectively.
/// @details Rank 0 resolves relative paths from its working directory. Each rank takes part even
/// when it owns zero faces on the patch. Uploads complete before return. Finish earlier boundary
/// work before rebinding and finalize the manager before dispatch.
template<class Exec, class Real, class... CustomConditions>
void configure_tabulated(Manager<Exec, Real, CustomConditions...>& manager, const Configuration& config,
                         const mesh::host::MeshPart& host, mpi::CommView comm) {
    mpi::require(!comm.is_null() && host.rank == comm.rank() && host.nprocs == comm.size(),
                 "Scalar boundary: CSV configuration requires the mesh communicator");
    const auto& exec = manager.execution_space();
    manager.configure(config, [&](const ConditionConfig& spec, PatchRange range, const auto& assign) {
        const auto& path = std::get<std::filesystem::path>(spec.parameter);
        // Sample once during setup and keep the per-face values with the assigned condition.
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
