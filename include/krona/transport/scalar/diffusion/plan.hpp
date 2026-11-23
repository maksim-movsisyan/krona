#pragma once

#include <krona/transport/operator_plan.hpp>
#include <krona/transport/scalar/diffusion/detail/plan.hpp>

namespace krona::transport::scalar::diffusion {

/// @brief Mesh plan with geometry checks for a scalar diffusion operator.
/// Construction waits for local validation. The mesh, any supplied caches and communicator must stay alive.
/// @tparam Exec Kokkos execution space used for validation and calculations.
/// @tparam Real Floating-point type used for diffusion arithmetic.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionPlan : public OperatorPlan<Exec, Real> {
    using Base = OperatorPlan<Exec, Real>;

public:
    /// @brief Validates the mesh, face incidence and any supplied metric caches.
    /// @param mesh Mesh with cell centres, face geometry and owner/neighbour IDs.
    /// @param connectivity Complete cell-to-face lists for owned rows.
    /// @param geometry Optional cache. Pass {} to compute all derived metrics on demand.
    /// @param comm Communicator for the mesh partition and global balance calculations.
    /// @param exec Execution instance used for validation and submissions.
    /// @param options Geometry checks applied during construction.
    DiffusionPlan(typename Base::Mesh mesh, typename Base::Connectivity connectivity,
                  typename Base::Geometry geometry, mpi::CommView comm, Exec exec = {}, DiffusionOptions options = {})
        : Base(mesh, connectivity, comm, exec, geometry) {
        detail::validate_geometry<Exec, Real>(mesh, geometry, exec, options);
    }
};

} // namespace krona::transport::scalar::diffusion
