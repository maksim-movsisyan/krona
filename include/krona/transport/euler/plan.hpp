#pragma once

#include <krona/transport/operator_plan.hpp>

namespace krona::transport::euler {

/// @brief Mesh plan with a host patch catalog for Euler assembly.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class EulerPlan : public OperatorPlan<Exec, Real> {
    using Base = OperatorPlan<Exec, Real>;

public:
    /// @brief Borrows the supplied mesh and builds the patch catalog from its host layout.
    EulerPlan(const mesh::host::MeshPart& host, typename Base::Mesh mesh,
              typename Base::Connectivity connectivity, mpi::CommView comm, Exec exec = {},
              typename Base::Geometry geometry = {})
        : Base(host, mesh, connectivity, comm, exec, geometry) {}
};

} // namespace krona::transport::euler
