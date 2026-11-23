#pragma once

#include <krona/transport/operator_plan.hpp>

namespace krona::transport::scalar::convection {

/// @brief Mesh plan for scalar convection on grouped faces, including skewed meshes.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class ConvectionPlan : public OperatorPlan<Exec, Real> {
    using Base = OperatorPlan<Exec, Real>;

public:
    /// @brief Borrows face geometry and the cell-to-face lists for owned rows.
    ConvectionPlan(typename Base::Mesh mesh, typename Base::Connectivity connectivity,
                   mpi::CommView comm, Exec exec = {}, typename Base::Geometry geometry = {})
        : Base(mesh, connectivity, comm, exec, geometry) {}
};

} // namespace krona::transport::scalar::convection
