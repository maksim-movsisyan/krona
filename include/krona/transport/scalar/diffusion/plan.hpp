#pragma once

#include <krona/transport/scalar/diffusion/detail/plan.hpp>

namespace krona::transport::scalar::diffusion {

/// @brief Validated borrowed static-mesh geometry/connectivity; no algebra graph, matrix or vector allocation.
/// Construction validates locally and is blocking. Mesh/auxiliary owners and communicator outlive all users.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionPlan {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using Mesh = mesh::compute::MeshView<memory_space>;
    using Connectivity = mesh::compute::AuxConnectivityView<memory_space>;
    using Geometry = mesh::compute::AuxGeometryView<memory_space>;

    DiffusionPlan(const mesh::MeshPart& host, Mesh mesh_view, Connectivity connectivity,
                  Geometry geometry, mpi::CommView comm, Exec exec = {}, DiffusionOptions options = {})
        : exec_(exec), mesh_(mesh_view), connectivity_(connectivity), geometry_(geometry), comm_(comm)
    { detail::validate_plan<Exec, Real>(host, mesh_, connectivity_, geometry_, comm_, exec_, options); }

    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
    [[nodiscard]] mpi::CommView communicator() const noexcept { return comm_; }
    [[nodiscard]] Mesh mesh_view() const noexcept { return mesh_; }
    [[nodiscard]] Connectivity connectivity_view() const noexcept { return connectivity_; }
    [[nodiscard]] Geometry geometry_view() const noexcept { return geometry_; }
private:
    Exec exec_;
    Mesh mesh_;
    Connectivity connectivity_;
    Geometry geometry_;
    mpi::CommView comm_;
};

} // namespace krona::transport::scalar::diffusion
