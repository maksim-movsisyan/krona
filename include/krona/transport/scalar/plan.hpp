#pragma once

#include <krona/transport/scalar/detail/plan.hpp>

#include <utility>

namespace krona::transport::scalar {

/// @brief Immutable shared static-mesh plan. Copies share graph and entry maps.
/// Mesh/auxiliary storage is borrowed and must outlive the plan and all its users.
/// Mesh validation is blocking; constructing a new assembly plan is collective.
/// No geometry or cell-face tables are allocated here.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionPlan {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using assembly_plan_type = linalg::FaceAssemblyPlan<Exec>;
    using Mesh = mesh::compute::MeshView<memory_space>;
    using Connectivity = mesh::compute::AuxConnectivityView<memory_space>;
    using Geometry = mesh::compute::AuxGeometryView<memory_space>;

    DiffusionPlan(const mesh::MeshPart& host, Mesh mesh_view, Connectivity connectivity,
                  Geometry geometry, mpi::CommView comm, Exec exec = {}, DiffusionOptions options = {})
        : DiffusionPlan(host, mesh_view, connectivity, geometry, assembly_plan_type(host, comm, exec), exec, options) {}

    /// @brief Attach an existing shared CSR/BSR assembly plan for the same indexed mesh and communicator.
    /// The caller must preserve the original cell/face numbering and halo topology.
    DiffusionPlan(const mesh::MeshPart& host, Mesh mesh_view, Connectivity connectivity,
                  Geometry geometry, assembly_plan_type assembly, Exec exec = {}, DiffusionOptions options = {})
        : exec_(exec), mesh_(mesh_view), connectivity_(connectivity), geometry_(geometry), assembly_(std::move(assembly))
    {
        detail::validate_plan<Exec, Real>(host, mesh_view, connectivity, geometry, assembly_, exec, options);
    }

    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
    [[nodiscard]] mpi::CommView communicator() const noexcept { return assembly_.communicator(); }
    [[nodiscard]] Mesh mesh_view() const noexcept { return mesh_; }
    [[nodiscard]] Connectivity connectivity_view() const noexcept { return connectivity_; }
    [[nodiscard]] Geometry geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] const assembly_plan_type& assembly_plan() const noexcept { return assembly_; }
    [[nodiscard]] const linalg::CsrPattern<Exec>& pattern() const noexcept { return assembly_.pattern(); }
    [[nodiscard]] linalg::FaceAssemblyMap<memory_space> row_plan() const noexcept { return assembly_.entry_map(); }

private:
    Exec exec_;
    Mesh mesh_;
    Connectivity connectivity_;
    Geometry geometry_;
    assembly_plan_type assembly_;
};

} // namespace krona::transport::scalar