#pragma once

#include <krona/transport/scalar/diffusion/detail/plan.hpp>

namespace krona::transport::scalar::diffusion {

/// @brief Borrows validated mesh geometry and connectivity for a diffusion operator.
/// Construction waits for local validation. The mesh, any supplied caches and communicator must stay alive.
/// @tparam Exec Kokkos execution space used for validation and calculations.
/// @tparam Real Floating-point type used for diffusion arithmetic.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionPlan {
    static_assert(std::is_floating_point_v<Real>);
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using Mesh = mesh::device::MeshView<memory_space>;
    using Connectivity = mesh::device::AuxConnectivityView<memory_space>;
    using Geometry = mesh::device::AuxGeometryView<memory_space>;

    /// @brief Validates the mesh, face incidence and any supplied metric caches.
    /// @param mesh_view Mesh with cell centres, face geometry and owner/neighbour IDs.
    /// @param connectivity Complete cell-to-face lists for owned rows.
    /// @param geometry Optional cache. Pass {} to compute all derived metrics on demand.
    /// @param comm Communicator for the mesh partition and global balance calculations.
    /// @param exec Execution instance used for validation and submissions.
    /// @param options Geometry checks applied during construction.
    DiffusionPlan(Mesh mesh_view, Connectivity connectivity,
                  Geometry geometry, mpi::CommView comm, Exec exec = {}, DiffusionOptions options = {})
        : exec_(exec), mesh_(mesh_view), connectivity_(connectivity), geometry_(geometry), comm_(comm)
    { detail::validate_plan<Exec, Real>(mesh_, connectivity_, geometry_, comm_, exec_, options); }

    /// @brief Returns the execution instance used for validation and submissions.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
    /// @brief Returns the borrowed communicator of the partition.
    [[nodiscard]] mpi::CommView communicator() const noexcept { return comm_; }
    /// @brief Returns the borrowed primary mesh arrays.
    [[nodiscard]] Mesh mesh_view() const noexcept { return mesh_; }
    /// @brief Returns the borrowed cell-to-face incidence.
    [[nodiscard]] Connectivity connectivity_view() const noexcept { return connectivity_; }
    /// @brief Returns the supplied metric cache. A missing array is evaluated when needed.
    [[nodiscard]] Geometry geometry_view() const noexcept { return geometry_; }
private:
    Exec exec_;
    Mesh mesh_;
    Connectivity connectivity_;
    Geometry geometry_;
    mpi::CommView comm_;
};

} // namespace krona::transport::scalar::diffusion
