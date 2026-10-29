#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/mesh/host/mesh_part.hpp>

#include <type_traits>
 
namespace krona::transport::scalar::convection {

/// @brief Blocking local validation of borrowed static mesh/CellFaces descriptors.
/// No algebra graph, geometry cache, work array or communicator is allocated/owned.
/// Nonorthogonal meshes are supported; no diffusion metrics are required.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class ConvectionPlan {
    static_assert(std::is_floating_point_v<Real>);
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using Mesh = mesh::device::MeshView<memory_space>;
    using Connectivity = mesh::device::AuxConnectivityView<memory_space>;

    ConvectionPlan(const mesh::host::MeshPart& host, Mesh mesh_view, Connectivity connectivity,
        mpi::CommView comm, Exec exec = {})
        : mesh_(mesh_view), connectivity_(connectivity), comm_(comm), exec_(exec)
    {
        execution::require_kokkos();
        const auto& s = mesh_.sizes;
        mpi::require(host.rank == comm.rank() && host.nprocs == comm.size(),
            "Scalar convection: mesh/communicator mismatch");
        mpi::require(host.faces_grouped && s.faces_grouped && host.n_own == s.n_own &&
            host.n_cells == s.n_cells && host.n_faces == s.n_faces &&
            host.n_internal_faces == s.n_internal_faces && host.n_inner_faces == s.n_inner_faces,
            "Scalar convection: grouped host/view mesh mismatch");
        using mesh::device::MeshData;
        const auto result = mesh::device::validation::validate<
            MeshData::FaceCells | MeshData::CellGeometry | MeshData::FaceGeometry,
            mesh::AuxConnType::CellFaces, mesh::AuxGeomType::None, true>(exec_, mesh_, connectivity_);
        mpi::require(result.ok(), "Scalar convection: invalid mesh geometry or cell-face incidence");
    }

    [[nodiscard]] Mesh mesh_view() const noexcept { return mesh_; }
    [[nodiscard]] Connectivity connectivity_view() const noexcept { return connectivity_; }
    [[nodiscard]] mpi::CommView communicator() const noexcept { return comm_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
private:
    Mesh mesh_;
    Connectivity connectivity_;
    mpi::CommView comm_;
    Exec exec_;
};

} // namespace krona::transport::scalar::convection