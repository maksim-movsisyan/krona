#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mesh/device/validation.hpp>

#include <type_traits>

namespace krona::transport::scalar::convection {

/// @brief Blocking local validation of borrowed static mesh/CellFaces descriptors.
/// No algebra graph, geometry cache, work array or communicator is allocated/owned.
/// Nonorthogonal meshes are supported; cell geometry and diffusion metrics are not required.
/// The communicator must match the device partition; descriptors contain no rank metadata.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class ConvectionPlan {
    static_assert(std::is_floating_point_v<Real>);
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using Mesh = mesh::device::MeshView<memory_space>;
    using Connectivity = mesh::device::AuxConnectivityView<memory_space>;

    ConvectionPlan(Mesh mesh_view, Connectivity connectivity,
        mpi::CommView comm, Exec exec = {})
        : mesh_(mesh_view), connectivity_(connectivity), comm_(comm), exec_(exec)
    {
        mpi::guarded([&] {
            execution::require_kokkos();
            mpi::require(comm.native() != MPI_COMM_NULL, "Scalar convection: null communicator");
            using mesh::device::MeshData;
            const auto result = mesh::device::validation::validate<
                MeshData::FaceCells | MeshData::FaceGeometry,
                mesh::AuxConnType::CellFaces, mesh::AuxGeomType::None, true>(exec_, mesh_, connectivity_);
            mpi::require(result.ok(), "Scalar convection: invalid face geometry or cell-face incidence");
        });
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
