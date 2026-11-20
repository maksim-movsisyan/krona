#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mesh/device/validation.hpp>

#include <type_traits>

namespace krona::transport::scalar::convection {

/// @brief Holds the face geometry and cell-to-face lists shared by convection launches.
/// The plan borrows the mesh arrays and uses only face geometry, so skewed meshes work too.
/// @tparam Exec Kokkos execution space used to validate the mesh and run convection kernels.
/// @tparam Real Floating-point type used to evaluate the convection terms.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class ConvectionPlan {
    static_assert(std::is_floating_point_v<Real>);
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using Mesh = mesh::device::MeshView<memory_space>;
    using Connectivity = mesh::device::AuxConnectivityView<memory_space>;

    /// @brief Validates the mesh and cell-to-face lists locally before retaining their views.
    /// @param mesh_view Device mesh with grouped faces, face geometry and owner/neighbour indices.
    /// @param connectivity Cell-to-face lists for the owned cells of that mesh.
    /// @param comm Communicator for the partition described by the mesh.
    /// @param exec Execution instance used for validation and subsequent launches.
    ConvectionPlan(Mesh mesh_view, Connectivity connectivity,
        mpi::CommView comm, Exec exec = {})
        : mesh_(mesh_view), connectivity_(connectivity), comm_(comm), exec_(exec) {
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

    /// @brief Returns the borrowed device mesh and its local sizes.
    [[nodiscard]] Mesh mesh_view() const noexcept { return mesh_; }
    /// @brief Returns the borrowed cell-to-face lists.
    [[nodiscard]] Connectivity connectivity_view() const noexcept { return connectivity_; }
    /// @brief Returns the communicator associated with this partition.
    [[nodiscard]] mpi::CommView communicator() const noexcept { return comm_; }
    /// @brief Returns the execution instance used by this plan.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
private:
    Mesh mesh_;
    Connectivity connectivity_;
    mpi::CommView comm_;
    Exec exec_;
};

} // namespace krona::transport::scalar::convection
