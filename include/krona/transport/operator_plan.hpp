#pragma once

#include <krona/boundary/patches.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/mpi/communicator.hpp>

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace krona::transport {

/// @brief Retains the mesh views, communicator and execution instance for a transport operator.
/// @tparam Exec Execution space used to validate the mesh and submit operator work.
/// @tparam Real Arithmetic type used by the operator consuming this plan.
/// @details Construction waits for local mesh checks. The mesh, auxiliary arrays and communicator
/// must outlive the plan and the work submitted with them.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class OperatorPlan {
    static_assert(std::is_floating_point_v<Real>);

public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;
    using Mesh = mesh::device::MeshView<memory_space>;
    using Connectivity = mesh::device::AuxConnectivityView<memory_space>;
    using Geometry = mesh::device::AuxGeometryView<memory_space>;

    /// @brief Validates face geometry and owned-cell incidence and retains any supplied metric views.
    OperatorPlan(Mesh mesh, Connectivity connectivity, mpi::CommView comm, Exec exec = {},
                 Geometry geometry = {})
        : OperatorPlan(exec, mesh, connectivity, geometry, comm) {
        validate_mesh<face_data>();
    }

    /// @brief Also validates cell geometry and builds a patch catalog from the matching host mesh.
    OperatorPlan(const mesh::host::MeshPart& host, Mesh mesh, Connectivity connectivity,
                 mpi::CommView comm, Exec exec = {}, Geometry geometry = {})
        : OperatorPlan(exec, mesh, connectivity, geometry, comm) {
        const auto& sizes = mesh_.sizes;
        mpi::require(host.rank == comm_.rank() && host.nprocs == comm_.size(),
                     "Transport operator plan: mesh/communicator mismatch");
        mpi::require(host.faces_grouped && sizes.faces_grouped &&
                     host.n_own == sizes.n_own && host.n_cells == sizes.n_cells &&
                     host.n_faces == sizes.n_faces && host.n_internal_faces == sizes.n_internal_faces &&
                     host.n_inner_faces == sizes.n_inner_faces,
                     "Transport operator plan: grouped host/view mesh mismatch");

        validate_mesh<face_data | mesh::device::MeshData::CellGeometry>();
        mpi::guarded([&] {
            validate_boundary_owners(host);
            catalog_.emplace(host);
        });
    }

    [[nodiscard]] Mesh mesh_view() const noexcept { return mesh_; }
    [[nodiscard]] Connectivity connectivity_view() const noexcept { return connectivity_; }
    [[nodiscard]] Geometry geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] mpi::CommView communicator() const noexcept { return comm_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
    [[nodiscard]] bool has_patch_catalog() const noexcept { return catalog_.has_value(); }

    /// @brief Returns the patch catalog of a plan built with a host mesh.
    [[nodiscard]] const boundary::PatchCatalog& catalog() const noexcept {
        mpi::require(has_patch_catalog(), "Transport operator plan: missing patch catalog");
        return *catalog_;
    }

private:
    using MeshData = mesh::device::MeshData;
    static constexpr auto face_data = MeshData::FaceCells | MeshData::FaceGeometry;

    Mesh mesh_;
    Connectivity connectivity_;
    Geometry geometry_;
    mpi::CommView comm_;
    Exec exec_;
    std::optional<boundary::PatchCatalog> catalog_;

    OperatorPlan(Exec exec, Mesh mesh, Connectivity connectivity, Geometry geometry, mpi::CommView comm)
        : mesh_(mesh), connectivity_(connectivity), geometry_(geometry), comm_(comm), exec_(exec) {
        execution::require_kokkos();
        mpi::require(comm_.native() != MPI_COMM_NULL, "Transport operator plan: null communicator");
    }

    template<MeshData Data>
    void validate_mesh() const {
        mpi::guarded([&] {
            const auto status = mesh::device::validation::validate<
                Data, mesh::AuxConnType::CellFaces, mesh::AuxGeomType::None, true>(
                    exec_, mesh_, connectivity_, geometry_);
            mpi::require(status.ok(),
                         "Transport operator plan: invalid face geometry or cell-face incidence (check normal metrics)");
        });
    }

    void validate_boundary_owners(const mesh::host::MeshPart& host) const {
        const auto first = static_cast<std::size_t>(mesh_.sizes.n_inner_faces);
        const auto end = static_cast<std::size_t>(mesh_.sizes.n_faces);
        mpi::require(host.face_owner.size() == end, "Transport operator plan: missing host face owners");

        // Boundary kernels combine the host patch ranges with the device owner indices.
        if (first == end) return;
        const auto device_owners = Kokkos::subview(mesh_.connectivity.face_owner, std::pair{first, end});
        const auto owners = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_owners);
        for (std::size_t i = 0; i < owners.extent(0); ++i)
            mpi::require(owners(i) == host.face_owner[first + i],
                         "Transport operator plan: host/view boundary ownership differs");
    }
};

} // namespace krona::transport
