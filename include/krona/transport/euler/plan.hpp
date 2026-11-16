#pragma once
#include <krona/boundary/patches.hpp>
#include <krona/numerics/reconstruction/types.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <utility>

namespace krona::transport::euler {
struct FaceRange { LocalIndex begin=0,end=-1; }; // end==-1 selects all inner faces.
using CellRange=numerics::reconstruction::CellRange;

/// @brief Borrows validated mesh connectivity and geometry and owns the host patch catalog.
/// Construction is local and blocking.
template<class Exec=execution::ExecutionSpace,class Real=ComputeReal>
class EulerPlan {
public:
    using execution_space=Exec;
    using memory_space=typename Exec::memory_space;
    using Mesh=mesh::device::MeshView<memory_space>;
    using Connectivity=mesh::device::AuxConnectivityView<memory_space>;
    EulerPlan(const mesh::host::MeshPart& host,Mesh view,Connectivity connectivity,mpi::CommView comm,Exec exec={})
        :mesh_(view),connectivity_(connectivity),comm_(comm),exec_(exec),catalog_(host) {
        execution::require_kokkos();
        const auto& s=mesh_.sizes;
        mpi::require(host.rank==comm.rank()&&host.nprocs==comm.size(),"Euler: mesh/communicator mismatch");
        mpi::require(host.faces_grouped&&s.faces_grouped&&host.n_own==s.n_own&&host.n_cells==s.n_cells&&
            host.n_faces==s.n_faces&&host.n_internal_faces==s.n_internal_faces&&host.n_inner_faces==s.n_inner_faces,
            "Euler: grouped host/view mesh mismatch");
        using mesh::device::MeshData;
        const auto checked=mesh::device::validation::validate<
            MeshData::FaceCells|MeshData::CellGeometry|MeshData::FaceGeometry,
            mesh::AuxConnType::CellFaces,mesh::AuxGeomType::None,true>(exec_,mesh_,connectivity_);
        mpi::require(checked.ok(),"Euler: invalid mesh geometry or cell-face incidence");
        mpi::require(host.face_owner.size()==static_cast<std::size_t>(s.n_faces),"Euler: missing host face owners");
        const std::size_t first = static_cast<std::size_t>(s.n_inner_faces);
        const std::size_t end = static_cast<std::size_t>(s.n_faces);
        if (first < end) {
            const auto device_owners = Kokkos::subview(mesh_.connectivity.face_owner, std::pair{first, end});
            const auto owners = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_owners);
            for (std::size_t i = 0; i < owners.extent(0); ++i)
                mpi::require(owners(i) == host.face_owner[first + i], "Euler: host/view boundary ownership differs");
        }
    }
    [[nodiscard]] Mesh mesh_view()const noexcept{return mesh_;}
    [[nodiscard]] Connectivity connectivity_view()const noexcept{return connectivity_;}
    [[nodiscard]] mpi::CommView communicator()const noexcept{return comm_;}
    [[nodiscard]] const Exec& execution_space_instance()const noexcept{return exec_;}
    [[nodiscard]] const boundary::PatchCatalog& catalog()const noexcept{return catalog_;}
private:
    Mesh mesh_;
    Connectivity connectivity_;
    mpi::CommView comm_;
    Exec exec_;
    boundary::PatchCatalog catalog_;
};
} // namespace krona::transport::euler