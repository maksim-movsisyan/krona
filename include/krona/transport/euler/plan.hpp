#pragma once
#include <krona/boundary/patches.hpp>
#include <krona/numerics/reconstruction/types.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/communicator.hpp>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace krona::transport::euler {
struct FaceRange { LocalIndex begin=0,end=-1; }; // end==-1 selects all inner faces.
using CellRange=numerics::reconstruction::CellRange;

namespace detail {
template<class Space>
struct BoundaryRows {
    Kokkos::View<const LocalIndex*,Space> cells,faces;
    Kokkos::View<const std::size_t*,Space> offsets;
};
template<class Owners,class Space>
struct ValidateBoundaryRows {
    Owners owners;
    BoundaryRows<Space> rows;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t row,std::size_t& count)const
    {
        for(auto k=rows.offsets(row);k<rows.offsets(row+1);++k)
            if(owners(static_cast<std::size_t>(rows.faces(k)))!=rows.cells(row))++count;
    }
};
struct PatchRows { boundary::PatchRange faces;std::size_t begin=0,end=0; };
}

/// @brief Borrowed validated mesh plus reusable boundary incidence grouped by (patch,owned cell).
/// Construction is local and blocking; copies share device maps. No state/flux/matrix storage.
template<class Exec=execution::ExecutionSpace,class Real=ComputeReal>
class EulerPlan {
public:
    using execution_space=Exec;
    using memory_space=typename Exec::memory_space;
    using Mesh=mesh::device::MeshView<memory_space>;
    using Connectivity=mesh::device::AuxConnectivityView<memory_space>;
    EulerPlan(const mesh::host::MeshPart& host,Mesh view,Connectivity connectivity,mpi::CommView comm,Exec exec={})
        :mesh_(view),connectivity_(connectivity),comm_(comm),exec_(exec),catalog_(host)
    {
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
        std::vector<LocalIndex> cells,faces;
        std::vector<std::size_t> offsets{0};
        auto append=[&](boundary::PatchRange range){
            detail::PatchRows rows{range,cells.size(),cells.size()};
            std::vector<std::pair<LocalIndex,LocalIndex>> entries;
            for(auto f=range.begin;f<range.end;++f){
                const auto owner=host.face_owner[static_cast<std::size_t>(f)];
                mpi::require(owner>=0&&owner<s.n_own,"Euler: invalid host boundary owner");
                entries.emplace_back(owner,f);
            }
            std::sort(entries.begin(),entries.end());
            for(std::size_t k=0;k<entries.size();){
                const auto owner=entries[k].first;cells.push_back(owner);
                do{faces.push_back(entries[k++].second);}while(k<entries.size()&&entries[k].first==owner);
                offsets.push_back(faces.size());
            }
            rows.end=cells.size();patch_rows_.push_back(rows);
        };
        for(const auto& patch:catalog_.patches())append(patch.range);
        append(catalog_.unlabelled());
        execution::complete(exec_,[&]{
            auto upload=[&]<class T>(const std::vector<T>& data,const char* label){
                Kokkos::View<T*,memory_space> buffer(Kokkos::view_alloc(exec_,Kokkos::WithoutInitializing,std::string(label)),data.size());
                Kokkos::View<const T*,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> source(data.data(),data.size());
                Kokkos::deep_copy(exec_,buffer,source);return buffer;
            };
            rows_.cells=upload(cells,"euler/boundary_cells");rows_.faces=upload(faces,"euler/boundary_faces");
            rows_.offsets=upload(offsets,"euler/boundary_offsets");
        });
        std::size_t bad=0;
        using Policy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<std::size_t>>;
        using Validator=detail::ValidateBoundaryRows<decltype(mesh_.connectivity.face_owner),memory_space>;
        Kokkos::parallel_reduce("euler/validate_boundary_map",Policy(exec_,0,cells.size()),
            Validator{mesh_.connectivity.face_owner,rows_},bad);
        mpi::require(bad==0,"Euler: host/view boundary ownership differs");
    }
    [[nodiscard]] Mesh mesh_view()const noexcept{return mesh_;}
    [[nodiscard]] Connectivity connectivity_view()const noexcept{return connectivity_;}
    [[nodiscard]] mpi::CommView communicator()const noexcept{return comm_;}
    [[nodiscard]] const Exec& execution_space_instance()const noexcept{return exec_;}
    [[nodiscard]] detail::BoundaryRows<memory_space> boundary_rows()const noexcept{return rows_;}
    [[nodiscard]] const boundary::PatchCatalog& catalog()const noexcept{return catalog_;}
    [[nodiscard]] detail::PatchRows patch_rows(boundary::PatchRange range)const
    {
        const auto id=range.patch_id<0?catalog_.patches().size():static_cast<std::size_t>(range.patch_id);
        mpi::require(id<patch_rows_.size(),"Euler: unknown boundary patch");
        const auto rows=patch_rows_[id];
        mpi::require(rows.faces.patch_id==range.patch_id&&rows.faces.begin==range.begin&&rows.faces.end==range.end,
            "Euler: boundary catalog differs from plan");
        return rows;
    }
private:
    Mesh mesh_;
    Connectivity connectivity_;
    mpi::CommView comm_;
    Exec exec_;
    boundary::PatchCatalog catalog_;
    detail::BoundaryRows<memory_space> rows_;
    std::vector<detail::PatchRows> patch_rows_;
};
} // namespace krona::transport::euler