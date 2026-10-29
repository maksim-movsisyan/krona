#pragma once
#include <krona/boundary/geometry_view.hpp>
#include <krona/transport/euler/state.hpp>
#include <krona/transport/euler/reconstruction.hpp>
#include <krona/transport/euler/boundary.hpp>
#include <krona/transport/euler/plan.hpp>
#include <krona/transport/euler/status.hpp>

namespace krona::transport::euler::detail {

template<class Space>
struct RowData {
    using Index=mesh::device::detail::MeshArray<LocalIndex,Space,false>;
    using Offset=mesh::device::detail::MeshArray<std::size_t,Space,false>;
    using Metric=mesh::device::detail::MeshArray<GeometryReal,Space,false>;
    Index owner,neighbour,faces;
    Offset offsets;
    Metric nx,ny,nz,area;
};
template<class Space>auto row_data(mesh::device::MeshView<Space> m,mesh::device::AuxConnectivityView<Space> c)
{ return RowData<Space>{m.connectivity.face_owner,m.connectivity.face_neigh,c.cell_faces,c.cell_faces_offsets,
    m.geometry.face_normal_x,m.geometry.face_normal_y,m.geometry.face_normal_z,m.geometry.face_area}; }

template<class R>struct CellState { boundary::physical::Primitive<R,true> primitive; numerics::riemann::State<R> state; };

template<class R,class Reconstruction,class StateLaw,class Solver,class Flags>
struct Physics {
    Reconstruction reconstruction;
    StateLaw state_law;
    Solver solver;
    Flags flags;

    KOKKOS_INLINE_FUNCTION auto prepare(boundary::physical::Primitive<R,true> q)const
    {
        if(!primitive_valid(q))return numerics::riemann::State<R>{};
        return state_law(q);
    }
    KOKKOS_INLINE_FUNCTION auto base(std::size_t i)const
    {
        const auto p=primitive<R>(reconstruction.data.values,i);
        return CellState<R>{p,prepare(p)};
    }
    KOKKOS_INLINE_FUNCTION auto reconstructed(const Kokkos::Array<R,5>& values,const CellState<R>& cell)const
    {
        const auto result=prepare(primitive(values));
        if(state_valid(result))return result;
        count(flags,ReconstructionFallbacks);
        return cell.state;
    }
    template<class Space>
    KOKKOS_INLINE_FUNCTION auto inner(const RowData<Space>& data,std::size_t f,std::size_t cell,const CellState<R>& current)const
    {
        const auto p=static_cast<std::size_t>(data.owner(f)),n=static_cast<std::size_t>(data.neighbour(f));
        const auto left=p==cell?current:base(p),right=n==cell?current:base(n);
        if(!state_valid(left.state)||!state_valid(right.state)){
            flag(flags,BadState);return numerics::riemann::Result<R>{{},R{0},numerics::riemann::Status::InvalidState};
        }
        auto l=left.state,r=right.state;
        if constexpr(!std::is_same_v<typename Reconstruction::scheme_type,numerics::reconstruction::FirstOrder>){
            const auto traces=reconstruction.template interior<R>(f);
            l=reconstructed(traces.left,left);r=reconstructed(traces.right,right);
        }
        auto result=solver(l,r,numerics::riemann::Vector<R>{static_cast<R>(data.nx(f)),static_cast<R>(data.ny(f)),static_cast<R>(data.nz(f))});
        check_flux(result);return result;
    }
    KOKKOS_INLINE_FUNCTION void check_flux(const numerics::riemann::Result<R>& result)const
    {
        if(!result.valid())flag(flags,BadFlux);
        else if(result.used_fallback())count(flags,RiemannFallbacks);
    }
    template<class Model,class Condition,class Closure>
    KOKKOS_INLINE_FUNCTION auto exterior(const boundary::FaceContext<R>& c,const Condition& condition,
        const Model& model,const Closure& closure,boundary::physical::Primitive<R,true> inside,
        boundary::physical::Primitive<R,true> face)const
    {
        if constexpr(requires{model.at(c,condition);})return closure(c,condition,inside,face,model.at(c,condition));
        else return closure(c,condition,inside,face,model);
    }
    template<class Model,class Condition,class Closure>
    KOKKOS_INLINE_FUNCTION auto boundary_flux(const boundary::FaceContext<R>& c,const Condition& condition,
        const Model& model,const Closure& closure,const CellState<R>& current)const
    {
        auto l=current.state;
        auto inside=current.primitive;
        if constexpr(!std::is_same_v<typename Reconstruction::boundary_scheme_type,numerics::reconstruction::FirstOrder>){
            const auto a=reconstruction.template boundary<R>(static_cast<std::size_t>(c.face));
            const auto candidate=prepare(primitive(a));
            if(state_valid(candidate)){l=candidate;inside=primitive(a);}else count(flags,ReconstructionFallbacks);
        }
        const auto face=primitive<R>(reconstruction.data.values,c.extended_index);
        const auto outside=exterior(c,condition,model,closure,inside,face);
        const auto r=prepare(outside);
        if(!state_valid(r)){
            flag(flags,BadBoundary);return numerics::riemann::Result<R>{{},R{0},numerics::riemann::Status::InvalidState};
        }
        auto result=solver(l,r,numerics::riemann::Vector<R>{c.nx,c.ny,c.nz});check_flux(result);return result;
    }
};

template<class A,class R>
KOKKOS_INLINE_FUNCTION void accumulate(Kokkos::Array<A,5>& sum,const numerics::riemann::Conservative<R>& f,A factor)
{
    sum[0]+=factor*static_cast<A>(f.mass);sum[1]+=factor*static_cast<A>(f.momentum.x);
    sum[2]+=factor*static_cast<A>(f.momentum.y);sum[3]+=factor*static_cast<A>(f.momentum.z);sum[4]+=factor*static_cast<A>(f.energy);
}
template<class A,class Output,class Flags>
KOKKOS_INLINE_FUNCTION void add(Output output,std::size_t i,const Kokkos::Array<A,5>& sum,Flags flags)
{
    using T=std::remove_const_t<typename Output::value_type>;
    Kokkos::Array<T,5> result;
    for(std::size_t c=0;c<5;++c){
        const A next=static_cast<A>(output(i,c))+sum[c];result[c]=static_cast<T>(next);
        if(!Kokkos::isfinite(sum[c])||!Kokkos::isfinite(next)||!Kokkos::isfinite(result[c])){flag(flags,BadOutput);return;}
    }
    for(std::size_t c=0;c<5;++c)output(i,c)=result[c];
}

template<class Space,class R,class A,class Output,class Physics>
struct InnerRows {
    RowData<Space> data;Output output;Physics physics;FaceRange range;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const
    {
        const auto current=physics.base(i);
        if(!state_valid(current.state)){flag(physics.flags,BadState);return;}
        Kokkos::Array<A,5> sum{};
        for(auto k=data.offsets(i);k<data.offsets(i+1);++k){
            const auto face=data.faces(k);
            if(face<range.begin||face>=range.end)continue;
            const auto f=static_cast<std::size_t>(face);
            const auto flux=physics.inner(data,f,i,current);
            if(flux.valid())accumulate(sum,flux.flux,(data.owner(f)==static_cast<LocalIndex>(i)?A{-1}:A{1})*static_cast<A>(data.area(f)));
        }
        add(output,i,sum,physics.flags);
    }
};

template<class Space,class R,class A,class Output,class Physics,class Model,class Condition,class Closure>
struct BoundaryCellRows {
    boundary::GeometryView<Space> geometry;
    BoundaryRows<Space> rows;Output output;Physics physics;Model model;Condition condition;Closure closure;
    CellRange cells;R time;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t row)const
    {
        const auto cell=static_cast<std::size_t>(rows.cells(row));
        if(cell<cells.begin||cell>=cells.end)return;
        const auto current=physics.base(cell);
        if(!state_valid(current.state)){flag(physics.flags,BadState);return;}
        Kokkos::Array<A,5> sum{};
        for(auto k=rows.offsets(row);k<rows.offsets(row+1);++k){
            const auto c=geometry.template context<R>(rows.faces(k),time);
            const auto flux=physics.boundary_flux(c,condition,model,closure,current);
            if(flux.valid())accumulate(sum,flux.flux,-static_cast<A>(geometry.face_area(static_cast<std::size_t>(c.face))));
        }
        add(output,cell,sum,physics.flags);
    }
};
} // namespace krona::transport::euler::detail