#pragma once
#include <krona/boundary/physical/manager.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/transport/euler/operator.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace euler_test {
namespace eu=krona::transport::euler;
namespace ph=krona::boundary::physical;
namespace rec=krona::numerics::reconstruction;
namespace ri=krona::numerics::riemann;
using Exec=krona::execution::ExecutionSpace;
using Space=typename Exec::memory_space;
using R=krona::ComputeReal;
using S=krona::StorageReal;
using A=krona::AccumulationReal;
using Gas=krona::thermodynamics::IdealGas<R>;
using Model=ph::Compressible<Gas>;
using Boundary=ph::Manager<Model,Exec,R>;
inline Gas gas(){Gas g;g.gas_constant=R{1};return g;}
inline ph::Primitive<R,true> profile(double x,bool affine=false)
{
    return {static_cast<R>(1.+(affine?0.1*x:0.)),{static_cast<R>(3.+(affine?0.2*x:0.)),0,0},static_cast<R>(1.+(affine?0.05*x:0.))};
}
inline krona::mesh::MeshPart mesh(krona::mpi::CommView comm,int start,int owned,int total,bool unlabelled=false,double ly=1.,double lz=1.,bool skew=false)
{
    krona::mesh::MeshPart m;m.rank=comm.rank();m.nprocs=comm.size();m.n_own=owned;m.faces_grouped=true;
    for(int i=0;i<owned;++i)m.cell_gid.push_back(start+i);
    if(start>0)m.cell_gid.push_back(start-1);
    if(start+owned<total)m.cell_gid.push_back(start+owned);
    m.n_cells=static_cast<krona::LocalIndex>(m.cell_gid.size());
    for(auto id:m.cell_gid){m.cell_centroid_x.push_back(static_cast<krona::GeometryReal>(id)+0.5);
        m.cell_centroid_y.push_back(0);m.cell_centroid_z.push_back(0);m.cell_volume.push_back(static_cast<krona::GeometryReal>(ly*lz));}
    auto local=[&](int gid){return static_cast<krona::LocalIndex>(std::find(m.cell_gid.begin(),m.cell_gid.end(),gid)-m.cell_gid.begin());};
    auto face=[&](int owner,krona::LocalIndex neighbour,int patch,double x,double y,double z,double nx,double ny,double nz){
        m.face_owner.push_back(owner);m.face_neigh.push_back(neighbour);m.face_patch.push_back(unlabelled&&patch==2?-1:patch);
        m.face_centroid_x.push_back(static_cast<krona::GeometryReal>(x));m.face_centroid_y.push_back(static_cast<krona::GeometryReal>(y));m.face_centroid_z.push_back(static_cast<krona::GeometryReal>(z));
        m.face_normal_x.push_back(static_cast<krona::GeometryReal>(nx));m.face_normal_y.push_back(static_cast<krona::GeometryReal>(ny));m.face_normal_z.push_back(static_cast<krona::GeometryReal>(nz));m.face_area.push_back(static_cast<krona::GeometryReal>(nx!=0.?ly*lz:ny!=0.?lz:ly));
    };
    for(int i=0;i+1<owned;++i)face(i,i+1,-1,start+i+1.,0,0,1,0,0);
    m.n_internal_faces=static_cast<krona::LocalIndex>(m.face_owner.size());
    if(start>0)face(0,local(start-1),-1,start,0,0,-1,0,0);
    if(start+owned<total)face(owned-1,local(start+owned),-1,start+owned,0,0,1,0,0);
    m.n_inner_faces=static_cast<krona::LocalIndex>(m.face_owner.size());
    if(start==0&&owned>0)face(0,-1,0,0,0,0,-1,0,0);
    if(start+owned==total&&owned>0)face(owned-1,-1,1,total,0,0,1,0,0);
    for(int i=0;i<owned;++i){const double x=start+i+0.5;
        face(i,-1,2,x,-0.5*ly,0,0,-1,0);face(i,-1,2,x,0.5*ly,0,0,1,0);
        face(i,-1,2,x,0,-0.5*lz,0,0,-1);face(i,-1,2,x,0,0.5*lz,0,0,1);}
    if(skew){
        for(std::size_t i=0;i<m.cell_centroid_x.size();++i){m.cell_centroid_x[i]+=static_cast<krona::GeometryReal>(0.2)*m.cell_centroid_y[i];m.cell_centroid_y[i]+=static_cast<krona::GeometryReal>(0.3)*m.cell_centroid_z[i];}
        for(std::size_t f=0;f<m.face_owner.size();++f){
            m.face_centroid_x[f]+=static_cast<krona::GeometryReal>(0.2)*m.face_centroid_y[f];
            m.face_centroid_y[f]+=static_cast<krona::GeometryReal>(0.3)*m.face_centroid_z[f];
            const double nx=m.face_normal_x[f],ny=m.face_normal_y[f]-0.2*nx,nz=m.face_normal_z[f]-0.3*m.face_normal_y[f]+0.06*nx;
            const double length=std::sqrt(nx*nx+ny*ny+nz*nz);m.face_normal_x[f]=static_cast<krona::GeometryReal>(nx/length);
            m.face_normal_y[f]=static_cast<krona::GeometryReal>(ny/length);m.face_normal_z[f]=static_cast<krona::GeometryReal>(nz/length);
            m.face_area[f]*=static_cast<krona::GeometryReal>(length);
        }
    }
    m.n_faces=static_cast<krona::LocalIndex>(m.face_owner.size());
    m.n_cells_g=static_cast<std::uint64_t>(total);m.n_faces_g=static_cast<std::uint64_t>(total==0?0:5*total+1);
    m.n_bfaces_g=m.n_labeled_bfaces_g=static_cast<std::uint64_t>(total==0?0:4*total+2);
    if(unlabelled)m.n_labeled_bfaces_g=2;
    m.patches.resize(3);m.patches[0].name="left";m.patches[1].name="right";m.patches[2].name="sides";
    return m;
}
inline krona::mesh::MeshAuxConnectivity incidence(const krona::mesh::MeshPart& m)
{
    krona::mesh::MeshAuxConnectivity c;c.active_mask=krona::mesh::AuxConnType::CellFaces;
    c.n_faces=m.n_faces;c.n_cells_own=m.n_own;c.n_nodes_own=m.n_nodes_own;c.cell_faces_offsets.push_back(0);
    for(krona::LocalIndex cell=0;cell<m.n_own;++cell){
        for(krona::LocalIndex f=0;f<m.n_faces;++f)
            if(m.face_owner[static_cast<std::size_t>(f)]==cell||m.face_neigh[static_cast<std::size_t>(f)]==cell)c.cell_faces.push_back(f);
        c.cell_faces_offsets.push_back(c.cell_faces.size());
    }return c;
}
struct Grid {
    krona::mesh::MeshPart host;
    krona::mesh::MeshAuxConnectivity cache;
    krona::mesh::compute::MeshStorage<Exec> storage;
    krona::mesh::compute::AuxConnectivityStorage<Exec> connectivity;
    eu::EulerPlan<Exec,R> plan;
    Grid(krona::mpi::CommView comm,const Exec& exec,int start=0,int owned=2,int total=2,bool unlabelled=false,double ly=1.,double lz=1.,bool skew=false)
        :host(mesh(comm,start,owned,total,unlabelled,ly,lz,skew)),cache(incidence(host)),
        storage(host,{krona::mesh::compute::MeshData::FaceCells|krona::mesh::compute::MeshData::CellGeometry|krona::mesh::compute::MeshData::FaceGeometry},exec),
        connectivity(host,cache,exec),plan(host,storage.view(),connectivity.view(),comm,exec) {}
};

template<class Layout=Kokkos::LayoutLeft>
struct Fields {
    krona::fields::FieldStorage<S,1,Exec,Layout> pressure,temperature;
    krona::fields::FieldStorage<S,3,Exec,Layout> velocity;
    krona::fields::FieldStorage<A,1,Exec,Layout> mass,energy;
    krona::fields::FieldStorage<A,3,Exec,Layout> momentum;
    Fields(const Grid& grid,const Exec& exec):
        pressure("p",extended(grid),krona::fields::FieldLocation::Custom,exec),temperature("T",extended(grid),krona::fields::FieldLocation::Custom,exec),
        velocity("u",extended(grid),krona::fields::FieldLocation::Custom,exec),
        mass("r_rho",static_cast<std::size_t>(grid.host.n_own),krona::fields::FieldLocation::Custom,exec),
        energy("r_E",static_cast<std::size_t>(grid.host.n_own),krona::fields::FieldLocation::Custom,exec),
        momentum("r_m",static_cast<std::size_t>(grid.host.n_own),krona::fields::FieldLocation::Custom,exec) {}
    static std::size_t extended(const Grid& g){return static_cast<std::size_t>(g.host.n_cells+g.host.n_faces-g.host.n_inner_faces);}
    auto primitive()const{return eu::primitive_fields(pressure.view(),velocity.view(),temperature.view());}
    auto physical(){return ph::primitive_views(pressure.view(),velocity.view(),temperature.view());}
    auto output(){return eu::conservative_fields(mass.view(),momentum.view(),energy.view());}
    auto reconstruction(const Grid& grid)const{
        return eu::Reconstruction{rec::FirstOrder{},rec::data(primitive()),rec::MeshStencil<Space,R>{grid.storage.view(),grid.connectivity.view()}};
    }
    void zero(const Exec& exec,A value=0){mass.fill(exec,value);momentum.fill(exec,value);energy.fill(exec,value);}
    void initialize(const Grid& grid,const Exec& exec,bool affine=false){
        auto p=pressure.create_host_buffer();auto t=temperature.create_host_buffer();auto u=velocity.create_host_buffer();
        for(std::size_t i=0;i<pressure.size();++i){const auto q=profile(i<grid.host.cell_centroid_x.size()?static_cast<double>(grid.host.cell_centroid_x[i]):0.,affine);
            p(i)=static_cast<S>(q.pressure);t(i)=static_cast<S>(q.temperature);u(i,0)=static_cast<S>(q.velocity.x);u(i,1)=0;u(i,2)=0;}
        krona::execution::complete(exec,[&]{Kokkos::deep_copy(exec,pressure.view().kokkos_view(),p);Kokkos::deep_copy(exec,temperature.view().kokkos_view(),t);Kokkos::deep_copy(exec,velocity.view().kokkos_view(),u);});zero(exec);
    }
    auto snapshot()const{
        Kokkos::View<double*[5],Kokkos::HostSpace> result("snapshot",mass.size());
        const auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},mass.view().kokkos_view());
        const auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},energy.view().kokkos_view());
        const auto m=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},momentum.view().kokkos_view());
        for(std::size_t i=0;i<mass.size();++i){result(i,0)=static_cast<double>(r(i));for(int c=0;c<3;++c)result(i,c+1)=static_cast<double>(m(i,c));result(i,4)=static_cast<double>(e(i));}return result;
    }
};
inline void open_boundaries(Boundary& boundary,bool affine=false)
{
    const auto left=profile(0.,affine);
    boundary.set(0,ph::SupersonicInlet<R>{left.pressure,left.temperature,left.velocity});
    boundary.set(1,ph::SupersonicOutlet{});boundary.set(2,ph::Symmetry{});
}
inline double tolerance(){return 1024.*static_cast<double>(Kokkos::epsilon_v<R>);}
}
