#pragma once

#include "common/line_mesh.hpp"

#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/numerics/reconstruction/stencil_data.hpp>
#include <krona/boundary/physical/manager.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>
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

/// @brief Creates the ideal gas used by references with unit gas constant.
inline Gas gas(){Gas g;g.gas_constant=R{1};return g;}

/// @brief Supplies a constant or affine pressure/velocity/temperature reference state.
inline ph::Primitive<R,true> profile(double x,bool affine=false)
{
    return {static_cast<R>(1.+(affine?0.1*x:0.)),{static_cast<R>(3.+(affine?0.2*x:0.)),0,0},static_cast<R>(1.+(affine?0.05*x:0.))};
}

inline auto mesh(krona::mpi::CommView comm, int start, int owned, int total,
                 bool unlabelled = false, double ly = 1., double lz = 1., bool skew = false) {
    return krona::test::line_mesh(comm, start, owned, total, unlabelled, ly, lz, skew);
}

inline auto incidence(const krona::mesh::host::MeshPart& mesh) {
    return krona::test::line_incidence(mesh);
}

/// @brief Owns host/device geometry, incidence and the Euler plan sharing their lifetime.
struct Grid {
    krona::mesh::host::MeshPart host;
    krona::mesh::host::MeshAuxConnectivity cache;
    krona::mesh::device::MeshStorage<Exec> storage;
    krona::mesh::device::AuxConnectivityStorage<Exec> connectivity;
    eu::EulerPlan<Exec,R> plan;
    Grid(krona::mpi::CommView comm,const Exec& exec,int start=0,int owned=2,int total=2,bool unlabelled=false,double ly=1.,double lz=1.,bool skew=false)
        :host(mesh(comm,start,owned,total,unlabelled,ly,lz,skew)),cache(incidence(host)),
        storage(host,{krona::mesh::device::MeshData::FaceCells|krona::mesh::device::MeshData::CellGeometry|krona::mesh::device::MeshData::FaceGeometry},exec),
        connectivity(cache,exec),plan(host,storage.view(),connectivity.view(),comm,exec) {}
};

/// @brief Owns extended primitive inputs and owned conservative residuals in a selectable layout.
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

    // Adapters borrow primitive inputs; conservative outputs contain owned rows only.
    auto primitive()const{return krona::fields::pack(pressure.view(),velocity.view(),temperature.view());}
    /// @brief Borrows only the real-cell prefix, including MPI ghosts but excluding face slots.
    auto cell_primitives(const Grid& grid) const {
        const std::pair cells{std::size_t{0}, static_cast<std::size_t>(grid.host.n_cells)};
        return krona::fields::pack(
            Kokkos::subview(pressure.view().component(0), cells),
            Kokkos::subview(velocity.view().component(0), cells),
            Kokkos::subview(velocity.view().component(1), cells),
            Kokkos::subview(velocity.view().component(2), cells),
            Kokkos::subview(temperature.view().component(0), cells));
    }
    auto physical(){return ph::primitive_views(pressure.view(),velocity.view(),temperature.view());}
    auto output(){return krona::fields::pack(mass.view(),momentum.view(),energy.view());}
    auto reconstruction(const Grid& grid)const{
        return eu::Reconstruction{rec::face_profiles(
            rec::StencilData<Space, R>{grid.storage.view(), grid.connectivity.view()}, primitive(), rec::FirstOrder{})};
    }
    void zero(const Exec& exec,A value=0){mass.fill_async(exec,value);momentum.fill_async(exec,value);energy.fill_async(exec,value);}
    void initialize(const Grid& grid,const Exec& exec,bool affine=false){
        auto p=pressure.create_host_buffer();auto t=temperature.create_host_buffer();auto u=velocity.create_host_buffer();
        for(std::size_t i=0;i<pressure.size();++i){const auto q=profile(i<grid.host.cell_centroid_x.size()?static_cast<double>(grid.host.cell_centroid_x[i]):0.,affine);
            p(i)=static_cast<S>(q.pressure);t(i)=static_cast<S>(q.temperature);u(i,0)=static_cast<S>(q.velocity.x);u(i,1)=0;u(i,2)=0;}
        krona::execution::complete(exec,[&]{Kokkos::deep_copy(exec,pressure.view().kokkos_view(),p);Kokkos::deep_copy(exec,temperature.view().kokkos_view(),t);Kokkos::deep_copy(exec,velocity.view().kokkos_view(),u);});zero(exec);
    }

    // Collect all five conservative components in a common host comparison layout.
    auto snapshot()const{
        Kokkos::View<double*[5],Kokkos::HostSpace> result("snapshot",mass.size());
        const auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},mass.view().kokkos_view());
        const auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},energy.view().kokkos_view());
        const auto m=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},momentum.view().kokkos_view());
        for(std::size_t i=0;i<mass.size();++i){result(i,0)=static_cast<double>(r(i));for(int c=0;c<3;++c)result(i,c+1)=static_cast<double>(m(i,c));result(i,4)=static_cast<double>(e(i));}return result;
    }
};

/// @brief Assigns an exact supersonic inlet, outlet and transverse symmetry walls.
inline void open_boundaries(Boundary& boundary,bool affine=false)
{
    const auto left=profile(0.,affine);
    boundary.set(0,ph::SupersonicInlet<R>{left.pressure,left.temperature,left.velocity});
    boundary.set(1,ph::SupersonicOutlet{});boundary.set(2,ph::Symmetry{});
}

/// @brief Finalizes registration and completes physical boundary publication with status checking.
template<class B,class F,class... Args>
void publish_boundary(B& boundary,F fields,Args... args)
{
    boundary.finalize();
    ph::Status<Exec> status(boundary.execution_space());
    status.begin_async(); boundary.apply_async(fields,status,args...); status.check();
}

/// @brief Scales comparison tolerance with the configured computation precision.
inline double tolerance(){return 1024.*static_cast<double>(Kokkos::epsilon_v<R>);}
}
