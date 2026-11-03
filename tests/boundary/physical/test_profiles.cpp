#include "common/test_support.hpp"
#include <krona/boundary/physical/manager.hpp>
#include <krona/boundary/scalar/manager.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/transport/diffusion/isotropic.hpp>
#include <krona/transport/euler/boundary.hpp>
#include <array>
#include <limits>
#include <type_traits>

namespace {
using namespace krona::test;
namespace bc=krona::boundary;
namespace ph=bc::physical;
namespace sc=bc::scalar;
namespace pr=bc::profile;
namespace grad=krona::numerics::gradient;
namespace thermo=krona::thermodynamics;
using R=krona::ComputeReal;
using Space=typename Exec::memory_space;
using Array=Kokkos::View<R*,Space>;
using Model=ph::Compressible<thermo::IdealGas<R>,ph::PerfectGasCharacteristics>;
constexpr auto mesh_data=krona::mesh::device::MeshData::FaceCells|krona::mesh::device::MeshData::Geometry;
constexpr R tol=static_cast<R>(1e-5);

template<class View> auto snapshot(const Exec& exec,View view)
{ auto host=Kokkos::create_mirror(Kokkos::WithoutInitializing,view); Kokkos::deep_copy(exec,host,view);exec.fence();return host; }
krona::mesh::host::MeshPart fixture(bool empty=false)
{
    krona::mesh::host::MeshPart m;m.faces_grouped=true;
    m.patches={{"first","",""},{"second","",""},{"empty","",""}};
    m.n_bfaces_g=4;m.n_labeled_bfaces_g=3;if(empty)return m;
    m.n_own=2;m.n_cells=3;m.n_faces=4;
    m.face_owner={0,1,0,1};m.face_neigh={-1,-1,-1,-1};m.face_patch={0,0,1,-1};
    m.cell_centroid_x={0,2,4};m.cell_centroid_y=m.cell_centroid_z={0,0,0};m.cell_volume={1,1,1};
    m.face_centroid_x={1,3,2,4};m.face_centroid_y={0.5,-0.5,0.25,0.75};m.face_centroid_z={0,0,0,0};
    m.face_normal_x={1,1,1,1};m.face_normal_y=m.face_normal_z={0,0,0,0};m.face_area={2,3,4,5};
    return m;
}
struct Fields {
    Array p,u,v,w,T;
    Fields(const Exec& exec,std::size_t n):p("p",n),u("u",n),v("v",n),w("w",n),T("T",n)
    {
        Kokkos::deep_copy(exec,p,R{100000});Kokkos::deep_copy(exec,u,R{100});
        Kokkos::deep_copy(exec,v,R{20});Kokkos::deep_copy(exec,w,R{30});Kokkos::deep_copy(exec,T,R{300});
    }
    auto views() const { return ph::primitive_components(p,u,v,w,T); }
};
template<class B> void rest(B& boundary)
{
    boundary.set(1,ph::Symmetry{});boundary.set(2,ph::Symmetry{});boundary.set_unlabelled(ph::AdiabaticWall<R>{});
}
struct Temperature {
    template<class C>KOKKOS_INLINE_FUNCTION auto operator()(const C& c)const
    { using T=std::remove_cvref_t<decltype(c.time)>;return T{350}+T{10}*c.y+T{5}*c.time; }
};
struct Velocity {
    template<class C>KOKKOS_INLINE_FUNCTION auto operator()(const C& c)const
    { using T=std::remove_cvref_t<decltype(c.time)>;return ph::Vector<T>{T{0},c.time,c.y}; }
};
struct Flux {
    template<class C>KOKKOS_INLINE_FUNCTION auto operator()(const C& c)const
    { using T=std::remove_cvref_t<decltype(c.time)>;return T{8}+T{2}*c.time; }
};
struct NoTemperatureOnFirstPatch {
    KOKKOS_INLINE_FUNCTION R operator()(const bc::FaceContext<R>& c)const
    { if(c.face<2)Kokkos::abort("Isothermal filling must not read temperature correction");return 0; }
};
struct Capture {
    Kokkos::View<R*[10],Space> values;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<R>& c,const ph::Evaluation<ph::Primitive<R,true>>& s)const
    {
        const auto b=c.boundary_index;
        values(b,0)=s.face.pressure;values(b,1)=s.face.velocity.x;values(b,2)=s.face.velocity.y;
        values(b,3)=s.face.velocity.z;values(b,4)=s.face.temperature;
        values(b,5)=s.exterior.pressure;values(b,6)=s.exterior.velocity.x;values(b,7)=s.exterior.velocity.y;
        values(b,8)=s.exterior.velocity.z;values(b,9)=s.exterior.temperature;
    }
};
void walls_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    const auto wall=ph::isothermal_wall<R>(Temperature{},Velocity{});
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);boundary.set(0,wall);boundary.finalize();
    Fields fields(exec,layout.required_field_size());Kokkos::View<R*[10],Space> result("traces",4);
    const ph::Corrections correction{grad::NoCorrection{},ph::ZeroVelocityIncrement{},NoTemperatureOnFirstPatch{}};
    ph::Status<Exec> status(exec);status.begin_async();
    boundary.apply_async(fields.views(),status,sc::NoFluxLaw{},correction,R{2});
    boundary.evaluate_async(Capture{result},ph::OwnerReader{fields.views()},status,sc::NoFluxLaw{},correction,R{2});status.check();
    const auto T=snapshot(exec,fields.T);const auto q=snapshot(exec,result);
    near(T(3),R{365},tol,"Analytic wall temperature at first face");near(T(4),R{355},tol,"Analytic wall temperature at second face");
    for(std::size_t i=0;i<3;++i)near(T(i),R{300},tol,"Owned and ghost temperatures remain unchanged");
    near(q(0,1),R{0},tol,"No-slip face normal velocity");near(q(0,2),R{2},tol,"Profiled tangential wall velocity");
    near(q(0,3),R{0.5},tol,"Wall profile uses face coordinates");
    near(q(0,6),R{-100},tol,"Inviscid mirror reverses relative normal velocity");
    near(q(0,7),R{20},tol,"Inviscid tangential velocity stays interior");near(q(0,9),R{300},tol,"Wall temperature does not alter inviscid thermal state");
}
void arrays_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);const auto range=layout.catalog().patches()[0].range;
    Kokkos::View<float*[2],Kokkos::LayoutRight,Space> temperature("strided",2);
    auto component=Kokkos::subview(temperature,Kokkos::ALL(),1);
    Kokkos::View<const float*,Kokkos::LayoutStride,Space> readonly=component;
    Kokkos::View<ph::Vector<R>*,Space> velocity("velocity",2);
    const auto wall=ph::isothermal_wall<R>(pr::patch_values(range,readonly),pr::patch_values(range,velocity));
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);boundary.set(0,wall);boundary.finalize();
    Fields fields(exec,layout.required_field_size());
    Kokkos::deep_copy(exec,temperature,-77.0f);Kokkos::deep_copy(exec,component,360.0f);
    Kokkos::deep_copy(exec,velocity,ph::Vector<R>{R{0},R{3},R{4}});
    ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
    near(snapshot(exec,fields.T)(3),R{360},tol,"Const strided scalar profile");near(snapshot(exec,fields.v)(4),R{3},tol,"Vector patch profile");
    Kokkos::deep_copy(exec,component,380.0f);Kokkos::deep_copy(exec,velocity,ph::Vector<R>{R{0},R{5},R{6}});
    status.begin_async();boundary.apply_async(fields.views(),status);status.check();
    near(snapshot(exec,fields.T)(4),R{380},tol,"Updated borrowed profile buffer");near(snapshot(exec,fields.v)(3),R{5},tol,"Updated vector buffer");
    check(boundary.finalized(),"Content update needs no refinalization");
    near(snapshot(exec,temperature)(0,0),-77.0f,1e-6f,"Unselected input component preserved");
}
void heat_flux_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);const auto wall=ph::heat_flux_wall<R>(Flux{});
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);
    boundary.set(0,wall);boundary.set_unlabelled(wall);boundary.finalize();
    Fields fields(exec,layout.required_field_size());Array gx("gx",2),gy("gy",2),gz("gz",2);
    Kokkos::deep_copy(exec,gx,R{0});Kokkos::deep_copy(exec,gy,R{4});Kokkos::deep_copy(exec,gz,R{0});
    const ph::Corrections correction{grad::NoCorrection{},ph::ZeroVelocityIncrement{},grad::GradientCorrection{grad::cell_gradients(gx,gy,gz)}};
    const auto law=krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<R>{2}};
    ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status,law,correction,R{2});status.check();
    const auto T=snapshot(exec,fields.T);
    near(T(3),R{296},tol,"Flux-to-gradient conversion plus tangential correction");near(T(4),R{292},tol,"Face-specific tangential offset");
    near(T(6),R{291},tol,"Unlabelled heat-flux profile");
}
struct InletVelocity {
    KOKKOS_INLINE_FUNCTION ph::Vector<R> operator()(const bc::FaceContext<R>& c)const { return {-10,c.time,1}; }
};
void liquid_case(const Exec& exec)
{
    using Liquid=ph::Incompressible<thermo::ThermalConstantDensity<R>,true>;
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);const auto inlet=ph::velocity_inlet<R>(InletVelocity{},Temperature{});
    ph::Manager<Liquid,Exec,R,decltype(inlet)> boundary(layout,Liquid{});rest(boundary);boundary.set(0,inlet);boundary.finalize();
    Fields fields(exec,layout.required_field_size());Kokkos::deep_copy(exec,fields.p,R{-5});
    ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status,sc::NoFluxLaw{},ph::Corrections<>{},R{2});status.check();
    near(snapshot(exec,fields.p)(3),R{-5},tol,"Incompressible pressure is extrapolated, including negative gauge values");
    near(snapshot(exec,fields.u)(3),R{-10},tol,"Velocity inlet profile");near(snapshot(exec,fields.T)(3),R{365},tol,"Velocity inlet temperature profile");
}

template<class T>
void direct_factories(const Exec& exec)
{
    Kokkos::View<T*[8],Space> difference("factory_difference",1);
    Kokkos::parallel_for("test/physical_factory_device",Kokkos::RangePolicy<Exec>(exec,0,1),KOKKOS_LAMBDA(int) {
        bc::FaceContext<T> c;c.nx=1;c.distance=1;c.area=1;c.time=2;c.x=1;c.y=T{0.5};
        const auto model=ph::Compressible{thermo::IdealGas<T>{},ph::PerfectGasCharacteristics{}};
        ph::Primitive<T,true> inside{100000,{100,20,30},300};
        const auto outlet=ph::evaluate(ph::pressure_outlet<T>(pr::Constant<T>{90000}),c,inside,model);
        const auto base=ph::evaluate(ph::PressureOutlet<T>{90000,{}},c,inside,model);
        difference(0,0)=outlet.face.pressure-base.face.pressure;difference(0,1)=outlet.face.velocity.x-base.face.velocity.x;
        inside.velocity={-50,0,0};
        const auto total=ph::total_conditions_inlet<T>(pr::Constant<T>{130000},T{330},ph::Vector<T>{-2,0,0});
        const auto a=ph::evaluate(total,c,inside,model);
        const auto b=ph::evaluate(ph::TotalConditionsInlet<T>{130000,330,{-1,0,0},false},c,inside,model);
        difference(0,2)=a.face.velocity.x-b.face.velocity.x;
        const auto back=ph::total_backflow<T>(total);
        const auto reversed=ph::evaluate(ph::pressure_outlet<T>(T{90000},back),c,inside,model);
        difference(0,3)=reversed.face.pressure-a.face.pressure;
        const auto sup=ph::evaluate(ph::supersonic_inlet<T>(T{120000},T{280},ph::Vector<T>{-700,0,0}),c,inside,model);
        difference(0,4)=sup.face.velocity.x-T{-700};
        const auto mach=ph::evaluate(ph::supersonic_mach_inlet<T>(T{120000},T{280},T{2},ph::Vector<T>{-3,0,0}),c,inside,model);
        const auto numeric=ph::evaluate(ph::SupersonicInlet<T>{120000,280,{},2,{-1,0,0}},c,inside,model);
        difference(0,5)=mach.face.velocity.x-numeric.face.velocity.x;
        const auto liquid=ph::Incompressible<thermo::ThermalConstantDensity<T>,true>{};
        inside.velocity={-5,1,0};
        const auto reverse=ph::directional_backflow<T>(ph::Vector<T>{-2,2,0},T{310});
        const auto liq=ph::evaluate(ph::pressure_outlet<T>(T{-10},reverse),c,inside,liquid);
        difference(0,6)=liq.face.velocity.y-T{5};difference(0,7)=liq.face.temperature-T{310};
    });
    const auto h=snapshot(exec,difference);
    for(int c=0;c<8;++c)near(h(0,c),T{0},static_cast<T>(1e-3),"Profiled and numeric characteristic closures agree");
}
void factories_case(const Exec& exec){direct_factories<float>(exec);direct_factories<double>(exec);}

void shared_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);ph::Manager<Model,Exec> physical(layout,Model{});
    sc::Manager<Exec,R> scalar(layout);
    for(krona::mesh::PatchId p=0;p<3;++p){physical.set(p,ph::IsothermalWall<R>{350,{}});scalar.set(p,sc::FixedValue<R>{500});}
    physical.set_unlabelled(ph::AdiabaticWall<R>{});scalar.set_unlabelled(sc::NormalGradient<R>{0});physical.finalize();scalar.finalize();
    check(&physical.catalog()==&scalar.catalog(),"Scalar and physical managers share one catalog");
    Fields fields(exec,layout.required_field_size());Array q("scalar",layout.required_field_size());Kokkos::deep_copy(exec,q,R{20});
    ph::Status<Exec> ps(exec);sc::Status<Exec> ss(exec);ps.begin_async();ss.begin_async();
    physical.apply_async(fields.views(),ps);scalar.apply_component_async(q,ss);ps.check();ss.check();
    near(snapshot(exec,fields.T)(3),R{350},tol,"Independent physical field");near(snapshot(exec,q)(3),R{500},tol,"Independent scalar field");
}
void lambda_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);const R initial=350;
    const auto provider=KOKKOS_LAMBDA(const bc::FaceContext<R>& c){return initial+c.time;};
    const auto wall=ph::isothermal_wall<R>(provider);
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);boundary.set(0,wall);boundary.finalize();
    Fields fields(exec,layout.required_field_size());ph::Status<Exec> status(exec);status.begin_async();
    boundary.apply_async(fields.views(),status,sc::NoFluxLaw{},ph::Corrections<>{},R{2});status.check();
    near(snapshot(exec,fields.T)(3),R{352},tol,"Captured lambda provider");
    boundary.set(0,wall);boundary.finalize(); // No assignment operator is required on the lambda.
    boundary.set(0,ph::IsothermalWall<R>{360,{}});boundary.finalize();
    status.begin_async();boundary.apply_async(fields.views(),status);status.check();near(snapshot(exec,fields.T)(3),R{360},tol,"Replace profiled alternative with a numeric condition");
    ph::Configuration config;
    for(std::size_t p=0;p<layout.catalog().patches().size();++p)
        config.patches.push_back({static_cast<std::int64_t>(p),layout.catalog().patches()[p].name,ph::IsothermalWall<double>{370,{}}});
    config.unlabelled=ph::AdiabaticWall<double>{};boundary.configure(config);boundary.finalize();
    status.begin_async();boundary.apply_async(fields.views(),status);status.check();near(snapshot(exec,fields.T)(3),R{370},tol,"Configuration assignment works with a nonassignable lambda alternative");
}
struct UnusedThermal {
    template<class C>KOKKOS_INLINE_FUNCTION R operator()(const C&)const
    { Kokkos::abort("Inviscid mechanical mirror must not sample thermal parameters"); return 0; }
};
void mechanical_projection_case(const Exec& exec)
{
    Array output("mirror",2);
    Kokkos::parallel_for("test/mechanical_projection",Kokkos::RangePolicy<Exec>(exec,0,1),KOKKOS_LAMBDA(int){
        bc::FaceContext<R> c;c.nx=1;c.distance=1;
        const ph::Primitive<R,true> inside{100000,{100,20,30},300};
        const auto wall=ph::heat_flux_wall<R>(UnusedThermal{},ph::Vector<R>{});
        const auto iso=ph::isothermal_wall<R>(UnusedThermal{},ph::Vector<R>{});
        const auto a=krona::transport::euler::PreparedExterior{}(c,wall,inside,inside,Model{});
        const auto b=krona::transport::euler::EvaluatedExterior<>{}(c,iso,inside,inside,Model{});
        output(0)=a.velocity.x;output(1)=b.velocity.x;
    });
    const auto h=snapshot(exec,output);near(h(0),R{-100},tol,"Prepared mirror bypasses wall heat flux");near(h(1),R{-100},tol,"Evaluated mirror bypasses wall temperature");
}
struct Counter {
    Kokkos::View<int*,Space> calls;
    KOKKOS_INLINE_FUNCTION R operator()(const bc::FaceContext<R>& c)const {++calls(c.boundary_index);return 350;}
};
void once_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);Kokkos::View<int*,Space> calls("calls",4);
    const auto wall=ph::isothermal_wall<R>(Counter{calls});ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});
    rest(boundary);boundary.set(0,wall);boundary.finalize();Fields fields(exec,layout.required_field_size());
    ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
    const auto h=snapshot(exec,calls);equal(h(0),1,"One parameter sample per first face");equal(h(1),1,"One sample per second face");equal(h(2),0,"Unassigned alternative is not sampled");
}
void empty_case(const Exec& exec)
{
    const auto host=fixture(true);krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);
    bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);Array empty("empty",0);
    const auto wall=ph::isothermal_wall<R>(pr::patch_values(layout.catalog().patches()[0].range,empty));
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);boundary.set(0,wall);boundary.finalize();
    Fields fields(exec,0);ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
}
void distributed_case(const Exec& exec)
{
    int rank=0; krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD,&rank),"rank");
    if(rank==0)arrays_case(exec);else empty_case(exec);
}

struct Invalid {
    KOKKOS_INLINE_FUNCTION R operator()(const bc::FaceContext<R>&)const{return Kokkos::quiet_NaN_v<R>;}
};
void invalid_profile_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    const auto wall=ph::isothermal_wall<R>(Invalid{});ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);boundary.set(0,wall);boundary.finalize();
    Fields fields(exec,layout.required_field_size());ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
}
void invalid_direction_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    const auto inlet=ph::total_conditions_inlet<R>(R{130000},R{330},pr::Constant{ph::Vector<R>{}});
    ph::Manager<Model,Exec,R,decltype(inlet)> boundary(layout,Model{});rest(boundary);boundary.set(0,inlet);boundary.finalize();
    Fields fields(exec,layout.required_field_size());ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
}
void dirty_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    ph::Manager<Model,Exec,R> boundary(layout,Model{});rest(boundary);boundary.set(0,ph::IsothermalWall<R>{350,{}});boundary.finalize();
    boundary.set(0,ph::IsothermalWall<R>{360,{}});boundary.for_each_patch([](bc::PatchRange,const auto&){});
}
void short_array_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);Array data("short",1);
    const auto wall=ph::isothermal_wall<R>(pr::patch_values(layout.catalog().patches()[0].range,data));
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});boundary.set(0,wall);
}
void wrong_range_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);Array data("one",1);
    const auto wall=ph::isothermal_wall<R>(pr::patch_values(layout.catalog().unlabelled(),data));
    ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});boundary.set(1,wall);
}
void missing_law_case(const Exec& exec)
{
    const auto host=fixture(true);krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    const auto wall=ph::heat_flux_wall<R>(Flux{});ph::Manager<Model,Exec,R,decltype(wall)> boundary(layout,Model{});rest(boundary);boundary.set(0,wall);boundary.finalize();
    Fields fields(exec,0);ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);
}
void incompatible_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    const auto inlet=ph::velocity_inlet<R>(InletVelocity{},Temperature{});ph::Manager<Model,Exec,R,decltype(inlet)> boundary(layout,Model{});boundary.set(0,inlet);
}
void interior_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    ph::Manager<Model,Exec,R> boundary(layout,Model{});rest(boundary);boundary.set(0,ph::Symmetry{});boundary.finalize();Fields fields(exec,layout.required_field_size());
    Kokkos::deep_copy(exec,fields.p,R{-1});ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
}
struct HugeCorrection {
    KOKKOS_INLINE_FUNCTION double operator()(const bc::FaceContext<double>&)const{return 1e100;}
};
void output_case(const Exec& exec)
{
    using Liquid=ph::Incompressible<thermo::ConstantDensity<double>,false>;
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,double> layout(host,mesh.view(),exec);
    ph::Manager<Liquid,Exec,double> boundary(layout,Liquid{});for(int p=0;p<3;++p)boundary.set(p,ph::Symmetry{});boundary.set_unlabelled(ph::Symmetry{});boundary.finalize();
    Kokkos::View<float*,Space> p("p",7),u("u",7),v("v",7),w("w",7);Kokkos::deep_copy(exec,p,100.0f);Kokkos::deep_copy(exec,u,1.0f);Kokkos::deep_copy(exec,v,2.0f);Kokkos::deep_copy(exec,w,3.0f);
    const ph::Corrections correction{HugeCorrection{},ph::ZeroVelocityIncrement{},grad::NoCorrection{}};
    ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(ph::primitive_components(p,u,v,w),status,sc::NoFluxLaw{},correction);
    exec.fence();equal(snapshot(exec,u)(3),1.0f,"An unrepresentable component skips the whole face write");status.check();
}
struct CheckedCustom {
    template<class Real,class M,class L,class C,class E>
    KOKKOS_INLINE_FUNCTION auto operator()(const bc::FaceContext<Real>&,ph::Primitive<Real,M::thermal> inside,
        const M&,const L&,const C&,const E& errors)const
    { errors.fail(ph::Error::State,"Custom boundary rejected its state");return ph::Evaluation{inside,inside}; }
};
void checked_custom_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    ph::Manager<Model,Exec,R,CheckedCustom> boundary(layout,Model{});rest(boundary);boundary.set(0,CheckedCustom{});boundary.finalize();
    Fields fields(exec,layout.required_field_size());ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
}
void backflow_case(const Exec& exec)
{
    const auto host=fixture();krona::mesh::device::MeshStorage<Exec> mesh(host,{.data=mesh_data},exec);bc::BoundaryLayout<Exec,R> layout(host,mesh.view(),exec);
    ph::Manager<Model,Exec,R> boundary(layout,Model{});for(int p=0;p<3;++p)boundary.set(p,ph::PressureOutlet<R>{90000,{}});boundary.set_unlabelled(ph::PressureOutlet<R>{90000,{}});boundary.finalize();
    Fields fields(exec,layout.required_field_size());Kokkos::deep_copy(exec,fields.u,R{-10});ph::Status<Exec> status(exec);status.begin_async();boundary.apply_async(fields.views(),status);status.check();
}
} // namespace
int main(int argc,char**argv)
{
    constexpr std::array cases{Case{"walls",walls_case},Case{"arrays",arrays_case},Case{"heat_flux",heat_flux_case},Case{"liquid",liquid_case},
        Case{"factories",factories_case},Case{"shared",shared_case},Case{"lambda",lambda_case},Case{"mechanical_projection",mechanical_projection_case},Case{"once",once_case},Case{"empty",empty_case},Case{"distributed",distributed_case},
        Case{"invalid_profile",invalid_profile_case},Case{"invalid_direction",invalid_direction_case},Case{"dirty",dirty_case},Case{"short_array",short_array_case},
        Case{"wrong_range",wrong_range_case},Case{"missing_law",missing_law_case},Case{"incompatible",incompatible_case},Case{"interior",interior_case},Case{"output",output_case},Case{"checked_custom",checked_custom_case},Case{"backflow",backflow_case}};
    return krona::test::run(argc,argv,"physical_profiles",cases);
}
