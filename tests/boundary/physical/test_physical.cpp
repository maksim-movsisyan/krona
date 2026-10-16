#include "common/test_support.hpp"
#include <krona/boundary/physical/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/transport/isotropic_diffusion.hpp>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

namespace {
using namespace krona::test;
namespace ph=krona::boundary::physical;
namespace thermo=krona::thermodynamics;
namespace scalar=krona::boundary::scalar;
using Real=krona::ComputeReal;
using Memory=typename Exec::memory_space;
using Array=Kokkos::View<Real*,Memory>;
using Gas=thermo::IdealGas<Real>;
using GasModel=ph::Compressible<Gas,ph::PerfectGasCharacteristics>;
constexpr Real tolerance=Real{512}*(std::numeric_limits<Real>::epsilon()+
    static_cast<Real>(std::numeric_limits<krona::GeometryReal>::epsilon()));

template<class A,class B>
void close(A actual,B expected,const char* message,Real factor=1)
{ near(static_cast<Real>(actual),static_cast<Real>(expected),factor*tolerance*
    std::max(Real{1},Kokkos::abs(static_cast<Real>(expected))),message); }
template<class View>
auto snapshot(const Exec& exec,View v)
{
    auto h=Kokkos::create_mirror(Kokkos::WithoutInitializing,v);
    Kokkos::deep_copy(exec,h,v);exec.fence();return h;
}
template<class V>
void initialize_component(const Exec& exec,V v,Real value)
{
    auto h=Kokkos::create_mirror(Kokkos::WithoutInitializing,v);
    Kokkos::deep_copy(h,Real{-777});
    if (v.extent(0)>=3) { h(0)=value;h(1)=value;h(2)=999; }
    Kokkos::deep_copy(exec,v,h);exec.fence();
}
template<class Views>
void initialize(const Exec& exec,Views fields)
{
    initialize_component(exec,fields.pressure,Real{100000});
    initialize_component(exec,fields.u,Real{100});
    initialize_component(exec,fields.v,Real{20});
    initialize_component(exec,fields.w,Real{30});
    if constexpr (Views::thermal) initialize_component(exec,fields.temperature,Real{300});
}

krona::mesh::MeshPart fixture(bool empty=false)
{
    krona::mesh::MeshPart m;m.faces_grouped=true;
    m.patches={{"iso","wall","BCWall"},{"adiabatic","wall","BCWall"},{"symmetry","","BCSymmetryPlane"},
        {"outlet","","BCOutflow"},{"inlet","","BCInflowSupersonic"},{"empty","","BCWall"}};
    m.n_bfaces_g=6;m.n_labeled_bfaces_g=5;
    if (empty) return m;
    m.n_own=2;m.n_cells=3;m.n_faces=7;m.n_inner_faces=1;m.n_internal_faces=0;
    m.face_owner={0,0,1,0,1,0,1};m.face_neigh={2,-1,-1,-1,-1,-1,-1};
    m.face_patch={-1,0,1,2,3,4,-1};
    m.cell_centroid_x={0,0,0};m.cell_centroid_y={0,0,0};m.cell_centroid_z={0,0,0};m.cell_volume={1,1,1};
    m.face_centroid_x={1,1,1,1,1,1,1};
    m.face_centroid_y={0,krona::GeometryReal{0.25},krona::GeometryReal{0.25},0,0,0,krona::GeometryReal{0.25}};
    m.face_centroid_z={0,0,0,0,0,0,0};m.face_normal_x={1,1,1,1,1,1,1};
    m.face_normal_y={0,0,0,0,0,0,0};m.face_normal_z={0,0,0,0,0,0,0};m.face_area={1,2,3,4,5,6,7};
    return m;
}
constexpr auto selection=krona::mesh::compute::MeshData::FaceCells|krona::mesh::compute::MeshData::Geometry;
GasModel gas_model() { return {Gas{},ph::PerfectGasCharacteristics{}}; }
template<class Manager>
void configure(Manager& manager)
{
    manager.set(0,ph::IsothermalWall<Real>{350,{}});
    manager.set(1,ph::AdiabaticWall<Real>{});manager.set(2,ph::Symmetry{});
    manager.set(3,ph::PressureOutlet<Real>{90000,{}});
    manager.set(4,ph::SupersonicInlet<Real>{120000,280,{-700,0,0}});
    manager.set(5,ph::Symmetry{});manager.set_unlabelled(ph::HeatFluxWall<Real>{10,{}});
}

void thermodynamics_case(const Exec& exec)
{
    auto gas=Gas{};gas.validate();
    Kokkos::View<Real*[12],Memory> result("thermodynamics",1);
    Kokkos::parallel_for("test/thermodynamics",Kokkos::RangePolicy<Exec>(exec,0,1),KOKKOS_LAMBDA(int) {
        const auto q=gas.from_pT(Real{100000},Real{300});
        const auto r=gas.from_rhoe(q.density,q.internal_energy);
        const auto s=gas.from_ps(q.pressure,q.entropy);
        const auto h=gas.from_hs(q.enthalpy,q.entropy);
        const auto d=gas.rhoe_derivatives(q.density,q.internal_energy);
        result(0,0)=r.pressure;result(0,1)=r.temperature;result(0,2)=s.temperature;result(0,3)=h.pressure;
        result(0,4)=q.enthalpy-q.internal_energy;result(0,5)=q.pressure/q.density;
        const Real step=q.density*static_cast<Real>(0.01);
        result(0,6)=(gas.from_rhoe(q.density+step,q.internal_energy).pressure-
            gas.from_rhoe(q.density-step,q.internal_energy).pressure)/(Real{2}*step);
        result(0,7)=d.dp_drho_e;
        const thermo::ThermalConstantDensity<Real> liquid{1000,4200,273};
        result(0,8)=liquid.internal_energy(Real{300});result(0,9)=liquid.enthalpy(Real{100000},Real{300});
        const auto model=ph::Compressible{gas,ph::PerfectGasCharacteristics{}};
        const ph::Primitive<Real,true> primitive{100000,{10,20,30},300};
        const auto restored=model.primitive(model.conservative(primitive));
        result(0,10)=restored.pressure;result(0,11)=restored.temperature;
    });
    const auto h=snapshot(exec,result);
    close(h(0,0),Real{100000},"rho,e -> pressure");close(h(0,1),Real{300},"rho,e -> temperature");
    close(h(0,2),Real{300},"p,s inversion");close(h(0,3),Real{100000},"h,s inversion");
    close(h(0,4),h(0,5),"h=e+p/rho");close(h(0,6),h(0,7),"EOS pressure derivative",Real{32});
    close(h(0,8),Real{113400},"Liquid internal energy reference");close(h(0,9),Real{113500},"Liquid pressure enthalpy term");
    close(h(0,10),Real{100000},"Conservative/primitive round trip");close(h(0,11),Real{300},"Kinetic energy removed before EOS inversion");
}

template<class Layout>
void wall_layout(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    krona::fields::FieldStorage<Real,1,Exec> p("p",11,krona::fields::FieldLocation::Custom,exec),T("T",11,krona::fields::FieldLocation::Custom,exec);
    krona::fields::FieldStorage<Real,3,Exec,Layout> velocity("u",11,krona::fields::FieldLocation::Custom,exec);
    const auto fields=ph::primitive_views(p.view(),velocity.view(),T.view());initialize(exec,fields);
    Array gx("gx",2),gy("gy",2),gz("gz",2);
    Kokkos::deep_copy(exec,gx,Real{0});Kokkos::deep_copy(exec,gy,Real{4});Kokkos::deep_copy(exec,gz,Real{0});
    const auto law=krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}};
    const auto correction=ph::Corrections{krona::boundary::NoCorrection{},ph::ZeroVelocityIncrement{},
        krona::boundary::GradientCorrection{gx,gy,gz}};
    manager.apply_async(fields,law,correction);
    const auto hp=snapshot(exec,fields.pressure),ht=snapshot(exec,fields.temperature);
    const auto hu=snapshot(exec,velocity.view().kokkos_view());
    for (std::size_t i=0;i<3;++i) {
        equal(hp(i),i==2?Real{999}:Real{100000},"Owned and MPI ghost pressure preserved");
        equal(ht(i),i==2?Real{999}:Real{300},"Owned and MPI ghost temperature preserved");
        equal(hu(i,0),i==2?Real{999}:Real{100},"Owned and MPI ghost velocity preserved");
    }
    close(ht(3),Real{350},"Isothermal physical face temperature");close(ht(4),Real{301},"Corrected adiabatic face temperature");
    close(ht(8),Real{296},"Heat flux sign, normal distance and tangential correction");
    close(hu(3,0),Real{0},"No slip x");close(hu(3,1),Real{0},"No slip y");close(hu(3,2),Real{0},"No slip z");
    close(hu(5,0),Real{0},"Symmetry removes normal velocity");close(hu(5,1),Real{20},"Symmetry retains tangential velocity");
    close(hp(6),Real{90000},"Subsonic outlet pressure");close(hp(7),Real{120000},"Supersonic prescribed pressure");
    close(ht(7),Real{280},"Supersonic prescribed temperature");close(hu(7,0),Real{-700},"Supersonic prescribed velocity");
    for (std::size_t i=9;i<11;++i) {
        equal(hp(i),Real{-777},"Trailing pressure untouched");equal(ht(i),Real{-777},"Trailing temperature untouched");
        for (int j=0;j<3;++j) equal(hu(i,j),Real{-777},"Trailing velocity untouched");
    }
}
void walls_case(const Exec& exec) { wall_layout<Kokkos::LayoutLeft>(exec);wall_layout<Kokkos::LayoutRight>(exec); }

struct Capture {
    Kokkos::View<Real*[10],Memory> result;
    KOKKOS_INLINE_FUNCTION void operator()(const krona::boundary::FaceContext<Real>& c,
        const ph::Evaluation<ph::Primitive<Real,true>>& e) const
    {
        const auto i=c.boundary_index;
        result(i,0)=e.face.pressure;result(i,1)=e.face.temperature;
        result(i,2)=e.face.velocity.x;result(i,3)=e.face.velocity.y;result(i,4)=e.face.velocity.z;
        result(i,5)=e.exterior.pressure;result(i,6)=e.exterior.temperature;
        result(i,7)=e.exterior.velocity.x;result(i,8)=e.exterior.velocity.y;result(i,9)=e.exterior.velocity.z;
    }
};
void evaluate_case(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec),T("T",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    const auto fields=ph::primitive_views(p.view(),u.view(),T.view());initialize(exec,fields);
    Kokkos::View<Real*[10],Memory> result("result",6);
    const auto law=krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}};
    manager.evaluate_async(Capture{result},ph::OwnerReader{fields},law);
    const auto h=snapshot(exec,result);
    const auto ht=snapshot(exec,fields.temperature);
    close(h(0,1),Real{350},"Physical wall trace is isothermal");close(h(0,6),Real{300},"Inviscid exterior keeps interior thermal state");
    close(h(0,7),Real{-100},"Inviscid wall mirror normal velocity");
    close(h(0,8),Real{20},"Inviscid wall keeps tangential velocity");
    close(h(2,7),Real{-100},"Symmetry mirror normal velocity");close(h(2,8),Real{20},"Symmetry mirror tangential velocity");
    for (std::size_t i=3;i<9;++i) equal(ht(i),Real{-777},"evaluate_async does not write primitive boundary slots");
}

template<class T> concept HasTemperature=requires(T s){s.temperature;};
void incompressible_case(const Exec& exec)
{
    static_assert(!HasTemperature<ph::Primitive<Real,false>>);
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    const auto model=ph::Incompressible<thermo::ConstantDensity<Real>>{{1000}};
    ph::Manager<decltype(model),Exec> manager(m,mesh.view(),model,exec);
    for (int id=0;id<6;++id) manager.set(id,ph::AdiabaticWall<Real>{});
    manager.set(2,ph::Symmetry{});manager.set(3,ph::PressureOutlet<Real>{-5,{}});
    manager.set(4,ph::VelocityInlet<Real>{{-3,2,1},0});manager.set_unlabelled(ph::AdiabaticWall<Real>{});
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec,Kokkos::LayoutRight> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    const auto fields=ph::primitive_views(p.view(),u.view());initialize(exec,fields);manager.apply_async(fields);
    const auto hp=snapshot(exec,fields.pressure);
    const auto hu=snapshot(exec,fields.u);
    close(hp(6),Real{-5},"Incompressible pressure may be gauge pressure");close(hu(7),Real{-3},"Incompressible velocity inlet");
}

void thermal_liquid_case(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    using Model=ph::Incompressible<thermo::ThermalConstantDensity<Real>,true>;
    ph::Manager<Model,Exec> manager(m,mesh.view(),Model{},exec);
    for (int id=0;id<6;++id) manager.set(id,ph::AdiabaticWall<Real>{});
    manager.set(0,ph::IsothermalWall<Real>{350,{}});
    manager.set(4,ph::VelocityInlet<Real>{{-3,2,1},280});
    manager.set_unlabelled(ph::HeatFluxWall<Real>{10,{}});
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec),T("T",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    const auto fields=ph::primitive_views(p.view(),u.view(),T.view());initialize(exec,fields);
    manager.apply_async(fields,krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}});
    const auto ht=snapshot(exec,fields.temperature);
    close(ht(3),Real{350},"Thermal incompressible isothermal wall");
    close(ht(4),Real{300},"Thermal incompressible adiabatic wall");
    close(ht(7),Real{280},"Thermal incompressible inlet temperature");
    close(ht(8),Real{295},"Thermal incompressible outward heat flux");
}

// Interface test: the boundary layer cannot access a gamma member on this EOS.
struct WrappedGasEOS {
    Gas implementation;
    void validate() const { implementation.validate(); }
    template<class S> KOKKOS_INLINE_FUNCTION auto from_pT(S p,S T) const { return implementation.from_pT(p,T); }
    template<class S> KOKKOS_INLINE_FUNCTION auto from_rhoe(S rho,S e) const { return implementation.from_rhoe(rho,e); }
};
struct WrappedCharacteristics {
    void validate(const WrappedGasEOS&) const {}
    template<class R>
    KOKKOS_INLINE_FUNCTION auto inlet(const WrappedGasEOS& eos,const krona::boundary::FaceContext<R>& c,
        ph::Primitive<R,true> s,ph::TotalConditionsInlet<R> data) const
    { return ph::PerfectGasCharacteristics{}.inlet(eos.implementation,c,s,data); }
    template<class R>
    KOKKOS_INLINE_FUNCTION auto outlet(const WrappedGasEOS& eos,const krona::boundary::FaceContext<R>& c,
        ph::Primitive<R,true> s,R p) const
    { return ph::PerfectGasCharacteristics{}.outlet(eos.implementation,c,s,p); }
};
struct PerOwnerModel {
    static constexpr bool compressible=true,thermal=true,has_characteristics=true;
    Array gas_constants;
    void validate() const { check(gas_constants.extent(0)==2,"Two owned material entries"); }
    template<class R,class C>
    KOKKOS_INLINE_FUNCTION auto at(const krona::boundary::FaceContext<R>& c,const C&) const
    {
        Gas gas;gas.gas_constant=gas_constants(static_cast<std::size_t>(c.owner));
        return ph::Compressible{WrappedGasEOS{gas},WrappedCharacteristics{}};
    }
};
void extension_case(const Exec& exec)
{
    static_assert(thermo::AcousticEOS<WrappedGasEOS,Real>);
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    Array constants("R",2);auto host=Kokkos::create_mirror(constants);
    host(0)=250;host(1)=1000;Kokkos::deep_copy(exec,constants,host);exec.fence();
    ph::Manager<PerOwnerModel,Exec> manager(m,mesh.view(),PerOwnerModel{constants},exec);
    for (int id=0;id<6;++id) manager.set(id,ph::Symmetry{});
    manager.set(0,ph::SupersonicInlet<Real>{100000,300,{},2,{-1,0,0}});
    manager.set(1,ph::SupersonicInlet<Real>{100000,300,{},2,{-1,0,0}});
    manager.set(3,ph::PressureOutlet<Real>{90000,{}});manager.set_unlabelled(ph::Symmetry{});
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec),T("T",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    const auto fields=ph::primitive_views(p.view(),u.view(),T.view());initialize(exec,fields);manager.apply_async(fields);
    const auto hu=snapshot(exec,fields.u);
    close(hu(3),-Real{2}*std::sqrt(static_cast<Real>(1.4)*Real{250}*Real{300}),"Per-owner EOS and Mach conversion");
    close(hu(4),Real{2}*hu(3),"Material provider selects the owner cell without a face array");
    const auto hp=snapshot(exec,fields.pressure);close(hp(6),Real{90000},"Custom characteristic policy with opaque EOS");
}

void characteristic_case(const Exec& exec)
{
    Kokkos::View<Real*[22],Memory> result("characteristics",1);
    const auto model=gas_model();
    Kokkos::parallel_for("test/characteristics",Kokkos::RangePolicy<Exec>(exec,0,1),KOKKOS_LAMBDA(int) {
        krona::boundary::FaceContext<Real> c;c.nx=1;c.distance=1;
        const ph::Primitive<Real,true> inside{100000,{100,20,30},300};
        const auto qi=model.thermodynamic(inside);
        const auto out=ph::evaluate(ph::PressureOutlet<Real>{90000,{}},c,inside,model).face;
        const auto qo=model.thermodynamic(out);
        result(0,0)=out.pressure;result(0,1)=qo.entropy-qi.entropy;
        result(0,2)=out.velocity.x+Real{2}*qo.sound_speed/(model.eos.gamma-Real{1});
        result(0,3)=inside.velocity.x+Real{2}*qi.sound_speed/(model.eos.gamma-Real{1});
        result(0,4)=out.velocity.y;
        const auto choked=ph::evaluate(ph::PressureOutlet<Real>{1000,{}},c,inside,model).face;
        result(0,5)=choked.velocity.x/model.thermodynamic(choked).sound_speed;
        result(0,6)=choked.pressure;
        auto sup=inside;sup.velocity.x=700;
        result(0,7)=ph::evaluate(ph::PressureOutlet<Real>{90000,{}},c,sup,model).face.pressure;
        const ph::Primitive<Real,true> input{100000,{-100,0,0},300};
        const Real T0=input.temperature+input.velocity.norm2()/(Real{2}*model.eos.cp());
        const Real p0=input.pressure*Kokkos::pow(T0/input.temperature,model.eos.gamma/(model.eos.gamma-Real{1}));
        const ph::TotalConditionsInlet<Real> specified{p0,T0,{-1,0,0},false};
        const auto inlet=ph::evaluate(specified,c,input,model).face;
        result(0,8)=inlet.pressure;result(0,9)=inlet.temperature;result(0,10)=inlet.velocity.x;
        const auto qb=model.thermodynamic(inlet),q0=model.eos.from_pT(p0,T0);
        result(0,11)=qb.enthalpy+inlet.velocity.norm2()/Real{2};result(0,12)=q0.enthalpy;
        result(0,13)=qb.entropy-q0.entropy;
        c.nx=Real{2}/3;c.ny=Real{2}/3;c.nz=Real{1}/3;
        const auto oblique=ph::evaluate(specified,c,input,model).face;
        result(0,14)=oblique.velocity.x;
        c.nx=1;c.ny=0;c.nz=0;
        ph::PressureOutlet<Real> reverse{90000,{}};
        reverse.backflow.mode=ph::BackflowMode::TotalConditions;reverse.backflow.reservoir=specified;
        result(0,15)=ph::evaluate(reverse,c,input,model).face.velocity.x;
        const ph::Primitive<Real,true> cold{100000,{-100,0,0},100};
        const auto sonic=ph::evaluate(ph::TotalConditionsInlet<Real>{200000,1000,{-1,0,0},true},c,cold,model).face;
        result(0,16)=sonic.velocity.x/model.thermodynamic(sonic).sound_speed;
        const ph::Incompressible<thermo::ThermalConstantDensity<Real>,true> liquid{{1000,4200,273}};
        ph::PressureOutlet<Real> pressure{-10,{}};pressure.backflow.mode=ph::BackflowMode::DirectionTemperature;
        const Real inv=Real{1}/Kokkos::sqrt(Real{2});pressure.backflow.direction={-inv,inv,0};pressure.backflow.temperature=290;
        const auto back=ph::evaluate(pressure,c,ph::Primitive<Real,true>{0,{-3,4,0},300},liquid).face;
        result(0,17)=back.velocity.x;result(0,18)=back.velocity.y;result(0,19)=back.temperature;result(0,20)=back.pressure;
        const ph::SupersonicInlet<Real> mach{100000,300,{},2,{-1,0,0}};
        const auto imposed=ph::evaluate(mach,c,input,model).face;
        result(0,21)=imposed.velocity.x/model.thermodynamic(imposed).sound_speed;
    });
    const auto h=snapshot(exec,result);
    close(h(0,0),Real{90000},"Subsonic pressure condition");close(h(0,1),Real{0},"Outgoing entropy",Real{64});
    close(h(0,2),h(0,3),"Outgoing acoustic invariant");close(h(0,4),Real{20},"Outgoing tangent velocity");
    close(h(0,5),Real{1},"Outlet sonic limit");check(h(0,6)>Real{1000},"Choked outlet does not impose unreachable back pressure");
    close(h(0,7),Real{100000},"Supersonic outflow ignores prescribed pressure");
    close(h(0,8),Real{100000},"Total inlet exact pressure");close(h(0,9),Real{300},"Total inlet exact static temperature");
    close(h(0,10),Real{-100},"Total inlet exact speed");close(h(0,11),h(0,12),"Total enthalpy");close(h(0,13),Real{0},"Reservoir entropy",Real{64});
    close(h(0,14),Real{-100},"Oblique normal uses full speed in the energy equation");close(h(0,15),Real{-100},"Explicit reservoir backflow");
    close(h(0,16),Real{-1},"Explicit inlet choking");close(h(0,17),Real{-3},"Liquid backflow normal component");
    close(h(0,18),Real{3},"Liquid backflow direction");close(h(0,19),Real{290},"Liquid backflow temperature");
    close(h(0,20),Real{-10},"Liquid backflow pressure");close(h(0,21),Real{-2},"Mach inlet conversion");
}

void relations_case(const Exec& exec)
{
    Kokkos::View<Real*[9],Memory> result("relations",1);
    Kokkos::parallel_for("test/relations",Kokkos::RangePolicy<Exec>(exec,0,1),KOKKOS_LAMBDA(int) {
        krona::boundary::FaceContext<Real> c;c.nx=Real{2}/3;c.ny=Real{2}/3;c.nz=Real{1}/3;c.distance=2;
        const ph::Vector<Real> owner{1,2,3},increment{4,5,6},wall{0,1,0};
        const auto r=ph::slip_relation(c,wall,increment);const auto n=ph::normal(c);
        const auto value=r.face_value(owner),grad=r.normal_gradient(owner);
        result(0,0)=value.dot(n);result(0,1)=wall.dot(n);
        const auto tangential=value-n*value.dot(n),expected=owner+increment-n*n.dot(owner+increment);
        result(0,2)=(tangential-expected).norm2();result(0,3)=(grad-n*grad.dot(n)).norm2();
        const auto no=ph::no_slip_relation(c,wall,increment);
        result(0,4)=(no.face_value(owner)-wall).norm2();
        result(0,5)=(no.normal_gradient(owner)-(wall-owner-increment)*Real{0.5}).norm2();
        const auto iso=ph::wall_temperature_relation(ph::IsothermalWall<Real>{350,{}},c,Real{2},scalar::NoFluxLaw{});
        result(0,6)=iso.normal_gradient(300);
        const auto flux=ph::wall_temperature_relation(ph::HeatFluxWall<Real>{10,{}},c,Real{2},
            krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}});
        result(0,7)=flux.face_value(300);result(0,8)=flux.normal_gradient(300);
    });
    const auto h=snapshot(exec,result);
    close(h(0,0),h(0,1),"Slip prescribed normal velocity");
    for (int j=2;j<=5;++j) close(h(0,j),Real{0},"Coupled velocity value/gradient relation");
    close(h(0,6),Real{24},"Temperature Dirichlet normal derivative");close(h(0,7),Real{292},"Heat-flux relation face value");
    close(h(0,8),Real{-5},"Heat-flux relation gradient");
}

struct RawVisitor {
    Array result;
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const krona::boundary::FaceContext<Real>& c,const Condition&) const
    { result(c.boundary_index)=c.area+c.time+(std::is_same_v<Condition,ph::HeatFluxWall<Real>>?Real{10}:Real{0}); }
};
void visitation_case(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    int calls=0;manager.for_each_patch([&](krona::boundary::PatchRange,const auto&){++calls;});
    equal(calls,6,"Skip locally empty patch and include unlabelled interval");
    Array result("raw",6);manager.for_each_face_async(RawVisitor{result},Real{3});
    const auto h=snapshot(exec,result);
    for (std::size_t i=0;i<6;++i) close(h(i),static_cast<Real>(i+5)+(i==5?Real{10}:Real{0}),"Raw visitation needs no state or flux law");
}

void configuration_case(const Exec& exec)
{
    const auto loaded=ph::read_config(KRONA_PHYSICAL_TEST_DATA);
    std::ifstream input(KRONA_PHYSICAL_TEST_DATA);check(input.is_open(),"Open physical TOML fixture");
    const std::string text{std::istreambuf_iterator<char>{input},std::istreambuf_iterator<char>{}};
    const auto parsed=ph::parse_config(text);
    equal(parsed.patches.size(),std::size_t{6},"Parse physical patch count");
    check(std::holds_alternative<ph::IsothermalWall<double>>(parsed.patches[0].condition),"Top-level type selects physical condition");
    close(std::get<ph::IsothermalWall<double>>(parsed.patches[0].condition).temperature,350,"Physical temperature parameter");
    check(std::holds_alternative<ph::HeatFluxWall<double>>(*loaded.unlabelled),"Unlabelled physical type");
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    auto shuffled=loaded;std::swap(shuffled.patches.front(),shuffled.patches.back());
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),shuffled,exec);
    equal(manager.required_field_size(),std::size_t{9},"Configuration bound by ID and name");
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec),T("T",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    const auto fields=ph::primitive_views(p.view(),u.view(),T.view());initialize(exec,fields);
    manager.apply_async(fields,krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}});
    const auto ht=snapshot(exec,fields.temperature);
    const auto hp=snapshot(exec,fields.pressure);
    close(ht(3),Real{350},"Real file drives wall fill");close(hp(6),Real{90000},"Real file drives outlet fill");
    close(ht(8),Real{295},"Real file drives unlabelled flux fill");
    const auto alias=ph::parse_config("[[boundary_condition]]\npatch_id=0\nname='a'\ntype='INLET_TOTAL_TP'\np0=120000\nT0=350\ndirection=[-1,0,0]\n");
    check(std::holds_alternative<ph::TotalConditionsInlet<double>>(alias.patches[0].condition),"Generator's legacy type and parameter spellings");
}

void empty_case(const Exec& exec)
{
    const auto m=fixture(true);krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    krona::fields::FieldStorage<Real,1,Exec> p("p",0,krona::fields::FieldLocation::CellExtnd,exec),T("T",0,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",0,krona::fields::FieldLocation::CellExtnd,exec);
    manager.apply_async(ph::primitive_views(p.view(),u.view(),T.view()),
        krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}});
    int calls=0;manager.for_each_patch([&](krona::boundary::PatchRange,const auto&){++calls;});equal(calls,0,"Empty local ranges do not dispatch");
    exec.fence();
}

struct TimedWall {
    template<class R,class Model,class Law,class Correction>
    KOKKOS_INLINE_FUNCTION auto operator()(const krona::boundary::FaceContext<R>& c,
        ph::Primitive<R,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction) const
    { return ph::evaluate(ph::AdiabaticWall<R>{{c.time,0,0}},c,inside,model,law,correction); }
};
void custom_case(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec,Real,TimedWall> manager(m,mesh.view(),gas_model(),exec);configure(manager);manager.set(0,TimedWall{});
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec),T("T",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    const auto fields=ph::primitive_views(p.view(),u.view(),T.view());initialize(exec,fields);
    const auto law=krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}};
    manager.apply_async(fields,law,ph::Corrections<>{},Real{4});
    manager.set(0,ph::AdiabaticWall<Real>{{8,0,0}});
    const auto before=snapshot(exec,fields.u);close(before(3),Real{4},"Custom condition sees face time and copied binding");
    manager.apply_async(fields,law);const auto after=snapshot(exec,fields.u);close(after(3),Real{8},"Updated binding used by later submissions");
}

void distributed_case(const Exec& exec)
{
    int rank=0,ranks=0;krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD,&rank),"rank");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD,&ranks),"size");equal(ranks,2,"Use two ranks");
    if(rank==0) walls_case(exec);else empty_case(exec);
    int one=1,total=0;krona::mpi::check(MPI_Allreduce(&one,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD),"join");equal(total,2,"Empty rank joins normally");
}

template<int Which>
struct RejectDevice {
    GasModel model;
    KOKKOS_INLINE_FUNCTION void operator()(int) const
    {
        krona::boundary::FaceContext<Real> c;c.nx=1;c.distance=1;
        ph::Primitive<Real,true> s{100000,{-100,0,0},300};
        if constexpr (Which==0) (void)ph::evaluate(ph::PressureOutlet<Real>{90000,{}},c,s,model);
        if constexpr (Which==1) (void)ph::evaluate(ph::SupersonicInlet<Real>{100000,300,{-10,0,0}},c,s,model);
        if constexpr (Which==2) (void)ph::evaluate(ph::TotalConditionsInlet<Real>{200000,400,{1,0,0},false},c,s,model);
        if constexpr (Which==3) {
            s.temperature=100;
            (void)ph::evaluate(ph::TotalConditionsInlet<Real>{200000,1000,{-1,0,0},false},c,s,model);
        }
    }
};
template<int Which>
void reject_device(const Exec& exec)
{
    Kokkos::parallel_for("test/reject",Kokkos::RangePolicy<Exec>(exec,0,1),RejectDevice<Which>{gas_model()});
    exec.fence();
}
void reject_reverse(const Exec& exec) { reject_device<0>(exec); }
void reject_supersonic(const Exec& exec) { reject_device<1>(exec); }
void reject_direction(const Exec& exec) { reject_device<2>(exec); }
void reject_choking(const Exec& exec) { reject_device<3>(exec); }
void reject_no_temperature(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    const auto model=ph::Incompressible<thermo::ConstantDensity<Real>>{};
    ph::Manager<decltype(model),Exec> manager(m,mesh.view(),model,exec);manager.set(0,ph::IsothermalWall<Real>{300,{}});
}
void reject_no_characteristics(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    const ph::Compressible<Gas> model{Gas{},ph::NoCharacteristics{}};
    ph::Manager<decltype(model),Exec> manager(m,mesh.view(),model,exec);manager.set(0,ph::PressureOutlet<Real>{100000,{}});
}
void reject_missing_law(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    krona::fields::FieldStorage<Real,1,Exec> p("p",9,krona::fields::FieldLocation::CellExtnd,exec),T("T",9,krona::fields::FieldLocation::CellExtnd,exec);
    krona::fields::FieldStorage<Real,3,Exec> u("u",9,krona::fields::FieldLocation::CellExtnd,exec);
    manager.apply_async(ph::primitive_views(p.view(),u.view(),T.view()));
}
void reject_missing_binding(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);
    manager.for_each_patch([](krona::boundary::PatchRange,const auto&){});
}
void reject_conflict(const Exec&)
{ (void)ph::parse_config("[[boundary_condition]]\npatch_id=0\nname='a'\ntype='ADIABATIC_WALL'\n[boundary_condition.scalar.temperature]\ntype='ZERO_GRADIENT'\n"); }
void reject_unknown(const Exec&)
{ (void)ph::parse_config("[[boundary_condition]]\npatch_id=0\nname='a'\ntype='UNKNOWN'\n"); }
void reject_typo(const Exec&)
{ (void)ph::parse_config("[[boundary_condition]]\npatch_id=0\nname='a'\ntype='ISOTHERMAL_WALL'\ntemperature=300\ntemperatur=300\n"); }
void reject_duplicate(const Exec&)
{
    const std::string t="[[boundary_condition]]\npatch_id=0\nname='a'\ntype='SYMMETRY'\n";(void)ph::parse_config(t+t);
}
void reject_name(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    auto config=ph::read_config(KRONA_PHYSICAL_TEST_DATA);config.patches[0].name="wrong";
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),config,exec);
}

void reject_short_field(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    Array p("short",8),u("u",9),v("v",9),w("w",9),T("T",9);
    manager.apply_async(ph::primitive_components(p,u,v,w,T),
        krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}});
}
void reject_short_correction(const Exec& exec)
{
    const auto m=fixture();krona::mesh::compute::MeshStorage<Exec> mesh(m,{.data=selection},exec);
    ph::Manager<GasModel,Exec> manager(m,mesh.view(),gas_model(),exec);configure(manager);
    Array p("p",9),u("u",9),v("v",9),w("w",9),T("T",9),gx("gx",1),gy("gy",2),gz("gz",2);
    const auto correction=ph::Corrections{krona::boundary::NoCorrection{},ph::ZeroVelocityIncrement{},
        krona::boundary::GradientCorrection{gx,gy,gz}};
    manager.apply_async(ph::primitive_components(p,u,v,w,T),
        krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{2}},correction);
}
}

int main(int argc,char** argv)
{
    const std::array cases{Case{"thermodynamics",thermodynamics_case},Case{"walls",walls_case},Case{"evaluate",evaluate_case},
        Case{"incompressible",incompressible_case},Case{"thermal_liquid",thermal_liquid_case},Case{"extension",extension_case},
        Case{"characteristics",characteristic_case},Case{"relations",relations_case},
        Case{"visitation",visitation_case},Case{"configuration",configuration_case},Case{"empty",empty_case},
        Case{"custom",custom_case},Case{"distributed",distributed_case},Case{"reject_reverse",reject_reverse},
        Case{"reject_supersonic",reject_supersonic},Case{"reject_direction",reject_direction},Case{"reject_choking",reject_choking},
        Case{"reject_no_temperature",reject_no_temperature},Case{"reject_no_characteristics",reject_no_characteristics},
        Case{"reject_missing_law",reject_missing_law},Case{"reject_missing_binding",reject_missing_binding},
        Case{"reject_conflict",reject_conflict},Case{"reject_unknown",reject_unknown},Case{"reject_typo",reject_typo},
        Case{"reject_duplicate",reject_duplicate},Case{"reject_name",reject_name},
        Case{"reject_short_field",reject_short_field},Case{"reject_short_correction",reject_short_correction}};
    return krona::test::run(argc,argv,"physical_boundary",cases);
}
