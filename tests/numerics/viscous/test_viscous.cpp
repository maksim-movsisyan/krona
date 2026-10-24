#include <common/test_support.hpp>
#include <krona/numerics/viscous/flux.hpp>
#include <krona/numerics/riemann/types.hpp>
#include <krona/transport/properties/properties.hpp>

#include <array>
#include <limits>

namespace v=krona::numerics::viscous;
namespace r=krona::numerics::riemann;
namespace p=krona::transport::properties;
using krona::test::Exec;
using krona::test::check;
using krona::test::near;
using Vector=r::Vector<double>;
using Flux=r::Conservative<double>;

void zero_stress(v::SymmetricStress<double> s)
{
    for(double value:{s.xx,s.yy,s.zz,s.xy,s.xz,s.yz})near(value,0.,0.,"zero stress");
}
void vector_near(v::Traction<double> value,Vector reference,double tolerance=1.e-12)
{
    near(value.x,reference.x,tolerance,"traction x");
    near(value.y,reference.y,tolerance,"traction y");
    near(value.z,reference.z,tolerance,"traction z");
}
void shear(const Exec&)
{
    v::VelocityGradient<double> g;g.du_dy=3.;
    const v::NewtonianStress law;
    const auto stress=law(g,2.);
    check(stress.valid(),"shear stress valid");
    near(stress.value.xy,6.,0.,"tau_xy=mu*du/dy");
    near(stress.value.xx,0.,0.,"no normal stress in shear");
    const auto direct=law.traction(g,Vector{0,1,0},2.);
    const auto projected=v::project_traction(stress,Vector{0,1,0});
    check(direct.valid()&&projected.valid(),"traction paths valid");
    vector_near(direct.value,{6,0,0});vector_near(projected.value,{6,0,0});
    const auto phi=law.dissipation(g,2.);
    check(phi.valid(),"shear dissipation valid");near(phi.value,18.,0.,"Phi=mu*shear_rate^2");
}
void rigid_rotation(const Exec&)
{
    v::VelocityGradient<double> g;g.du_dy=-2.;g.dv_dx=2.;
    const auto stress=v::NewtonianStress{}(g,3.,4.);
    check(stress.valid(),"rotation stress valid");zero_stress(stress.value);
    near(v::NewtonianStress{}.dissipation(g,3.,4.).value,0.,0.,"rigid rotation does not dissipate");
}
void dilation(const Exec&)
{
    v::VelocityGradient<double> g;g.du_dx=g.dv_dy=g.dw_dz=2.;
    const auto stokes=v::NewtonianStress{}(g,3.);
    check(stokes.valid(),"Stokes dilation valid");zero_stress(stokes.value);
    const auto bulk=v::NewtonianStress{}(g,3.,5.);
    check(bulk.valid(),"bulk stress valid");
    near(bulk.value.xx,30.,0.,"bulk xx");near(bulk.value.yy,30.,0.,"bulk yy");near(bulk.value.zz,30.,0.,"bulk zz");
    near(v::NewtonianStress{}.dissipation(g,3.,5.).value,180.,0.,"bulk dissipation=zeta*theta^2");
    g.du_dx=2.;g.dv_dy=g.dw_dz=-1.;
    const auto extension=v::NewtonianStress{}(g,3.);
    near(extension.value.xx,12.,0.,"extension xx");near(extension.value.yy,-6.,0.,"extension yy");
    near(v::NewtonianStress{}.dissipation(g,3.).value,36.,0.,"deviatoric extension dissipates");
}
void fourier(const Exec&)
{
    const auto heat=v::FourierHeatFlux{}.operator()<double>(Vector{2,-3,4},Vector{0.6,0.8,0},5.);
    check(heat.valid(),"Fourier flux valid");near(heat.value,6.,1.e-14,"outward Fourier sign");
    const auto opposite=v::FourierHeatFlux{}.operator()<double>(Vector{2,-3,4},Vector{-0.6,-0.8,0},5.);
    near(opposite.value,-heat.value,1.e-14,"normal reversal reverses heat flux");
    near(v::FourierHeatFlux{}.operator()<double>(Vector{2,-3,4},Vector{1,0,0},0.).value,0.,0.,"zero conductivity");
}
void energy(const Exec&)
{
    v::VelocityGradient<double> g;g.du_dy=3.;
    const Vector n{0,1,0},u{4,1,0};
    const auto traction=v::NewtonianStress{}.traction(g,n,2.);
    const auto heat=v::FourierHeatFlux{}.operator()<double>(Vector{0,-1,0},n,5.);
    const auto surface=v::surface_transport(traction,heat);
    const auto outward=v::outward_flux<Flux>(surface,u),rhs=v::rhs_flux<Flux>(surface,u);
    check(outward.valid()&&rhs.valid(),"energy adapters valid");
    near(outward.value.mass,0.,0.,"zero viscous mass flux");
    near(outward.value.momentum.x,-6.,0.,"outward momentum=-traction");
    near(outward.value.energy,-19.,0.,"outward energy=q_n-u*t");
    near(rhs.value.energy,19.,0.,"RHS energy has opposite sign");
    near(rhs.value.momentum.x,6.,0.,"RHS momentum=traction");
    const auto integrated=v::integrated_outward_flux<Flux>(surface,u,2.);
    near(integrated.value.momentum.x,-12.,0.,"separate face area");near(integrated.value.energy,-38.,0.,"integrated energy");
    const auto reversed=v::surface_transport(v::NewtonianStress{}.traction(g,-n,2.),
        v::FourierHeatFlux{}.operator()<double>(Vector{0,-1,0},-n,5.));
    near(v::outward_flux<Flux>(reversed,u).value.energy,-outward.value.energy,0.,"outward energy reversal");
    const auto moving=v::outward_flux<Flux>(surface,u+Vector{2,0,0});
    near(moving.value.energy-outward.value.energy,-12.,0.,"energy change is minus boost dot traction");
}

using Matrix=std::array<std::array<double,3>,3>;
Vector rotate(const Matrix& q,Vector x)
{
    return {q[0][0]*x.x+q[0][1]*x.y+q[0][2]*x.z,
            q[1][0]*x.x+q[1][1]*x.y+q[1][2]*x.z,
            q[2][0]*x.x+q[2][1]*x.y+q[2][2]*x.z};
}
Matrix rotate_tensor(const Matrix& q,const Matrix& a)
{
    Matrix b{};
    for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)
        for(std::size_t k=0;k<3;++k)for(std::size_t l=0;l<3;++l)b[i][j]+=q[i][k]*a[k][l]*q[j][l];
    return b;
}
v::VelocityGradient<double> gradient(const Matrix& a)
{ return v::velocity_gradient<double>(Vector{a[0][0],a[0][1],a[0][2]},Vector{a[1][0],a[1][1],a[1][2]},Vector{a[2][0],a[2][1],a[2][2]}); }
void covariance(const Exec&)
{
    const Matrix q{{{0.6,-0.8,0},{0.8,0.6,0},{0,0,1}}},a{{{1,2,-3},{4,-2,5},{-6,7,3}}};
    const Vector n{0.6,0,0.8},grad_t{3,-2,4};
    const auto g=gradient(a),rotated_g=gradient(rotate_tensor(q,a));
    const auto t=v::NewtonianStress{}.traction(g,n,0.9,0.2);
    const auto rt=v::NewtonianStress{}.traction(rotated_g,rotate(q,n),0.9,0.2);
    check(t.valid()&&rt.valid(),"rotated tractions valid");
    vector_near(rt.value,rotate(q,Vector{t.value.x,t.value.y,t.value.z}),2.e-13);
    const auto heat=v::FourierHeatFlux{}.operator()<double>(grad_t,n,2.);
    const auto rotated_heat=v::FourierHeatFlux{}.operator()<double>(rotate(q,grad_t),rotate(q,n),2.);
    near(rotated_heat.value,heat.value,1.e-13,"heat flux is rotationally invariant");
    const auto phi=v::NewtonianStress{}.dissipation(g,0.9,0.2);
    near(v::NewtonianStress{}.dissipation(rotated_g,0.9,0.2).value,phi.value,2.e-13,"dissipation invariant");
    const auto s=v::NewtonianStress{}(g,0.9,0.2).value;
    const double contraction=s.xx*g.du_dx+s.yy*g.dv_dy+s.zz*g.dw_dz+
        s.xy*(g.du_dy+g.dv_dx)+s.xz*(g.du_dz+g.dw_dx)+s.yz*(g.dv_dz+g.dw_dy);
    near(phi.value,contraction,1.e-13,"Phi equals tau:grad(u)");
}
void nonthermal(const Exec&)
{
    struct NoTemperature {};
    const p::ConstantViscosity mu{0.001};mu.validate();
    v::VelocityGradient<double> g;g.du_dy=2.;
    const auto surface=v::surface_transport(v::NewtonianStress{}.traction(g,Vector{0,1,0},mu.evaluate<double>(NoTemperature{})));
    check(surface.valid(),"momentum-only path needs no thermal state");
    near(surface.value.outward_heat_flux,0.,0.,"omitted heat flux is zero");
    near(surface.value.traction.x,0.002,0.,"nonthermal traction");
    const auto custom=v::project_traction(v::SymmetricStress<double>{1,2,3,4,5,6},Vector{0,0,1});
    vector_near(custom.value,{5,6,3});
}
void properties(const Exec&)
{
    struct State {double temperature;};
    const auto law=p::TransportProperties{p::SutherlandViscosity{1.716e-5,273.15,110.4},
        p::ConstantPrandtl{0.72,p::ConstantHeatCapacity{1005.}}};law.validate();
    const auto coefficients=law.evaluate<double>(State{300.});
    v::VelocityGradient<double> g;g.du_dy=3.;
    const auto traction=v::NewtonianStress{}.traction(g,Vector{0,1,0},coefficients.dynamic_viscosity);
    const auto heat=v::FourierHeatFlux{}.operator()<double>(Vector{0,2,0},Vector{0,1,0},coefficients.thermal_conductivity);
    near(traction.value.x,3.*coefficients.dynamic_viscosity,0.,"viscosity integration");
    near(heat.value,-2.*coefficients.thermal_conductivity,0.,"conductivity integration");
    const auto default_gradient=v::velocity_gradient(Vector{},Vector{},Vector{});
    const auto default_heat=v::FourierHeatFlux{}(Vector{},Vector{1,0,0},krona::ComputeReal{1});
    static_assert(std::is_same_v<std::remove_cv_t<decltype(default_gradient)>,v::VelocityGradient<krona::ComputeReal>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(default_heat)>,v::Result<krona::ComputeReal>>);
    check(default_heat.valid(),"default thermal arithmetic type");
}
void invalid(const Exec&)
{
    const double nan=std::numeric_limits<double>::quiet_NaN(),max=std::numeric_limits<double>::max();
    v::VelocityGradient<double> g;g.du_dy=1.;
    check(v::NewtonianStress{}(g,-1.).status==v::Status::InvalidCoefficient,"negative mu rejected");
    check(v::NewtonianStress{}(g,1.,-1.).status==v::Status::InvalidCoefficient,"negative bulk rejected");
    auto bad=g;bad.du_dx=nan;
    const auto failed=v::NewtonianStress{}(bad,0.);
    check(failed.status==v::Status::InvalidGradient,"zero mu does not hide invalid gradient");
    const auto propagated=v::project_traction(failed,Vector{1,0,0});
    check(propagated.status==failed.status,"stress failure propagated");
    check(v::NewtonianStress{}.traction(g,Vector{2,0,0},1.).status==v::Status::InvalidNormal,"nonunit normal rejected");
    check(v::FourierHeatFlux{}.operator()<double>(Vector{1,0,0},Vector{1,0,0},-1.).status==v::Status::InvalidCoefficient,"negative lambda rejected");
    check(v::FourierHeatFlux{}.operator()<double>(Vector{nan,0,0},Vector{1,0,0},0.).status==v::Status::InvalidGradient,"zero lambda does not hide invalid gradient");
    check(v::project_traction(v::SymmetricStress<double>{nan},Vector{1,0,0}).status==v::Status::InvalidStress,"invalid supplied stress");
    const auto surface=v::surface_transport(v::NewtonianStress{}.traction(g,Vector{0,1,0},1.));
    check(v::outward_flux<Flux>(surface,Vector{nan,0,0}).status==v::Status::InvalidVelocity,"invalid velocity rejected");
    check(v::integrated_outward_flux<Flux>(surface,Vector{},0.).status==v::Status::InvalidArea,"zero area rejected");
    bad={};bad.du_dy=4.;
    check(v::NewtonianStress{}(bad,max).status==v::Status::NonfiniteResult,"stress overflow");
    check(v::NewtonianStress{}.dissipation(bad,max).status==v::Status::NonfiniteResult,"dissipation overflow");
    const auto bad_flux=v::outward_flux<Flux>(v::surface_transport(propagated),Vector{});
    check(!bad_flux.valid()&&std::isnan(bad_flux.value.mass)&&std::isnan(bad_flux.value.energy),"invalid flux has NaNs");
    const auto mixed=v::NewtonianStress{}.traction(g,r::Vector<float>{0.6f,0.8f,0},1.);
    check(mixed.valid(),"normal tolerance includes its storage precision");
}

template<class R,class Output>
struct Batch {
    Output output;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        v::VelocityGradient<R> g;g.du_dx=static_cast<R>(i)/R{10};g.du_dy=R{1}+static_cast<R>(i);
        const r::Vector<R> n{0,1,0};
        const auto t=v::NewtonianStress{}.traction(g,n,static_cast<R>(1.5),static_cast<R>(0.25));
        const auto q=v::FourierHeatFlux{}.template operator()<R>(r::Vector<R>{0,2,0},n,static_cast<R>(2.5));
        output(i)=v::outward_flux<r::Conservative<R>>(v::surface_transport(t,q),r::Vector<R>{4,1,0});
    }
};
template<class R> void batch(const Exec& exec)
{
    using Output=Kokkos::View<v::Result<r::Conservative<R>>*,typename Exec::memory_space>;
    Output output("viscous/flux",37);
    krona::test::complete(exec,[&]{
        Kokkos::parallel_for("viscous/flux",Kokkos::RangePolicy<Exec>(exec,0,37),Batch<R,Output>{output});
    });
    const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
    for(int i=0;i<37;++i){
        check(host(i).valid(),"kernel flux valid");
        const double tx=1.5*(1.+i),ty=(-1.+0.25)*(static_cast<double>(i)/10.);
        const double tolerance=128.*static_cast<double>(Kokkos::epsilon_v<R>)*(1.+tx);
        near(static_cast<double>(host(i).value.momentum.x),-tx,tolerance,"kernel momentum x");
        near(static_cast<double>(host(i).value.momentum.y),-ty,tolerance,"kernel momentum y");
        near(static_cast<double>(host(i).value.energy),-5.-4.*tx-ty,tolerance,"kernel energy");
    }
}
void device(const Exec& exec){batch<float>(exec);batch<double>(exec);}

int main(int argc,char** argv)
{
    const std::array cases{
        krona::test::Case{"shear",shear},krona::test::Case{"rigid_rotation",rigid_rotation},
        krona::test::Case{"dilation",dilation},krona::test::Case{"fourier",fourier},
        krona::test::Case{"energy",energy},krona::test::Case{"covariance",covariance},
        krona::test::Case{"nonthermal",nonthermal},krona::test::Case{"properties",properties},
        krona::test::Case{"invalid",invalid},krona::test::Case{"device",device}};
    return krona::test::run(argc,argv,"numerics.viscous",cases);
}
