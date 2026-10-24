#include <common/test_support.hpp>
#include <krona/numerics/riemann/flux.hpp>
#include <array>
#include <vector> 

namespace r = krona::numerics::riemann;
using krona::test::check;
using krona::test::near;
using krona::test::Exec;
template<class R> auto state(R rho,R u,R p,R v=0,R w=0)
{ return r::ideal_gas_from_rhop(krona::thermodynamics::IdealGas<R>{},rho,r::Vector<R>{u,v,w},p); }
template<class R> void compare(r::Conservative<R> a,r::Conservative<R> b,R tol)
{
    near(a.mass,b.mass,tol,"mass flux"); near(a.energy,b.energy,tol,"energy flux");
    near(a.momentum.x,b.momentum.x,tol,"x momentum");
    near(a.momentum.y,b.momentum.y,tol,"y momentum");
    near(a.momentum.z,b.momentum.z,tol,"z momentum");
}
template<class R,class Solver> void invariants(Solver solver)
{
    const R tol=R{512}*Kokkos::epsilon_v<R>;
    const auto l=state<R>(R{1},static_cast<R>(0.3),R{1},static_cast<R>(0.2),static_cast<R>(-0.1));
    const auto q=state<R>(static_cast<R>(0.4),static_cast<R>(-0.2),static_cast<R>(0.3),static_cast<R>(-0.3),static_cast<R>(0.4));
    const r::Vector<R> n{static_cast<R>(0.6),static_cast<R>(0.8),0};
    const auto same=solver(l,l,n),forward=solver(l,q,n),reverse=solver(q,l,-n);
    check(same.valid()&&forward.valid()&&reverse.valid(),"valid fluxes");
    compare(same.flux,r::physical_flux(l,n),tol);
    compare(forward.flux,reverse.flux*R{-1},tol);
    auto rotate=[](r::Vector<R> v){return r::Vector<R>{v.z,v.x,v.y};};
    auto lr=l,qr=q; lr.velocity=rotate(l.velocity);qr.velocity=rotate(q.velocity);
    auto rotated=solver(lr,qr,rotate(n));auto expected=forward.flux;expected.momentum=rotate(expected.momentum);
    compare(rotated.flux,expected,tol);
    const auto area=r::integrated_flux(solver,l,q,n*R{3});
    compare(area.flux,forward.flux*R{3},tol*R{3});
    near(area.max_signal_speed,forward.max_signal_speed,tol,"area does not scale speed");
    check(!solver(l,q,r::Vector<R>{2,0,0}).valid(),"nonunit normal rejected");
    auto bad=l;bad.density=0;check(!solver(bad,q,n).valid(),"invalid density rejected");
}
void consistency(const Exec&)
{
    invariants<double>(r::Rusanov{});invariants<double>(r::Hll<>{});
    invariants<double>(r::Hllc<>{});invariants<double>(r::Roe{1.4});
    invariants<float>(r::Rusanov{});invariants<float>(r::Hll<>{});
    invariants<float>(r::Hllc<>{});invariants<float>(r::Roe{1.4f});
}
void contacts(const Exec&)
{
    for(double u: {0.,0.7,-0.7}) {
        const auto l=state(1.,u,1.,0.3,-0.2),q=state(2.,u,1.,-0.4,0.8);
        const auto exact=r::physical_flux(u>=0?l:q,r::Vector<double>{1,0,0});
        compare(r::Hllc<>{}(l,q,{1,0,0}).flux,exact,1.e-12);
        compare(r::Roe{1.4}(l,q,{1,0,0}).flux,exact,1.e-12);
    }
    const auto l=state(1.,2.,1.),q=state(24./11.,11./12.,19./6.);
    compare(r::Roe{1.4}(l,q,{1,0,0}).flux,r::Conservative<double>{2.,{5.,0,0},11.},1.e-12);
}
struct InvalidSpeeds {
    template<class R> KOKKOS_INLINE_FUNCTION r::WaveSpeeds<R> operator()(r::State<R>,r::State<R>,r::Vector<R>)const
    {return {R{1},R{-1}};}
};
void fallback(const Exec&)
{
    const auto l=state(1.,-2.,0.4),q=state(1.,2.,0.4);
    const auto h=r::Hll<>{}(l,q,{1,0,0});
    const auto c=r::Hllc<>{}(l,q,{1,0,0}),roe=r::Roe{1.4}(l,q,{1,0,0});
    check(c.used_fallback()&&roe.used_fallback(),"rarefaction fallback reported");
    compare(c.flux,h.flux,1.e-12);compare(roe.flux,h.flux,1.e-12);
    check(!r::Hllc<>{r::DavisSpeeds{},false}(l,q,{1,0,0}).valid(),"strict HLLC");
    check(!r::Roe{1.4,true,false}(l,q,{1,0,0}).valid(),"strict Roe");
    auto incompatible=l;incompatible.internal_energy+=10;
    check(r::Roe{1.4}(incompatible,q,{1,0,0}).status==r::Status::IncompatibleModel,"EOS mismatch");
    bool threw=false;try{r::Roe bad{1.};}catch(const std::invalid_argument&){threw=true;}
    check(threw,"invalid gamma rejected");
    check(r::Hll<InvalidSpeeds>{}(l,q,{1,0,0}).status==r::Status::InvalidFan,"invalid HLL bounds");
    check(r::Hllc<InvalidSpeeds>{}(l,q,{1,0,0}).status==r::Status::InvalidFan,"invalid HLLC bounds");
    check(!r::integrated_flux(r::Rusanov{},l,q,r::Vector<double>{}).valid(),"zero area rejected");
    const auto tl=state(1.,0.8,1.),tr=state(1.,1.6,1.);
    const auto fixed=r::Roe{1.4}(tl,tr,{1,0,0}),raw=r::Roe{1.4,false}(tl,tr,{1,0,0});
    check(fixed.valid()&&raw.valid()&&!fixed.used_fallback(),"entropy-fix fan is admissible");
    check(std::abs(fixed.flux.mass-raw.flux.mass)>1.e-4,"acoustic entropy correction is active");
}
void eos(const Exec&)
{
    const krona::thermodynamics::IdealGas<double> gas{};
    const auto q=r::from_pT(gas,101325.,300.,r::Vector<double>{12.,-2.,1.});
    const auto back=r::from_conservative(gas,r::conservative(q));
    near(back.pressure,q.pressure,1.e-9,"EOS round trip pressure");
    near(back.density,q.density,1.e-14,"EOS round trip density");
    auto shifted=q;shifted.internal_energy-=1.e6;
    check(r::Hllc<>{}(shifted,shifted,{1,0,0}).valid(),"general energy reference allowed");
    check(r::Roe{1.4}(shifted,shifted,{1,0,0}).status==r::Status::IncompatibleModel,"Roe energy reference restriction");
}
template<class R,class Solver,class Output>
struct Batch {
    Solver solver;Output out;
    KOKKOS_INLINE_FUNCTION void operator()(int i)const
    {
        const r::State<R> q{R{1},{static_cast<R>(0.5),static_cast<R>(0.2),0},R{1},static_cast<R>(2.5),Kokkos::sqrt(static_cast<R>(1.4))};
        out(i)=solver(q,q,r::Vector<R>{1,0,0});
    }
};
template<class R,class Solver> void launch(const Exec& exec,Solver solver)
{
    using View=Kokkos::View<r::Result<R>*,typename Exec::memory_space>;
    View output("flux",37);
    krona::test::complete(exec,[&]{Kokkos::parallel_for("riemann/batch",Kokkos::RangePolicy<Exec>(exec,0,37),
        Batch<R,Solver,View>{solver,output});});
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
    for(int i=0;i<37;++i){check(host(i).valid(),"device status");near(host(i).flux.mass,static_cast<R>(0.5),static_cast<R>(1.e-6),"device flux");}
}
void device(const Exec& exec)
{
    launch<double>(exec,r::Rusanov{});launch<double>(exec,r::Hll<>{});
    launch<double>(exec,r::Hllc<>{});launch<double>(exec,r::Roe{1.4});
    launch<float>(exec,r::Rusanov{});launch<float>(exec,r::Hllc<>{});launch<float>(exec,r::Roe{1.4f});
}
// Independent exact Sod solution at one point, using the known pressure/velocity root.
double exact_sod(double x,double t)
{
    constexpr double p=0.30313017805064685,u=0.9274526200489499,g=1.4;
    const double s=(x-0.5)/t,a=std::sqrt(g),astar=a*std::pow(p,(g-1)/(2*g));
    if(s<-a)return 1.;
    if(s<u-astar)return std::pow((2/(g+1)+(g-1)/(g+1)*(-s)/a),2/(g-1));
    if(s<u)return std::pow(p,1/g);
    const double shock=std::sqrt((g+1)/2*(p/0.125)+(g-1)/2*(0.1/0.125));
    if(s<shock)return 0.125*((p/0.1)+(g-1)/(g+1))/((g-1)/(g+1)*(p/0.1)+1);
    return 0.125;
}
template<class Solver> void sod_run(Solver solver)
{
    constexpr int count=240;const double dx=1./count;
    std::vector<r::Conservative<double>> q(count),next(count),flux(count+1);
    for(int i=0;i<count;++i)q[i]=r::conservative(i<count/2?state(1.,0.,1.):state(0.125,0.,0.1));
    const auto initial=q;double t=0; r::Conservative<double> boundary{};
    const krona::thermodynamics::IdealGas<double> gas{};
    while(t<0.15){double speed=0;
        for(int f=0;f<=count;++f){const auto l=r::from_conservative(gas,q[f==0?0:f-1]);
            const auto rr=r::from_conservative(gas,q[f==count?count-1:f]);
            const auto result=solver(l,rr,r::Vector<double>{1,0,0});check(result.valid(),"Sod flux valid");
            flux[f]=result.flux;speed=std::max(speed,result.max_signal_speed);}
        const double dt=std::min(0.3*dx/speed,0.15-t);
        for(int i=0;i<count;++i){next[i]=q[i]-(flux[i+1]-flux[i])*(dt/dx);
            check(r::detail::valid(r::from_conservative(gas,next[i])),"positive Sod state");}
        boundary=boundary+(flux[count]-flux[0])*dt;q.swap(next);t+=dt;
    }
    double error=0;r::Conservative<double> change{};
    for(int i=0;i<count;++i){error+=dx*std::abs(q[i].mass-exact_sod((i+0.5)*dx,t));change=change+(q[i]-initial[i])*dx;}
    check(error<0.025,"Sod density L1 error");compare(change+boundary,r::Conservative<double>{},2.e-13);
}
void sod(const Exec&){sod_run(r::Rusanov{});sod_run(r::Hll<>{});sod_run(r::Hllc<>{});sod_run(r::Roe{1.4});}
int main(int argc,char**argv)
{
    const std::array cases{krona::test::Case{"consistency",consistency},krona::test::Case{"contacts",contacts},
        krona::test::Case{"fallback",fallback},krona::test::Case{"eos",eos},
        krona::test::Case{"device",device},krona::test::Case{"sod",sod}};
    return krona::test::run(argc,argv,"numerics.riemann",cases);
}
