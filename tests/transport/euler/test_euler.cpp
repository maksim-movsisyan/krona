#include <common/test_support.hpp>
#include "fixture.hpp"
#include <krona/transport/diffusion/isotropic.hpp>
#include <krona/numerics/riemann/roe.hpp>
#include <array>
using namespace euler_test;
using krona::test::check;
using krona::test::near;

template<class Solver>
void uniform_run(const Exec& exec,Solver solver)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);
    boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()},solver};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);
    const auto report=status.check(comm);check(report.reconstruction_fallbacks==0,"uniform reconstruction has no fallback");
    const auto result=fields.snapshot();for(std::size_t i=0;i<result.extent(0);++i)for(int c=0;c<5;++c)near(result(i,c),0.,tolerance(),"uniform flow residual");
}
void uniform(const Exec& exec){uniform_run(exec,ri::Hllc<>{});uniform_run(exec,ri::Rusanov{});uniform_run(exec,ri::Roe{gas().gamma});}

void split_additive(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec,true);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary,true);boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto expected=fields.snapshot();fields.zero(exec,A{2});
    status.begin_async();
    for(std::size_t i=0;i<2;++i){
        op.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status,{0,grid.host.n_internal_faces},{i,i+1});
        op.add_boundary_residual_async(fields.reconstruction(grid),fields.output(),boundary,status,eu::PreparedExterior{},R{0},{i,i+1});
    }
    (void)status.check(comm);
    const auto result=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)
        near(result(i,c),expected(i,c)+2.,tolerance(),"split calls preserve existing residual");
    // Internal flux cancellation in canonical orientation, independently of physical boundaries.
    fields.zero(exec);status.begin_async();op.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status);(void)status.check(comm);
    const auto inner=fields.snapshot();for(int c=0;c<5;++c)near(inner(0,c)+inner(1,c),0.,tolerance(),"internal conservation");
}

auto high_order(const Grid& grid,const auto& fields,auto gradients,auto scheme)
{
    return eu::Reconstruction{scheme,rec::data(fields.primitive(),rec::gradient_pack(gradients)),
        rec::MeshStencil<Space,R>{grid.storage.view(),grid.connectivity.view()}};
}
struct ScaledLinear {
    static constexpr unsigned requirements=rec::Gradients;
    R scale=0;
    template<class T,class Side,class Data>
    KOKKOS_INLINE_FUNCTION T increment(const Side& side,const Data& data,std::size_t component)const
    {return static_cast<T>(scale)*rec::SecondOrderUpwind{}.template increment<T>(side,data,component);}
};
void affine(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec,true);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary,true);boundary.apply_async(fields.physical());
    krona::fields::FieldStorage<S,15,Exec> gradients("grad",static_cast<std::size_t>(grid.host.n_cells),krona::fields::FieldLocation::Custom,exec);
    auto host=gradients.create_host_buffer();for(std::size_t i=0;i<gradients.size();++i)for(int c=0;c<15;++c)
        host(i,c)=static_cast<S>(c==0?0.1:c==3?0.2:c==12?0.05:0.);
    krona::execution::complete(exec,[&]{Kokkos::deep_copy(exec,gradients.view().kokkos_view(),host);});
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    const auto reconstruction=high_order(grid,fields,gradients.view(),rec::SecondOrderUpwind{});
    status.begin_async();op.add_residual_async(reconstruction,fields.output(),boundary,status);
    check(status.check(comm).reconstruction_fallbacks==0,"affine states admissible");
    const auto result=fields.snapshot();const eu::Thermodynamics thermodynamics{gas()};
    for(int i=0;i<2;++i){
        const auto expected=ri::physical_flux(thermodynamics(profile(i,true)),ri::Vector<R>{1,0,0})-
            ri::physical_flux(thermodynamics(profile(i+1.,true)),ri::Vector<R>{1,0,0});
        const std::array<double,5> values{static_cast<double>(expected.mass),static_cast<double>(expected.momentum.x),static_cast<double>(expected.momentum.y),static_cast<double>(expected.momentum.z),static_cast<double>(expected.energy)};
        for(int c=0;c<5;++c)near(result(static_cast<std::size_t>(i),c),values[static_cast<std::size_t>(c)],tolerance(),"affine face flux residual");
    }
    fields.zero(exec);status.begin_async();
    op.add_residual_async(high_order(grid,fields,gradients.view(),ScaledLinear{R{1}}),fields.output(),boundary,status);(void)status.check(comm);
    const auto configured=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)
        near(configured(i,c),result(i,c),tolerance(),"boundary scheme inherits configured parameters");
    fields.zero(exec);status.begin_async();
    op.add_residual_async(reconstruction,fields.output(),boundary,status,eu::EvaluatedExterior<>{});(void)status.check(comm);
    const auto evaluated=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)
        near(evaluated(i,c),result(i,c),tolerance(),"boundary re-evaluation from reconstructed inside");
    // The TVD/Darwish adapter uses the same five-component batch and canonical stencil.
    fields.zero(exec);status.begin_async();
    op.add_inner_residual_async(high_order(grid,fields,gradients.view(),rec::Tvd<>{}),fields.output(),status);(void)status.check(comm);
    const auto tvd=fields.snapshot();const auto f=ri::physical_flux(thermodynamics(profile(1.,true)),ri::Vector<R>{1,0,0});
    near(tvd(0,0),-static_cast<double>(f.mass),tolerance(),"TVD internal face");near(tvd(0,4),-static_cast<double>(f.energy),tolerance(),"TVD batch energy");
    fields.zero(exec);status.begin_async();op.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status);(void)status.check(comm);
    const auto first_order=fields.snapshot();fields.zero(exec);
    krona::fields::FieldStorage<S,1,Exec> factors("factors",2,krona::fields::FieldLocation::Custom,exec);factors.fill(exec,S{0});
    const auto limited=eu::Reconstruction{rec::LimitedLinear{},rec::data(fields.primitive(),rec::gradient_pack(gradients.view()),rec::field_pack(factors.view())),
        rec::MeshStencil<Space,R>{grid.storage.view(),grid.connectivity.view()}};
    status.begin_async();op.add_inner_residual_async(limited,fields.output(),status);(void)status.check(comm);
    const auto result_limited=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)
        near(result_limited(i,c),first_order(i,c),tolerance(),"zero external factors select first order");
}

void reconstruction_fallback(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status);(void)status.check(comm);const auto expected=fields.snapshot();fields.zero(exec);
    krona::fields::FieldStorage<S,15,Exec> gradients("bad_increment",2,krona::fields::FieldLocation::Custom,exec);
    gradients.fill(exec,S{0});auto h=gradients.create_host_buffer();for(std::size_t i=0;i<2;++i)for(int c=0;c<15;++c)h(i,c)=0;
    h(0,0)=-10.;h(1,0)=10.;krona::execution::complete(exec,[&]{Kokkos::deep_copy(exec,gradients.view().kokkos_view(),h);});
    status.begin_async();op.add_inner_residual_async(high_order(grid,fields,gradients.view(),rec::SecondOrderUpwind{}),fields.output(),status);
    const auto report=status.check(comm);check(report.reconstruction_fallbacks==4,"two bad sides per two row evaluations");
    const auto actual=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)near(actual(i,c),expected(i,c),tolerance(),"whole-side first-order fallback");
}

void walls(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);
    boundary.set(0,ph::IsothermalWall<R>{R{2}});boundary.set(1,ph::HeatFluxWall<R>{static_cast<R>(0.1)});boundary.set(2,ph::AdiabaticWall<R>{});
    boundary.apply_async(fields.physical(),krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<R>{R{1}}});
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto result=fields.snapshot();double mass=0,energy=0;for(std::size_t i=0;i<2;++i){mass+=result(i,0);energy+=result(i,4);}
    near(mass,0.,tolerance(),"stationary walls have zero inviscid mass flux");near(energy,0.,tolerance(),"wall heat data do not enter Euler energy flux");
    fields.zero(exec);status.begin_async();
    op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status,eu::EvaluatedExterior<>{});(void)status.check(comm);
    const auto evaluated=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)
        near(evaluated(i,c),result(i,c),tolerance(),"inviscid wall re-evaluation needs no thermal law");
}

void layouts(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<Kokkos::LayoutRight> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto result=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)near(result(i,c),0.,tolerance(),"strided component packs");
}

void distributed(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();check(comm.size()>=2,"distributed test needs at least two ranks");
    Grid grid(comm,exec,comm.rank(),1,comm.size());Fields<> fields(grid,exec);fields.initialize(grid,exec);
    const auto own=profile(comm.rank()+0.5,true);
    std::array<double,5> local{static_cast<double>(own.pressure),static_cast<double>(own.velocity.x),0,0,static_cast<double>(own.temperature)};
    std::vector<double> gathered(static_cast<std::size_t>(comm.size())*5);
    // Application/test exchange: the Euler operator itself performs none.
    krona::mpi::check(MPI_Allgather(local.data(),5,MPI_DOUBLE,gathered.data(),5,MPI_DOUBLE,comm.native()),"test exchange primitives");
    auto p=fields.pressure.create_host_buffer();auto t=fields.temperature.create_host_buffer();auto u=fields.velocity.create_host_buffer();
    for(std::size_t i=0;i<fields.pressure.size();++i){const auto id=i<grid.host.cell_gid.size()?static_cast<std::size_t>(grid.host.cell_gid[i]):0;
        p(i)=static_cast<S>(gathered[id*5]);t(i)=static_cast<S>(gathered[id*5+4]);u(i,0)=static_cast<S>(gathered[id*5+1]);u(i,1)=u(i,2)=0;}
    krona::execution::complete(exec,[&]{Kokkos::deep_copy(exec,fields.pressure.view().kokkos_view(),p);Kokkos::deep_copy(exec,fields.temperature.view().kokkos_view(),t);Kokkos::deep_copy(exec,fields.velocity.view().kokkos_view(),u);});
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary,true);boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status,{grid.host.n_internal_faces,grid.host.n_inner_faces});
    op.add_boundary_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto result=fields.snapshot();const eu::Thermodynamics law{gas()};const ri::Hllc<> solver;
    auto state=[&](int rank){return law(profile(rank+0.5,true));};
    const auto left=comm.rank()==0?solver(law(profile(0.,true)),state(0),ri::Vector<R>{1,0,0}).flux:
        solver(state(comm.rank()-1),state(comm.rank()),ri::Vector<R>{1,0,0}).flux;
    const auto right=comm.rank()+1==comm.size()?ri::physical_flux(state(comm.rank()),ri::Vector<R>{1,0,0}):
        solver(state(comm.rank()),state(comm.rank()+1),ri::Vector<R>{1,0,0}).flux;
    const auto expected=left-right;
    const std::array<double,5> values{static_cast<double>(expected.mass),static_cast<double>(expected.momentum.x),static_cast<double>(expected.momentum.y),static_cast<double>(expected.momentum.z),static_cast<double>(expected.energy)};
    for(int c=0;c<5;++c)near(result(0,c),values[static_cast<std::size_t>(c)],tolerance(),"MPI interface flux orientation");
    std::array<double,5> sum{},local_result{};for(int c=0;c<5;++c)local_result[static_cast<std::size_t>(c)]=result(0,c);
    krona::mpi::check(MPI_Allreduce(local_result.data(),sum.data(),5,MPI_DOUBLE,MPI_SUM,comm.native()),"test global balance");
    const auto a=solver(law(profile(0.,true)),state(0),ri::Vector<R>{1,0,0}).flux-ri::physical_flux(state(comm.size()-1),ri::Vector<R>{1,0,0});
    near(sum[0],static_cast<double>(a.mass),tolerance(),"global mass balance");near(sum[4],static_cast<double>(a.energy),tolerance(),"global energy balance");
}

template<class Velocity>
struct SetSubsonicVelocity {
    Velocity velocity;
    KOKKOS_INLINE_FUNCTION void operator()(int i)const
    {velocity(static_cast<std::size_t>(i),0)=static_cast<S>(0.3);}
};
void partition_algebra(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();
    auto initialize=[&](auto& fields,const Grid& grid){
        fields.initialize(grid,exec,true);fields.velocity.fill(exec,S{0});const auto velocity=fields.velocity.view();
        Kokkos::parallel_for("test/subsonic",Kokkos::RangePolicy<Exec>(exec,0,static_cast<int>(fields.velocity.size())),
            SetSubsonicVelocity<decltype(velocity)>{velocity});
    };
    Grid serial(comm,exec,0,3,3);Fields<> whole(serial,exec);initialize(whole,serial);
    const eu::EulerOperator serial_op{serial.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();serial_op.add_inner_residual_async(whole.reconstruction(serial),whole.output(),status);(void)status.check(comm);
    const auto reference=whole.snapshot();
    for(int part=0;part<3;++part){
        // Synthetic partitions with explicit ghost values; this checks the algebra without a multi-process runtime.
        Grid piece(comm,exec,part,1,3);Fields<> fields(piece,exec);initialize(fields,piece);
        const eu::EulerOperator op{piece.plan,eu::Thermodynamics{gas()}};
        status.begin_async();op.add_inner_residual_async(fields.reconstruction(piece),fields.output(),status);(void)status.check(comm);
        const auto result=fields.snapshot();for(int c=0;c<5;++c)
            near(result(0,c),reference(static_cast<std::size_t>(part),c),tolerance(),"partition-independent subsonic residual");
    }
}
struct TimeBoundary {
    template<class T,class M,class L,class C>
    KOKKOS_INLINE_FUNCTION auto operator()(const krona::boundary::FaceContext<T>& c,ph::Primitive<T,true> inside,
        const M&,const L&,const C&)const
    {inside.pressure=T{1}+static_cast<T>(0.1)*c.time;return ph::Evaluation{inside,inside};}
};
void custom_time(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    ph::Manager<Model,Exec,R,TimeBoundary> boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);
    boundary.set(0,TimeBoundary{});boundary.set(1,ph::SupersonicOutlet{});boundary.set(2,ph::Symmetry{});
    boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status,eu::PreparedExterior{},R{2});(void)status.check(comm);
    const auto result=fields.snapshot();auto external=profile(0.);external.pressure=static_cast<R>(1.2);
    const eu::Thermodynamics law{gas()};const ri::Hllc<> solver;
    const auto f=solver(law(external),law(profile(0.5)),ri::Vector<R>{1,0,0}).flux-ri::physical_flux(law(profile(0.5)),ri::Vector<R>{1,0,0});
    near(result(0,0),static_cast<double>(f.mass),tolerance(),"custom boundary observes requested time");
    near(result(0,4),static_cast<double>(f.energy),tolerance(),"custom boundary energy");
}
void empty(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec,0,0,0);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);
    const auto report=status.check(comm);check(report.reconstruction_fallbacks==0&&report.riemann_fallbacks==0,"empty rank is a no-op");
}
void prepared_profile(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.apply_async(fields.physical());
    // A caller-provided profile overrides the already prepared inlet face, without changing the stored condition.
    const auto index=static_cast<std::size_t>(grid.host.n_cells);
    Kokkos::deep_copy(exec,Kokkos::subview(fields.pressure.view().kokkos_view(),index),static_cast<S>(1.2));
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto result=fields.snapshot();auto face=profile(0.);face.pressure=static_cast<R>(1.2);const eu::Thermodynamics law{gas()};
    const auto expected=ri::Hllc<>{}(law(face),law(profile(0.5)),ri::Vector<R>{1,0,0}).flux-ri::physical_flux(law(profile(0.5)),ri::Vector<R>{1,0,0});
    near(result(0,0),static_cast<double>(expected.mass),tolerance(),"prepared inlet profile is consumed");
    near(result(0,4),static_cast<double>(expected.energy),tolerance(),"prepared inlet energy");
}
void reject_boundary(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.apply_async(fields.physical());
    Kokkos::deep_copy(exec,Kokkos::subview(fields.temperature.view().kokkos_view(),static_cast<std::size_t>(grid.host.n_cells)),S{0});
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);status.begin_async();
    op.add_boundary_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
}

void unlabelled(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec,0,2,2,true);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.set_unlabelled(ph::Symmetry{});
    boundary.apply_async(fields.physical());const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto result=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)near(result(i,c),0.,tolerance(),"unlabelled patch row map");
}
struct PressureDomain {
    template<class E,class P>KOKKOS_INLINE_FUNCTION bool operator()(const E&,const P& q)const
    {return q.pressure<=static_cast<decltype(q.pressure)>(1.1);}
};
struct RestrictedEOS {
    Gas gas= euler_test::gas();
    void validate()const{gas.validate();}
    template<class T>KOKKOS_INLINE_FUNCTION auto from_pT(T pressure,T temperature)const
    {
        if(pressure>static_cast<T>(1.1))Kokkos::abort("test: domain guard was bypassed");
        return gas.from_pT(pressure,temperature);
    }
};
void eos_domain(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    const eu::EulerOperator baseline{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();baseline.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status);(void)status.check(comm);const auto expected=fields.snapshot();fields.zero(exec);
    krona::fields::FieldStorage<S,15,Exec> gradients("domain_gradient",2,krona::fields::FieldLocation::Custom,exec);
    auto h=gradients.create_host_buffer();for(std::size_t i=0;i<2;++i)for(int c=0;c<15;++c)h(i,c)=0;
    h(0,0)=2.;h(1,0)=-2.;krona::execution::complete(exec,[&]{Kokkos::deep_copy(exec,gradients.view().kokkos_view(),h);});
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{RestrictedEOS{},PressureDomain{}}};status.begin_async();
    op.add_inner_residual_async(high_order(grid,fields,gradients.view(),rec::SecondOrderUpwind{}),fields.output(),status);
    check(status.check(comm).reconstruction_fallbacks==4,"EOS domain checked before calling restricted EOS");
    const auto result=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)near(result(i,c),expected(i,c),tolerance(),"domain fallback");
}

void geometry(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec,0,2,2,false,1.3,0.7,true);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    Boundary boundary(grid.host,grid.storage.view(),Model{gas(),{}},exec);open_boundaries(boundary);boundary.apply_async(fields.physical());
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()},ri::Rusanov{}};eu::AssemblyStatus<Exec> status(exec);
    status.begin_async();op.add_residual_async(fields.reconstruction(grid),fields.output(),boundary,status);(void)status.check(comm);
    const auto result=fields.snapshot();for(std::size_t i=0;i<2;++i)for(int c=0;c<5;++c)
        near(result(i,c),0.,512.*static_cast<double>(Kokkos::epsilon_v<A>),"skew-cell free stream and geometry precision");
}

void reject_alias(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);status.begin_async();
    auto output=eu::primitive_fields(fields.pressure.view(),fields.velocity.view(),fields.temperature.view());
    op.add_inner_residual_async(fields.reconstruction(grid),output,status);
}
void reject_state(const Exec& exec)
{
    const auto comm=krona::mpi::CommView::world();Grid grid(comm,exec);Fields<> fields(grid,exec);fields.initialize(grid,exec);fields.pressure.fill(exec,S{-1});
    const eu::EulerOperator op{grid.plan,eu::Thermodynamics{gas()}};eu::AssemblyStatus<Exec> status(exec);status.begin_async();
    op.add_inner_residual_async(fields.reconstruction(grid),fields.output(),status);(void)status.check(comm);
}

int main(int argc,char** argv)
{
    const std::array cases{krona::test::Case{"uniform",uniform},krona::test::Case{"split_additive",split_additive},
        krona::test::Case{"affine",affine},krona::test::Case{"reconstruction_fallback",reconstruction_fallback},
        krona::test::Case{"walls",walls},krona::test::Case{"layouts",layouts},
        krona::test::Case{"partition_algebra",partition_algebra},krona::test::Case{"custom_time",custom_time},
        krona::test::Case{"empty",empty},krona::test::Case{"prepared_profile",prepared_profile},
        krona::test::Case{"unlabelled",unlabelled},krona::test::Case{"eos_domain",eos_domain},krona::test::Case{"geometry",geometry}};
    const std::array special{krona::test::Case{"distributed",distributed},krona::test::Case{"reject_alias",reject_alias},krona::test::Case{"reject_state",reject_state},krona::test::Case{"reject_boundary",reject_boundary}};
    if(argc==2&&(std::string_view(argv[1])=="distributed"||std::string_view(argv[1]).starts_with("reject_")))return krona::test::run(argc,argv,"transport.euler",special);
    return krona::test::run(argc,argv,"transport.euler",cases);
}