#include <common/test_support.hpp>
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/fields/field_storage.hpp>
#include <algorithm>
#include <array>
#include <vector>

namespace {
using namespace krona;
namespace cv = transport::scalar::convection;
namespace df = transport::scalar::diffusion;
namespace ts = transport::scalar;
namespace rec = numerics::reconstruction;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using Real = ComputeReal;
using test::check;
#include "fixture.hpp"

/// @brief Returns a completed independent host snapshot on the selected execution instance.
template<class View> auto snapshot(View view, const Exec& exec) {
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view); exec.fence("Convection core snapshot"); return host;
}

/// @brief Owns device mesh and incidence allocations borrowed by the convection plan.
struct Attached {
    mesh::device::MeshStorage<Exec> mesh;
    mesh::device::AuxConnectivityStorage<Exec> connectivity;
    Attached(const Fixture& f, const Exec& exec): mesh(f.host, {}, exec), connectivity(f.connectivity, exec) {}
};

/// @brief Supplies an analytic space/time boundary profile or an intentionally invalid value.
struct AffineValue {
    bool invalid = false;
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const
    { return invalid ? Kokkos::quiet_NaN_v<Real> : Real{2}+Real(0.7)*c.x+c.time; }
};

/// @brief Converts a signed x velocity into an oriented integrated face flux.
struct VelocityFlux {
    Real velocity = 1;
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const { return velocity*c.nx*c.area; }
};
using Profile = bc::FixedValue<Real, AffineValue>;
using Boundary = bc::Manager<Exec, Real, Profile>;

/// @brief Provides independent graph-entry maps and strided matrix/RHS storage without Trilinos.
template<class R>
struct NativeSystem {
    struct Entries { Kokkos::View<std::size_t*, Space> diagonal, owner_entries, neighbour_entries; } entries;
    Kokkos::View<R***, Kokkos::LayoutRight, Space> blocks;
    Kokkos::View<R**, Kokkos::LayoutRight, Space> right;
    std::vector<std::size_t> offsets{0};
    std::vector<LocalIndex> columns;
    std::size_t owned;

    NativeSystem(const Fixture& f, const Exec& exec) : owned(static_cast<std::size_t>(f.host.n_own))
    {
        const auto& m = f.host;
        for (LocalIndex cell = 0; cell < m.n_own; ++cell) {
            std::vector<LocalIndex> row{cell};
            for (LocalIndex face = 0; face < m.n_inner_faces; ++face) {
                const auto i = static_cast<std::size_t>(face);
                if (m.face_owner[i] == cell) row.push_back(m.face_neigh[i]);
                else if (m.face_neigh[i] == cell) row.push_back(m.face_owner[i]);
            }
            std::sort(row.begin(), row.end()); row.erase(std::unique(row.begin(), row.end()), row.end());
            columns.insert(columns.end(), row.begin(), row.end()); offsets.push_back(columns.size());
        }
        entries = {Kokkos::View<std::size_t*, Space>("diagonal", owned),
                   Kokkos::View<std::size_t*, Space>("owner entries", m.n_inner_faces),
                   Kokkos::View<std::size_t*, Space>("neighbour entries", m.n_internal_faces)};

        // Resolve native entry indices independently of the production assembly helpers.
        const auto find = [&](LocalIndex row, LocalIndex col) {
            const auto i = static_cast<std::size_t>(row);
            auto first = columns.begin()+static_cast<std::ptrdiff_t>(offsets[i]);
            const auto last = columns.begin()+static_cast<std::ptrdiff_t>(offsets[i+1]);
            const auto p = std::find(first, last, col); check(p != last, "missing test graph entry");
            return static_cast<std::size_t>(p-columns.begin());
        };
        auto diagonal = Kokkos::create_mirror(entries.diagonal);
        auto owner = Kokkos::create_mirror(entries.owner_entries);
        auto neighbour = Kokkos::create_mirror(entries.neighbour_entries);
        for (LocalIndex i = 0; i < m.n_own; ++i) diagonal(i) = find(i, i);
        for (LocalIndex fidx = 0; fidx < m.n_inner_faces; ++fidx) {
            const auto k = static_cast<std::size_t>(fidx);
            owner(fidx) = find(m.face_owner[k], m.face_neigh[k]);
            if (fidx < m.n_internal_faces) neighbour(fidx) = find(m.face_neigh[k], m.face_owner[k]);
        }
        Kokkos::deep_copy(exec, entries.diagonal, diagonal); Kokkos::deep_copy(exec, entries.owner_entries, owner);
        Kokkos::deep_copy(exec, entries.neighbour_entries, neighbour);
        blocks = decltype(blocks)("native blocks", columns.size(), 2, 2);
        right = decltype(right)("strided RHS", owned+1, 3);
        reset(exec);
    }

    // Expose one matrix/RHS component while unrelated entries remain sentinel-filled.
    auto view() const
    {
        const auto values = Kokkos::subview(blocks, Kokkos::ALL(), 1, 1);
        const auto rhs = Kokkos::subview(right, std::pair<std::size_t, std::size_t>{0, owned}, 1);
        return ts::SystemView{values, rhs, entries};
    }

    // Clear only the selected equation; preserve sentinels in every other storage component.
    void reset(const Exec& exec)
    {
        Kokkos::deep_copy(exec, blocks, R{7}); Kokkos::deep_copy(exec, right, R{7});
        Kokkos::deep_copy(exec, view().values, R{0}); Kokkos::deep_copy(exec, view().rhs, R{0});
    }
};


/// @brief Checks native convection assembly, direct residuals, frozen corrections and additive diffusion.
template<class Storage, bool HighOrder, class Scheme = rec::SecondOrderUpwind>
void equation(const Exec& exec, bool duplicate=false, bool empty_last=false, bool empty=false,
              bool minimal=false, Scheme scheme={})
{
    const auto comm=mpi::CommView::world(); const auto f=bar(comm,duplicate,empty_last,empty);
    Attached a(f,exec); auto m=a.mesh.view();
    boundary::BoundaryLayout<Exec> layout(f.host,m,exec); Boundary boundary(layout);
    boundary.set(0,Profile{AffineValue{}}); boundary.set(1,Profile{AffineValue{}}); boundary.finalize();
    auto limited=m;
    if(minimal){
        limited.active_mask=mesh::device::MeshData::FaceCells|mesh::device::MeshData::FaceGeometry;
        limited.geometry.cell_centroid_x={}; limited.geometry.cell_centroid_y={}; limited.geometry.cell_centroid_z={};
        limited.geometry.cell_volume={};
    }
    cv::ConvectionPlan<Exec> plan(limited,a.connectivity.view(),comm,exec); cv::ConvectionOperator op(plan);
    fields::FieldStorage<Storage,3,Exec,Kokkos::LayoutRight> values("state",layout.required_field_size(),fields::FieldLocation::Custom,exec);
    fields::FieldStorage<Storage,3,Exec,Kokkos::LayoutRight> gradients("gradient",static_cast<std::size_t>(m.sizes.n_cells),fields::FieldLocation::Custom,exec);
    const auto phi=values.view().component(1); const auto g=gradients.view(); constexpr Real time=Real(0.25);
    Kokkos::parallel_for("affine input",Kokkos::RangePolicy<Exec>(exec,0,m.sizes.n_cells),KOKKOS_LAMBDA(int i){
        phi(i)=static_cast<Storage>(Real{2}+Real(0.7)*m.geometry.cell_centroid_x(i)+time);
        g(i,0)=Storage(0.7); g(i,1)=Storage{0}; g(i,2)=Storage{0};
    });
    const auto reconstruction=[&]{
        if constexpr(HighOrder) return cv::make_reconstruction(plan,scheme,rec::data(fields::pack(phi),rec::gradient_pack(g)));
        else return cv::NoReconstruction{};
    }();
    NativeSystem<Storage> system(f,exec); ts::AssemblyStatus<Exec> status(exec);
    constexpr bool float_storage = std::is_same_v<Storage, float>;
    const double tolerance = float_storage ? 3.0e-4 : 3.0e-11;

    // Reverse flow so both physical boundaries act as inflow and outflow in turn.
    for(const Real velocity : {Real{3},Real{-3}}){
        system.reset(exec);
        status.begin_async(); op.add_system_async(system.view(),boundary,status,VelocityFlux{velocity},reconstruction,cv::FixedValueInflow{},time); (void)status.check(comm);
        const auto coefficients=snapshot(system.view().values,exec); const auto rhs=snapshot(system.view().rhs,exec);
        Kokkos::View<Storage*,Space> residual("residual",m.sizes.n_own); Kokkos::deep_copy(exec,residual,Storage{0});
        status.begin_async(); op.add_residual_async(phi,residual,boundary,status,VelocityFlux{velocity},reconstruction,cv::FixedValueInflow{},time); (void)status.check(comm);

        // Evaluate b-A*phi independently from stored rows and compare with the direct residual.
        const auto actual=snapshot(residual,exec); const auto field=snapshot(phi,exec);
        for(std::size_t row=0;row<system.owned;++row){
            double action=static_cast<double>(rhs(row));
            for(auto k=system.offsets[row];k<system.offsets[row+1];++k) action-=static_cast<double>(coefficients(k))*static_cast<double>(field(system.columns[k]));
            const auto global=f.first+static_cast<int>(row);
            const bool entrance=velocity>0?global==0:global+1==f.total;
            const double expected=-static_cast<double>(velocity)*0.7*((!HighOrder&&entrance)?0.5:1.);
            test::near(action,expected,tolerance,"affine convective balance");
            test::near(static_cast<double>(actual(row)),action,tolerance,"matrix-free versus b-A*phi");
        }
        system.reset(exec);
        status.begin_async(); op.add_system_async(system.view(),boundary,status,VelocityFlux{velocity},cv::NoReconstruction{},cv::FixedValueInflow{},time); (void)status.check(comm);
        status.begin_async(); op.add_correction_async(system.view().rhs,boundary,status,VelocityFlux{velocity},reconstruction,time); (void)status.check(comm);

        // A correction-only RHS update must reproduce full assembly without changing coefficients.
        const auto updated=snapshot(system.view().rhs,exec); const auto fixed=snapshot(system.view().values,exec);
        for(std::size_t i=0;i<system.owned;++i) test::near(static_cast<double>(updated(i)),static_cast<double>(rhs(i)),tolerance,"frozen matrix RHS update");
        for(std::size_t k=0;k<system.columns.size();++k) test::near(static_cast<double>(fixed(k)),static_cast<double>(coefficients(k)),0.,"reconstruction changed matrix");

        // Unselected matrix components and the extra RHS row must retain their sentinels.
        const auto blocks=snapshot(system.blocks,exec); const auto right=snapshot(system.right,exec);
        for(std::size_t k=0;k<system.columns.size();++k) for(int i=0;i<2;++i) for(int j=0;j<2;++j)
            if(i!=1||j!=1) test::near(static_cast<double>(blocks(k,i,j)),7.,0.,"unselected block changed");
        for(std::size_t i=0;i<=system.owned;++i) for(int c=0;c<3;++c)
            if(i==system.owned||c!=1) test::near(static_cast<double>(right(i,c)),7.,0.,"unselected RHS or tail changed");
        if constexpr(HighOrder){
            mesh::device::AuxGeometryStorage<Exec> geometry(f.geometry,exec);
            df::DiffusionPlan<Exec> dp(m,a.connectivity.view(),geometry.view(),comm,exec,{true,1e-6}); df::DiffusionOperator diffusion(dp);
            const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
            status.begin_async(); diffusion.add_system_async(system.view(),boundary,status,law,time); (void)status.check(comm);
            status.begin_async(); diffusion.add_residual_async(phi,residual,boundary,status,law,time); (void)status.check(comm);
            const auto combined=snapshot(residual,exec); const auto av=snapshot(system.view().values,exec); const auto bv=snapshot(system.view().rhs,exec);
            for(std::size_t row=0;row<system.owned;++row){
                double action=static_cast<double>(bv(row));
                for(auto k=system.offsets[row];k<system.offsets[row+1];++k) action-=static_cast<double>(av(k))*static_cast<double>(field(system.columns[k]));
                test::near(action,static_cast<double>(combined(row)),tolerance,"additive convection/diffusion");
            }
        }
    }
}

/// @brief Checks first-order upwind assembly for both flow directions.
void upwind(const Exec&e){equation<double,false>(e);}

/// @brief Checks affine exactness with linear deferred correction.
void linear(const Exec&e){equation<double,true>(e);}

/// @brief Checks the TVD reconstruction path against the assembled affine balance.
void tvd(const Exec&e){equation<double,true>(e,false,false,false,false,rec::Tvd<>{});}

/// @brief Checks the MUSCL-kappa reconstruction path against the affine balance.
void muscl(const Exec&e){equation<double,true>(e,false,false,false,false,rec::MusclKappa<>{});}

/// @brief Checks convection with float state, coefficient and residual storage.
void float_storage(const Exec&e){equation<float,true>(e);}

/// @brief Checks that duplicate faces accumulate into shared graph entries.
void duplicate(const Exec&e){equation<double,true>(e,true);}

/// @brief Checks collective assembly when the final MPI rank owns no cells.
void empty_rank(const Exec&e){equation<double,true>(e,false,true);}

/// @brief Checks a globally empty convection problem.
void empty(const Exec&e){equation<double,true>(e,false,false,true);}

/// @brief Checks first-order operation with only face connectivity and geometry available.
void minimal(const Exec&e){equation<double,false>(e,false,false,false,true);}

/// @brief Supplies a time-dependent inflow independently of the registered condition type.
struct CustomInflow {
    template<class C,class Condition> KOKKOS_INLINE_FUNCTION Real operator()(const C&c,const Condition&)const{return Real{4}+c.time;}
};

/// @brief Forces inflow on both physical boundaries to exercise their value providers.
struct InwardFlux {
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C&c)const{return -Real{2}*c.area;}
};

/// @brief Checks strided borrowed patch values, profile updates and custom inflow callbacks.
void patch_values(const Exec&exec){
    const auto comm=mpi::CommView::world(); auto f=bar(comm,false,false,false,4); Attached a(f,exec);
    boundary::BoundaryLayout<Exec> layout(f.host,a.mesh.view(),exec);
    using P=bc::FixedValue<Real,boundary::profile::PatchValues<float,Space>>;
    bc::Manager<Exec,Real,P> boundary(layout);
    std::array<Kokkos::View<float**,Kokkos::LayoutRight,Space>,2> backing;
    for(std::size_t p=0;p<2;++p){
        const auto range=boundary.catalog().patches()[p].range;
        backing[p]=decltype(backing)::value_type("profile",range.end-range.begin,3);
        auto host=Kokkos::create_mirror(backing[p]);
        for(LocalIndex i=0;i<range.end-range.begin;++i){host(i,0)=-99;host(i,1)=static_cast<float>((i+1)*(p==0?1:10));host(i,2)=-99;}
        Kokkos::deep_copy(exec,backing[p],host);
        boundary.set(static_cast<mesh::PatchId>(p),bc::fixed_value<Real>(boundary::profile::patch_values(range,Kokkos::subview(backing[p],Kokkos::ALL(),1))));
    }
    boundary.finalize(); cv::ConvectionPlan<Exec> plan(a.mesh.view(),a.connectivity.view(),comm,exec); cv::ConvectionOperator op(plan);
    NativeSystem<double> system(f,exec); ts::AssemblyStatus<Exec> status(exec);
    status.begin_async(); op.add_system_async(system.view(),boundary,status,InwardFlux{}); (void)status.check(comm);
    const auto rhs=snapshot(system.view().rhs,exec);
    for(std::size_t i=0;i<system.owned;++i){const double expected=(comm.rank()==0&&i==0?5.:0.)+(comm.rank()+1==comm.size()&&i==2?50.:0.);test::near(rhs(i),expected,1e-12,"strided patch profile");}

    // Updating the borrowed profile needs no new registration.
    Kokkos::deep_copy(exec,Kokkos::subview(backing[0],Kokkos::ALL(),1),6.f); system.reset(exec);
    status.begin_async();op.add_system_async(system.view(),boundary,status,InwardFlux{});(void)status.check(comm);
    const auto updated=snapshot(system.view().rhs,exec);if(comm.rank()==0)test::near(updated(0),12.,1e-12,"borrowed profile refresh");
    boundary.set(0,bc::NormalGradient<Real>{0});boundary.set(1,bc::FixedFlux<Real>{0});boundary.finalize();system.reset(exec);
    status.begin_async();op.add_system_async(system.view(),boundary,status,InwardFlux{},cv::NoReconstruction{},CustomInflow{},Real{2});(void)status.check(comm);
    const auto custom=snapshot(system.view().rhs,exec);
    for(std::size_t i=0;i<system.owned;++i){const double expected=((comm.rank()==0&&i==0)||(comm.rank()+1==comm.size()&&i==2))?12.:0.;test::near(custom(i),expected,1e-12,"custom inflow callback");}
}

/// @brief Checks inflow values supplied for unlabelled boundary faces.
void unlabelled(const Exec&exec){
    const auto comm=mpi::CommView::world(); auto f=bar(comm);
    for(auto&id:f.host.face_patch)if(id==1)id=-1;
    f.host.n_labeled_bfaces_g=f.total==0?0:1;
    Attached a(f,exec);boundary::BoundaryLayout<Exec>layout(f.host,a.mesh.view(),exec);
    using P=bc::FixedValue<Real,boundary::profile::PatchValues<double,Space>>;
    bc::Manager<Exec,Real,P>boundary(layout);boundary.set(0,bc::FixedValue<Real>{2});boundary.set(1,bc::FixedValue<Real>{0});
    const auto range=boundary.catalog().unlabelled();
    Kokkos::View<double*,Space>profile("unlabelled",range.end-range.begin);Kokkos::deep_copy(exec,profile,9.);
    boundary.set_unlabelled(bc::fixed_value<Real>(boundary::profile::patch_values(range,profile)));boundary.finalize();
    cv::ConvectionPlan<Exec>plan(a.mesh.view(),a.connectivity.view(),comm,exec);cv::ConvectionOperator op(plan);NativeSystem<double>system(f,exec);ts::AssemblyStatus<Exec>status(exec);
    status.begin_async();op.add_system_async(system.view(),boundary,status,InwardFlux{});(void)status.check(comm);
    const auto rhs=snapshot(system.view().rhs,exec);
    for(std::size_t i=0;i<system.owned;++i){const double expected=(comm.rank()==0&&i==0?4.:0.)+(comm.rank()+1==comm.size()&&i==2?18.:0.);test::near(rhs(i),expected,1e-12,"unlabelled boundary provider");}
}

/// @brief Returns NaN corrections to detect access on zero-flux faces.
struct InvalidReconstruction {
    template<class C>KOKKOS_INLINE_FUNCTION Real interior(const C&,bool)const{return Kokkos::quiet_NaN_v<Real>;}
    template<class C>KOKKOS_INLINE_FUNCTION Real boundary(const C&)const{return Kokkos::quiet_NaN_v<Real>;}
};

/// @brief Poisons unused inputs to verify zero-flux and outflow short circuits.
void unused_inputs(const Exec&exec){
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,exec);boundary::BoundaryLayout<Exec>layout(f.host,a.mesh.view(),exec);Boundary boundary(layout);
    boundary.set(0,Profile{AffineValue{true}});boundary.set(1,Profile{AffineValue{true}});boundary.finalize();
    cv::ConvectionPlan<Exec>plan(a.mesh.view(),a.connectivity.view(),comm,exec);cv::ConvectionOperator op(plan);NativeSystem<double>system(f,exec);ts::AssemblyStatus<Exec>status(exec);
    Kokkos::View<double*,Space>phi("unused field",f.host.n_cells),residual("untouched residual",f.host.n_own);
    Kokkos::deep_copy(exec,phi,Kokkos::quiet_NaN_v<double>);Kokkos::deep_copy(exec,residual,7.);
    status.begin_async();op.add_system_async(system.view(),boundary,status,cv::UniformMassFlux<>{0},InvalidReconstruction{});
    op.add_residual_async(phi,residual,boundary,status,cv::UniformMassFlux<>{0},InvalidReconstruction{});
    op.add_correction_async(residual,boundary,status,cv::UniformMassFlux<>{0},InvalidReconstruction{});(void)status.check(comm);
    const auto untouched=snapshot(residual,exec);for(std::size_t i=0;i<system.owned;++i)test::near(untouched(i),7.,0.,"zero flux evaluated unused inputs");

    // For a positive velocity the right boundary is outflow: its NaN profile must stay unread.
    boundary.set(0,Profile{AffineValue{false}});boundary.finalize();
    status.begin_async();op.add_system_async(system.view(),boundary,status,VelocityFlux{1});(void)status.check(comm);
}

/// @brief Triggers one invalid plan or boundary input for the diagnostic wrapper.
void reject(const Exec&exec,std::string_view which){
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,exec);auto m=a.mesh.view();
    if(which=="null_comm"){cv::ConvectionPlan<Exec> p(m,a.connectivity.view(),mpi::CommView{MPI_COMM_NULL},exec);return;}
    if(which=="missing_faces"){m.connectivity.face_owner={};cv::ConvectionPlan<Exec>p(m,a.connectivity.view(),comm,exec);return;}
    boundary::BoundaryLayout<Exec> layout(f.host,m,exec);Boundary boundary(layout);
    boundary.set(0,Profile{AffineValue{which=="bad_profile"}});boundary.set(1,Profile{AffineValue{which=="bad_profile"}});
    if(which=="missing_inflow"){boundary.set(0,bc::NormalGradient<Real>{0});boundary.set(1,bc::FixedFlux<Real>{0});}
    boundary.finalize();cv::ConvectionPlan<Exec>plan(m,a.connectivity.view(),comm,exec);cv::ConvectionOperator op(plan);NativeSystem<double>system(f,exec);ts::AssemblyStatus<Exec>status(exec);status.begin_async();
    op.add_system_async(system.view(),boundary,status,InwardFlux{});(void)status.check(comm);
}

/// @brief Rejects inflow without a value provider for gradient/flux boundary conditions.
void missing_inflow(const Exec&e){reject(e,"missing_inflow");}

/// @brief Rejects a nonfinite inflow profile value.
void bad_profile(const Exec&e){reject(e,"bad_profile");}

/// @brief Rejects a convection plan with a null communicator.
void null_comm(const Exec&e){reject(e,"null_comm");}

/// @brief Rejects a convection plan missing required face connectivity.
void missing_faces(const Exec&e){reject(e,"missing_faces");}
} // namespace

/// @brief Runs native convection cases or an individually selected expected failure.
int main(int argc,char**argv){
    const std::array cases{test::Case{"upwind",upwind},test::Case{"linear",linear},test::Case{"tvd",tvd},test::Case{"muscl",muscl},test::Case{"float_storage",float_storage},test::Case{"duplicate",duplicate},test::Case{"empty_rank",empty_rank},test::Case{"empty",empty},test::Case{"minimal",minimal},test::Case{"patch_values",patch_values},test::Case{"unlabelled",unlabelled},test::Case{"unused_inputs",unused_inputs},test::Case{"missing_inflow",missing_inflow},test::Case{"bad_profile",bad_profile},test::Case{"null_comm",null_comm},test::Case{"missing_faces",missing_faces}};
    return test::run(argc,argv,"convection.core",cases);
}
