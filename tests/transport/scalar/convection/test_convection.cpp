#include "common/test_support.hpp"

#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/halo/exchange.hpp>

#include <Kokkos_NumericTraits.hpp>

#include <algorithm>
#include <array>
#include <string_view>

namespace {
using namespace krona;
namespace ts = transport::scalar;
namespace cv = transport::scalar::convection;
namespace df = transport::scalar::diffusion;
namespace rec = numerics::reconstruction;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;
using Boundary = bc::Manager<Exec, Real>;
using krona::test::check;

// References are evaluated in double; account for configured compute/geometry rounding.
void near(double actual,double expected,double tolerance,std::string_view message,
    std::source_location where=std::source_location::current())
{
    if(tolerance>0){
        const double epsilon=static_cast<double>(Kokkos::epsilon_v<Real>)+static_cast<double>(Kokkos::epsilon_v<GeometryReal>);
        tolerance=std::max(tolerance,32.0*epsilon*(1.0+std::abs(expected)));
    }
    test::near(actual,expected,tolerance,message,where);
}

struct Fixture {
    mesh::MeshPart host;
    mesh::MeshAuxConnectivity connectivity;
    mesh::MeshAuxGeometry geometry;
    int first = 0, total = 0;
};

Fixture bar(mpi::CommView comm, bool duplicate = false, bool empty_last = false, bool all_empty = false)
{
    Fixture f;
    auto& m = f.host;
    const int active = all_empty ? 0 : (empty_last && comm.size() > 1 ? comm.size()-1 : comm.size());
    const int count = comm.rank() < active ? 3 : 0;
    f.first = comm.rank()*3; f.total = active*3;
    m.rank = comm.rank(); m.nprocs = comm.size(); m.n_own = count;
    m.n_cells_g = static_cast<std::uint64_t>(f.total);
    m.faces_grouped = true;
    m.patches.resize(2); m.patches[0].name = "left"; m.patches[1].name = "right";
    m.n_bfaces_g = f.total == 0 ? 0 : 2; m.n_labeled_bfaces_g = m.n_bfaces_g;
    for (int i = 0; i < count; ++i) m.cell_centroid_x.push_back(static_cast<GeometryReal>(f.first+i)+GeometryReal{0.5});
    for (int peer : {comm.rank()-1, comm.rank()+1}) {
        if (count == 0 || peer < 0 || peer >= active) continue;
        m.nb_ranks.push_back(peer);
        m.send_owned_local.push_back(peer < comm.rank() ? 0 : count-1);
        m.recv_ghost_local.push_back(static_cast<LocalIndex>(m.cell_centroid_x.size()));
        m.cell_centroid_x.push_back(static_cast<GeometryReal>(peer < comm.rank() ? f.first-1 : f.first+count)+GeometryReal{0.5});
        m.send_offsets.push_back(m.send_owned_local.size()); m.recv_offsets.push_back(m.recv_ghost_local.size());
    }
    m.n_cells = static_cast<LocalIndex>(m.cell_centroid_x.size());
    m.cell_centroid_y.assign(m.cell_centroid_x.size(), 0); m.cell_centroid_z.assign(m.cell_centroid_x.size(), 0);
    m.cell_volume.assign(m.cell_centroid_x.size(), 1);
    const auto face = [&](LocalIndex owner, LocalIndex neighbour, GeometryReal x, GeometryReal normal, GeometryReal area, mesh::PatchId patch) {
        m.face_owner.push_back(owner); m.face_neigh.push_back(neighbour); m.face_patch.push_back(patch);
        m.face_centroid_x.push_back(x); m.face_centroid_y.push_back(0); m.face_centroid_z.push_back(0);
        m.face_normal_x.push_back(normal); m.face_normal_y.push_back(0); m.face_normal_z.push_back(0); m.face_area.push_back(area);
        const GeometryReal distance = neighbour < 0 ? GeometryReal{0.5} : GeometryReal{1};
        f.geometry.face_normal_dist.push_back(distance);
        f.geometry.face_area_over_normal_dist.push_back(area/distance);
        f.geometry.face_interp_weight.push_back(neighbour < 0 ? 0 : GeometryReal{0.5});
    };
    for (LocalIndex cell = 0; cell+1 < count; ++cell)
        for (int copy = 0; copy < (duplicate ? 2 : 1); ++copy)
            face(cell, cell+1, static_cast<GeometryReal>(f.first+cell+1), 1, duplicate ? GeometryReal{0.5} : GeometryReal{1}, -1);
    m.n_internal_faces = static_cast<LocalIndex>(m.face_owner.size());
    for (std::size_t p = 0; p < m.nb_ranks.size(); ++p) {
        const bool left = m.nb_ranks[p] < comm.rank();
        face(left ? 0 : count-1, m.recv_ghost_local[p], static_cast<GeometryReal>(left ? f.first : f.first+count), left ? -1 : 1, 1, -1);
    }
    m.n_inner_faces = static_cast<LocalIndex>(m.face_owner.size());
    if (count != 0 && comm.rank() == 0) face(0, -1, 0, -1, 1, 0);
    if (count != 0 && comm.rank() == active-1) face(count-1, -1, static_cast<GeometryReal>(f.total), 1, 1, 1);
    m.n_faces = static_cast<LocalIndex>(m.face_owner.size());
    auto& c = f.connectivity;
    c.active_mask = mesh::AuxConnType::CellFaces; c.n_faces = m.n_faces; c.n_cells_own = m.n_own;
    c.cell_faces_offsets = {0};
    for (LocalIndex cell = 0; cell < count; ++cell) {
        for (LocalIndex entry = 0; entry < m.n_faces; ++entry)
            if (m.face_owner[static_cast<std::size_t>(entry)] == cell || m.face_neigh[static_cast<std::size_t>(entry)] == cell)
                c.cell_faces.push_back(entry);
        c.cell_faces_offsets.push_back(c.cell_faces.size());
    }
    f.geometry.active_mask = mesh::AuxGeomType::FaceNormalDistance | mesh::AuxGeomType::FaceAreaOverNormalDistance |
                             mesh::AuxGeomType::FaceInterpolationWeights;
    f.geometry.n_faces = m.n_faces;
    return f;
}



struct Attached {
    mesh::compute::MeshStorage<Exec> mesh;
    mesh::compute::AuxConnectivityStorage<Exec> connectivity;
    cv::ConvectionPlan<Exec> plan;
    linalg::FaceAssemblyPlan<Exec> assembly;
    Attached(const Fixture& f, mpi::CommView comm, const Exec& exec)
        : mesh(f.host, {}, exec), connectivity(f.host, f.connectivity, exec),
          plan(f.host, mesh.view(), connectivity.view(), comm, exec), assembly(f.host, comm, exec) {}
};

struct VelocityFlux {
    Real velocity = 1;
    template<class C>
    KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const
    { return velocity*c.nx*c.area; }
};
struct DivergentFlux {
    template<class C>
    KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const { return (Real{1}+c.x)*c.nx*c.area; }
};
template<class View>
auto snapshot(View view, const Exec& exec)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view);
    exec.fence("Convection test snapshot ready");
    return host;
}
void conditions(Boundary& boundary, Real left = 2, Real right = 7)
{ boundary.set(0, bc::FixedValue<Real>{left}); boundary.set(1, bc::FixedValue<Real>{right}); }

// NVCC extended lambdas cannot be defined in constructors; use a named kernel here.
template<class R, class Mesh, class Values, class Gradients, class Factors>
struct InitializeInputs {
    Mesh mesh;
    Values values;
    Gradients gradients;
    Factors factors;
    bool owned_only = false;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        const R x = static_cast<R>(mesh.geometry.cell_centroid_x(i));
        const R y = static_cast<R>(mesh.geometry.cell_centroid_y(i));
        if (owned_only && cell >= mesh.sizes.n_own) {
            for (int c = 0; c < 2; ++c) {
                values(i,c) = Kokkos::quiet_NaN_v<R>;
                factors(i,c) = Kokkos::quiet_NaN_v<R>;
            }
            for (int c = 0; c < 6; ++c) gradients(i,c) = Kokkos::quiet_NaN_v<R>;
        } else {
            values(i,0) = R{0.5}+x;
            values(i,1) = R{2}+R{2}*x+static_cast<R>(0.3)*y;
            gradients(i,0) = R{1}; gradients(i,1) = gradients(i,2) = R{0};
            gradients(i,3) = R{2}; gradients(i,4) = static_cast<R>(0.3); gradients(i,5) = R{0};
            factors(i,0) = factors(i,1) = R{1};
        }
    }
};

template<class R = double, class Layout = Kokkos::LayoutLeft>
struct Inputs {
    fields::FieldStorage<R, 2, Exec, Layout> values, factors;
    fields::FieldStorage<R, 6, Exec, Layout> gradients;
    Inputs(const Attached& a, const Exec& exec, bool owned_only = false)
        : values("convection values", static_cast<std::size_t>(a.mesh.view().sizes.n_cells), fields::FieldLocation::Custom, exec),
          factors("convection factors", values.view().size(), fields::FieldLocation::Custom, exec),
          gradients("convection gradients", values.view().size(), fields::FieldLocation::Custom, exec)
    {
        auto m = a.mesh.view();
        auto q = values.view(); auto g = gradients.view(); auto t = factors.view();
        using Kernel = InitializeInputs<R, decltype(m), decltype(q), decltype(g), decltype(t)>;
        Kokkos::parallel_for("test/convection_inputs", Kokkos::RangePolicy<Exec>(exec, 0, m.sizes.n_cells),
            Kernel{m,q,g,t,owned_only});
    }
    auto data()
    { return rec::data(rec::field_pack(values.view()), rec::gradient_pack(gradients.view()), rec::field_pack(factors.view())); }
};

template<class Matrix, class Field, class Output>
void compare_system_residual(Matrix& matrix, linalg::Vector<typename Matrix::scalar_type, Exec>& rhs,
    Field field, Output residual, const Exec& exec, double tolerance = 1e-11, int component = 0)
{
    using S = typename Matrix::scalar_type;
    linalg::Vector<S, Exec> x(matrix.distribution()), ax(matrix.distribution());
    const auto local_x = x.overwrite(exec);
    Kokkos::deep_copy(exec, local_x, S{0});
    const auto n = field.extent(0);
    const auto width = matrix.block_size();
    const auto rows = matrix.distribution().local_size()/static_cast<std::size_t>(width);
    check(n >= rows, "Test field is shorter than owned rows");
    Kokkos::parallel_for("test/convection_algebra_input", Kokkos::RangePolicy<Exec>(exec, 0, rows),
        KOKKOS_LAMBDA(std::size_t i) { local_x(i*static_cast<std::size_t>(width)+static_cast<std::size_t>(component),0) = static_cast<S>(field(i)); });
    x.commit();
    matrix.apply(x, ax);
    const auto hax = snapshot(Kokkos::subview(ax.read(exec), Kokkos::ALL(), 0), exec); ax.commit();
    const auto hb = snapshot(Kokkos::subview(rhs.read(exec), Kokkos::ALL(), 0), exec); rhs.commit();
    const auto hr = snapshot(residual, exec);
    for (std::size_t i = 0; i < rows; ++i) {
        const auto k = i*static_cast<std::size_t>(width)+static_cast<std::size_t>(component);
        near(static_cast<double>(hr(i)), static_cast<double>(hb(k)-hax(k)), tolerance,
            "Convection matrix-free residual differs from b-A*phi");
    }
}

void upwind(const Exec& exec)
{
    const auto comm = mpi::CommView::world(); const auto f = bar(comm); Attached a(f, comm, exec);
    Boundary boundary(f.host, a.mesh.view(), exec); conditions(boundary, 2, 7);
    cv::ConvectionOperator convection(a.plan); ts::AssemblyStatus<Exec> status(exec);
    Inputs input(a, exec);
    const auto entries = a.assembly.entry_map();
    const auto diagonal = snapshot(entries.diagonal, exec);
    const auto owner = snapshot(entries.owner_entries, exec), neighbour = snapshot(entries.neighbour_entries, exec);
    for (const Real velocity : {Real{3}, Real{-3}}) {
        linalg::CsrMatrix<double, Exec> matrix(a.assembly.pattern()); linalg::Vector<double, Exec> rhs(matrix.distribution());
        const auto target = ts::make_system_view(matrix, rhs, a.assembly, exec);
        Kokkos::deep_copy(exec, target.values, 0.0); Kokkos::deep_copy(exec, target.rhs, 0.0);
        status.begin_async(); convection.add_system_async(target, boundary, status, VelocityFlux{velocity});
        (void)status.check(comm); matrix.commit(); rhs.commit();
        const auto values = snapshot(matrix.read_values(exec), exec); matrix.commit();
        const auto b = snapshot(Kokkos::subview(rhs.read(exec), Kokkos::ALL(), 0), exec); rhs.commit();
        for (std::size_t i = 0; i < diagonal.extent(0); ++i) {
            near(values(diagonal(i)), 3.0, 0.0, "Upwind diagonal sign");
            const bool inflow = velocity > 0 ? f.first+static_cast<int>(i) == 0 : f.first+static_cast<int>(i)+1 == f.total;
            near(b(i), inflow ? (velocity > 0 ? 6.0 : 21.0) : 0.0, 0.0, "Inflow RHS");
        }
        for (LocalIndex face = 0; face < f.host.n_inner_faces; ++face) {
            const auto k = static_cast<std::size_t>(face);
            const double m = static_cast<double>(velocity*f.host.face_normal_x[k]);
            near(values(owner(k)), m < 0 ? m : 0.0, 0.0, "Owner off-diagonal sign");
            if (face < f.host.n_internal_faces)
                near(values(neighbour(k)), m > 0 ? -m : 0.0, 0.0, "Neighbour off-diagonal sign");
        }
        Kokkos::View<double*, Space> residual("upwind residual", f.host.n_own);
        Kokkos::deep_copy(exec, residual, 0.0);
        status.begin_async(); convection.add_residual_async(input.values.view().component(1), residual, boundary, status, VelocityFlux{velocity});
        (void)status.check(comm);
        compare_system_residual(matrix, rhs, input.values.view().component(1), residual, exec);
    }
}

template<class Scheme, class R = double, class Layout = Kokkos::LayoutLeft>
void affine_scheme(const Exec& exec, Scheme scheme)
{
    const auto comm = mpi::CommView::world(); auto f = bar(comm);
    f.host.face_centroid_y.assign(static_cast<std::size_t>(f.host.n_faces), static_cast<GeometryReal>(0.2));
    Attached a(f, comm, exec); Boundary boundary(f.host, a.mesh.view(), exec);
    cv::ConvectionOperator convection(a.plan); ts::AssemblyStatus<Exec> status(exec); Inputs<R,Layout> input(a, exec);
    const auto reconstruction = cv::make_reconstruction(a.plan, scheme, input.data(), 1);
    const double tolerance = std::is_same_v<R,float> ? 3e-6 : 2e-11;
    for (const Real velocity : {Real{3}, Real{-3}}) {
        conditions(boundary, velocity > 0 ? static_cast<Real>(2.06) : Real{900},
            velocity < 0 ? static_cast<Real>(2.06)+Real{2}*static_cast<Real>(f.total) : Real{-900});
        linalg::CsrMatrix<double,Exec> base(a.assembly.pattern()), matrix(a.assembly.pattern());
        linalg::Vector<double,Exec> base_rhs(base.distribution()), rhs(matrix.distribution());
        const auto plain = ts::make_system_view(base, base_rhs, a.assembly, exec);
        const auto target = ts::make_system_view(matrix, rhs, a.assembly, exec);
        Kokkos::deep_copy(exec, plain.values, 0.0); Kokkos::deep_copy(exec, plain.rhs, 0.0);
        Kokkos::deep_copy(exec, target.values, 0.0); Kokkos::deep_copy(exec, target.rhs, 0.0);
        status.begin_async();
        convection.add_system_async(plain, boundary, status, VelocityFlux{velocity});
        convection.add_system_async(target, boundary, status, VelocityFlux{velocity}, reconstruction);
        (void)status.check(comm); base.commit(); base_rhs.commit(); matrix.commit(); rhs.commit();
        const auto a0 = snapshot(base.read_values(exec), exec); base.commit();
        const auto a1 = snapshot(matrix.read_values(exec), exec); matrix.commit();
        for (std::size_t i = 0; i < a0.extent(0); ++i) near(a0(i), a1(i), 0.0, "Reconstruction changed the upwind matrix");
        Kokkos::View<double*,Space> residual("affine residual", f.host.n_own);
        Kokkos::deep_copy(exec, residual, 0.0); status.begin_async();
        convection.add_residual_async(input.values.view().component(1), residual, boundary, status, VelocityFlux{velocity}, reconstruction);
        (void)status.check(comm);
        compare_system_residual(matrix, rhs, input.values.view().component(1), residual, exec, tolerance);
        const auto h = snapshot(residual, exec);
        for (LocalIndex i = 0; i < f.host.n_own; ++i)
            near(h(static_cast<std::size_t>(i)), -2.0*static_cast<double>(velocity), tolerance, "Skew affine convective divergence");
        const auto revision = matrix.revision();
        Kokkos::View<double*,Space> corrected("correction only", f.host.n_own+2);
        Kokkos::deep_copy(exec, corrected, 17.0);
        const auto prefix = Kokkos::subview(corrected, std::pair<std::size_t,std::size_t>{0,static_cast<std::size_t>(f.host.n_own)});
        Kokkos::deep_copy(exec, prefix, Kokkos::subview(base_rhs.read(exec), Kokkos::ALL(), 0)); base_rhs.commit();
        status.begin_async(); convection.add_correction_async(prefix, boundary, status, VelocityFlux{velocity}, reconstruction); (void)status.check(comm);
        const auto hc = snapshot(corrected, exec);
        const auto hb = snapshot(Kokkos::subview(rhs.read(exec), Kokkos::ALL(), 0), exec); rhs.commit();
        for (std::size_t i = 0; i < hb.extent(0); ++i) near(hc(i), hb(i), tolerance, "Correction-only RHS differs from full assembly");
        near(hc(static_cast<std::size_t>(f.host.n_own)), 17.0, 0.0, "Correction overwrote output tail");
        check(matrix.revision() == revision, "Correction-only update touched matrix revision");
    }
}
void linear(const Exec& e) { affine_scheme(e, rec::SecondOrderUpwind{}); }
void tvd(const Exec& e) { affine_scheme(e, rec::Tvd{}); }
void muscl(const Exec& e) { affine_scheme(e, rec::MusclKappa{}); }
void strided_float(const Exec& e) { affine_scheme<rec::Tvd<>,float,Kokkos::LayoutRight>(e, rec::Tvd{}); }

void limiting(const Exec& exec)
{
    const auto comm = mpi::CommView::world(); const auto f = bar(comm); Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec); conditions(boundary);
    cv::ConvectionOperator convection(a.plan); ts::AssemblyStatus<Exec> status(exec); Inputs input(a,exec);
    input.factors.fill(exec,0.0); input.gradients.fill(exec,Kokkos::quiet_NaN_v<double>);
    const auto reconstruction = cv::make_reconstruction(a.plan,rec::Limited<rec::Tvd<>>{},input.data(),1);
    Kokkos::View<double*,Space> plain("plain",f.host.n_own),limited("limited",f.host.n_own);
    Kokkos::deep_copy(exec,plain,0.0);Kokkos::deep_copy(exec,limited,0.0);
    status.begin_async();
    convection.add_residual_async(input.values.view().component(1),plain,boundary,status,VelocityFlux{});
    convection.add_residual_async(input.values.view().component(1),limited,boundary,status,VelocityFlux{},reconstruction);
    (void)status.check(comm);
    const auto hp=snapshot(plain,exec),hl=snapshot(limited,exec);
    for(std::size_t i=0;i<hp.extent(0);++i)near(hp(i),hl(i),0.0,"Zero limiter did not recover upwind without gradient reads");
}

struct InvalidReconstruction {
    template<class R> KOKKOS_INLINE_FUNCTION R interior(const cv::InteriorFaceContext<R>&,bool) const { return Kokkos::quiet_NaN_v<R>; }
    template<class R> KOKKOS_INLINE_FUNCTION R boundary(const boundary::FaceContext<R>&) const { return Kokkos::quiet_NaN_v<R>; }
};
void zero_flux(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);boundary.set(0,bc::FixedFlux<Real>{1});boundary.set(1,bc::NormalGradient<Real>{1});
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);
    Kokkos::View<double*,Space> field("poisoned field",f.host.n_cells),out("unchanged output",f.host.n_own);
    Kokkos::deep_copy(exec,field,Kokkos::quiet_NaN_v<double>);Kokkos::deep_copy(exec,out,7.0);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);
    status.begin_async();
    convection.add_system_async(target,boundary,status,cv::UniformMassFlux<>{0},InvalidReconstruction{});
    convection.add_residual_async(field,out,boundary,status,cv::UniformMassFlux<>{0},InvalidReconstruction{});
    convection.add_correction_async(out,boundary,status,cv::UniformMassFlux<>{0},InvalidReconstruction{});
    (void)status.check(comm);matrix.commit();rhs.commit();
    const auto h=snapshot(out,exec);for(std::size_t i=0;i<h.extent(0);++i)near(h(i),7.0,0.0,"Zero flux read unused data or modified residual");
}

struct TimedFlux {
    template<class C>KOKKOS_INLINE_FUNCTION Real operator()(const C& c)const{return (Real{1}+c.time)*c.nx*c.area;}
};
struct TimedInflow {
    template<class C,class Condition>KOKKOS_INLINE_FUNCTION Real operator()(const C& c,const Condition&)const{return Real{2}+c.time+c.x;}
};
void inflow(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);boundary.set(0,bc::FixedFlux<Real>{10});boundary.set(1,bc::NormalGradient<Real>{20});
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);Inputs input(a,exec);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);status.begin_async();
    convection.add_system_async(target,boundary,status,TimedFlux{},cv::NoReconstruction{},TimedInflow{},Real{2});
    (void)status.check(comm);matrix.commit();rhs.commit();
    Kokkos::View<double*,Space> residual("custom inflow residual",f.host.n_own);Kokkos::deep_copy(exec,residual,0.0);status.begin_async();
    convection.add_residual_async(input.values.view().component(1),residual,boundary,status,TimedFlux{},cv::NoReconstruction{},TimedInflow{},Real{2});
    (void)status.check(comm);compare_system_residual(matrix,rhs,input.values.view().component(1),residual,exec);
    const auto hb=snapshot(Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0),exec);rhs.commit();
    if(f.host.n_own>0&&f.first==0)near(hb(0),12.0,0.0,"Time-dependent inflow callback was not used");
}

void face_inputs(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);
    boundary.set(0,bc::FixedFlux<Real>{10});boundary.set(1,bc::NormalGradient<Real>{20});
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);Inputs input(a,exec);
    Kokkos::View<double**,Kokkos::LayoutRight,Space> storage("strided face inputs",f.host.n_faces,3);
    const auto mesh=a.mesh.view();
    for(const Real velocity:{Real{3},Real{-3}}){
        Kokkos::parallel_for("test/face_input_values",Kokkos::RangePolicy<Exec>(exec,0,f.host.n_faces),
            KOKKOS_LAMBDA(LocalIndex face){
                const auto k=static_cast<std::size_t>(face);
                const double mass=static_cast<double>(velocity*mesh.geometry.face_normal_x(k));
                storage(k,0)=mass;
                storage(k,1)=face>=mesh.sizes.n_inner_faces&&mass<0 ? 4.0 : Kokkos::quiet_NaN_v<double>;
                storage(k,2)=19.0;
            });
        const auto flux=cv::FaceMassFlux{Kokkos::subview(storage,Kokkos::ALL(),0)};
        const auto incoming=cv::FaceInflowValues{Kokkos::subview(storage,Kokkos::ALL(),1)};
        linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
        const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
        Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);status.begin_async();
        convection.add_system_async(target,boundary,status,flux,cv::NoReconstruction{},incoming);
        (void)status.check(comm);matrix.commit();rhs.commit();
        Kokkos::View<double*,Space> residual("face input residual",f.host.n_own);
        Kokkos::deep_copy(exec,residual,0.0);status.begin_async();
        convection.add_residual_async(input.values.view().component(1),residual,boundary,status,flux,cv::NoReconstruction{},incoming);
        (void)status.check(comm);compare_system_residual(matrix,rhs,input.values.view().component(1),residual,exec);
        const auto hb=snapshot(Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0),exec);rhs.commit();
        for(std::size_t i=0;i<hb.extent(0);++i){
            const bool inlet=velocity>0 ? f.first+static_cast<int>(i)==0 : f.first+static_cast<int>(i)+1==f.total;
            near(hb(i),inlet?12.0:0.0,0.0,"Strided inflow input or flow reversal failed");
        }
    }
}

void divergent(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary,3,3);
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);
    Kokkos::View<double*,Space> field("constant",f.host.n_cells),out("divergent residual",f.host.n_own);
    Kokkos::deep_copy(exec,field,3.0);Kokkos::deep_copy(exec,out,0.0);status.begin_async();
    convection.add_residual_async(field,out,boundary,status,DivergentFlux{});(void)status.check(comm);
    const auto h=snapshot(out,exec);for(std::size_t i=0;i<h.extent(0);++i)near(h(i),-3.0,1e-12,"Conservative transport incorrectly removed phi*div(mass flux)");
    Kokkos::deep_copy(exec,out,0.0);status.begin_async();convection.add_residual_async(field,out,boundary,status,VelocityFlux{3});(void)status.check(comm);
    const auto hc=snapshot(out,exec);for(std::size_t i=0;i<hc.extent(0);++i)near(hc(i),0.0,0.0,"Constant field changed under divergence-free mass flux");
}

void bsr(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary,2,Real{-999});
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);Inputs<float,Kokkos::LayoutRight> input(a,exec);
    const auto reconstruction=cv::make_reconstruction(a.plan,rec::SecondOrderUpwind{},input.data(),1);
    linalg::BsrMatrix<double,Exec> matrix(a.assembly.pattern(),3);linalg::Vector<double,Exec> rhs(matrix.distribution());
    Kokkos::deep_copy(exec,matrix.overwrite_blocks(exec),0.0);Kokkos::deep_copy(exec,rhs.overwrite(exec),0.0);
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec,1);
    status.begin_async();convection.add_system_async(target,boundary,status,VelocityFlux{3},reconstruction);(void)status.check(comm);matrix.commit();rhs.commit();
    Kokkos::View<double*,Space> residual("bsr residual",f.host.n_own);Kokkos::deep_copy(exec,residual,0.0);status.begin_async();
    convection.add_residual_async(input.values.view().component(1),residual,boundary,status,VelocityFlux{3},reconstruction);(void)status.check(comm);
    compare_system_residual(matrix,rhs,input.values.view().component(1),residual,exec,3e-6,1);
    const auto values=snapshot(matrix.read_blocks(exec),exec);matrix.commit();
    for(std::size_t i=0;i<values.extent(0);++i)for(int r=0;r<3;++r)for(int c=0;c<3;++c)
        if(r!=1||c!=1)near(values(i,r,c),0.0,0.0,"Convection modified another BSR component");
}

void additive(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm,true);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary);
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,4.0);Kokkos::deep_copy(exec,target.rhs,5.0);
    status.begin_async();convection.add_system_async(target,boundary,status,VelocityFlux{3});convection.add_system_async(target,boundary,status,VelocityFlux{3});(void)status.check(comm);
    matrix.commit();rhs.commit();const auto v=snapshot(matrix.read_values(exec),exec);matrix.commit();
    const auto d=snapshot(a.assembly.entry_map().diagonal,exec);
    for(std::size_t i=0;i<d.extent(0);++i)near(v(d(i)),10.0,0.0,"Convection cleared an existing coefficient or miscounted duplicate faces");
    const auto hb=snapshot(Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0),exec);rhs.commit();
    for(std::size_t i=0;i<hb.extent(0);++i)near(hb(i),f.first+static_cast<int>(i)==0?17.0:5.0,0.0,"Convection cleared the existing RHS");
}

struct QuickStencil {
    mesh::compute::MeshView<Space> mesh;
    KOKKOS_INLINE_FUNCTION auto side(std::size_t cell,std::size_t face)const
    {
        const auto other=cell==0?std::size_t{1}:std::size_t{0};
        const auto last=std::size_t{3}-cell-other;
        const Kokkos::Array<std::size_t,3> samples{other,cell,last};
        const auto& g=mesh.geometry;
        return rec::cell_average_quick_stencil(samples,
            Kokkos::Array<double,3>{g.cell_centroid_x(other),g.cell_centroid_x(cell),g.cell_centroid_x(last)},
            Kokkos::Array<double,3>{1.0/12,1.0/12,1.0/12},static_cast<double>(g.face_centroid_x(face)));
    }
    KOKKOS_INLINE_FUNCTION auto face(std::size_t f)const
    {return rec::FaceStencil{side(static_cast<std::size_t>(mesh.connectivity.face_owner(f)),f),side(static_cast<std::size_t>(mesh.connectivity.face_neigh(f)),f)};}
    KOKKOS_INLINE_FUNCTION auto boundary_side(std::size_t f)const
    {return side(static_cast<std::size_t>(mesh.connectivity.face_owner(f)),f);}
};
struct Hessians {
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t,std::size_t)const{return rec::SymmetricTensor<double>{2,0,0,0,0,0};}
};
struct Moments {
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t)const{return rec::SymmetricTensor<double>{1.0/12,0,0,0,0,0};}
};
template<bool Quick>
void quadratic_case(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    check(comm.size()==1,"Explicit three-cell polynomial fixture is a serial case");
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary,0,static_cast<Real>(f.total*f.total));
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);
    Kokkos::View<double*,Space> phi("quadratic averages",f.host.n_cells),gx("quadratic gradient",f.host.n_cells),zero("zero",f.host.n_cells);
    const auto mesh=a.mesh.view();
    Kokkos::parallel_for("test/quadratic_input",Kokkos::RangePolicy<Exec>(exec,0,f.host.n_cells),KOKKOS_LAMBDA(LocalIndex cell){
        const auto i=static_cast<std::size_t>(cell);const double x=mesh.geometry.cell_centroid_x(i);
        phi(i)=x*x+1.0/12;gx(i)=2*x;zero(i)=0;
    });
    const auto reconstruction=[&]{
        if constexpr(Quick)return cv::make_reconstruction(rec::Quick{},rec::data(rec::pack(phi)),QuickStencil{mesh});
        else return cv::make_reconstruction(a.plan,rec::Quadratic{},rec::data(rec::pack(phi),
            rec::gradients(rec::pack(gx),rec::pack(zero),rec::pack(zero)),rec::UnitFactors{},Hessians{},Moments{}));
    }();
    for(const Real velocity:{Real{3},Real{-3}}){
        linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
        const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
        Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);status.begin_async();
        convection.add_system_async(target,boundary,status,VelocityFlux{velocity},reconstruction);(void)status.check(comm);matrix.commit();rhs.commit();
        Kokkos::View<double*,Space> residual("quadratic residual",f.host.n_own);Kokkos::deep_copy(exec,residual,0.0);status.begin_async();
        convection.add_residual_async(phi,residual,boundary,status,VelocityFlux{velocity},reconstruction);(void)status.check(comm);
        compare_system_residual(matrix,rhs,phi,residual,exec);
        const auto h=snapshot(residual,exec);
        for(std::size_t i=0;i<h.extent(0);++i)near(h(i),-2*static_cast<double>(velocity)*(static_cast<double>(i)+0.5),2e-11,"Quadratic cell-average advection is not exact");
    }
}
void quick(const Exec& e){quadratic_case<true>(e);}
void quadratic(const Exec& e){quadratic_case<false>(e);}
void frozen_system(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary,0,static_cast<Real>(f.total));
    mesh::compute::AuxGeometryStorage<Exec> geometry(f.host,f.geometry,exec);
    df::DiffusionPlan<Exec> diffusion_plan(f.host,a.mesh.view(),a.connectivity.view(),geometry.view(),comm,exec);
    df::DiffusionOperator diffusion(diffusion_plan);cv::ConvectionOperator convection(a.plan);
    ts::SourceOperator source(a.mesh.view(),exec);ts::AssemblyStatus<Exec> status(exec);
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{Real{0.5}}};
    Inputs input(a,exec);input.values.fill(exec,0.0);input.gradients.fill(exec,0.0);
    const auto reconstruction=cv::make_reconstruction(a.plan,rec::SecondOrderUpwind{},input.data());
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution()),x(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);status.begin_async();
    diffusion.add_system_async(target,boundary,status,law);
    convection.add_system_async(target,boundary,status,VelocityFlux{});
    source.add_async(target.rhs,status,ts::UniformSource<Real>{1});
    const auto report=status.check(comm);check(report.has_diffusion_anchor,"Convection lost the preceding diffusion anchor");matrix.commit();rhs.commit();
    const auto revision=matrix.revision();
    Kokkos::View<double*,Space> base_rhs("base combined RHS",f.host.n_own);
    Kokkos::deep_copy(exec,base_rhs,Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0));rhs.commit();
    linalg::SolverOptions options;options.method="GMRES";options.relative_tolerance=1e-12;options.preconditioner.type="RELAXATION";
    // This fixture deliberately uses double algebra; vary transport compute precision independently.
    linalg::Solver<decltype(matrix),double> linear_solver(matrix,options);
    for(int iteration=0;iteration<2;++iteration){
        if(iteration==1){
            const auto gradients=input.gradients.view();
            Kokkos::parallel_for("test/frozen_external_gradient",Kokkos::RangePolicy<Exec>(exec,0,f.host.n_cells),
                KOKKOS_LAMBDA(LocalIndex cell){gradients(static_cast<std::size_t>(cell),0)=1;});
        }
        const auto b=Kokkos::subview(rhs.overwrite(exec),Kokkos::ALL(),0);Kokkos::deep_copy(exec,b,base_rhs);
        status.begin_async();convection.add_correction_async(b,boundary,status,VelocityFlux{},reconstruction);(void)status.check(comm);rhs.commit();
        x.copy_from_component(input.values.view().component(0),exec);x.commit();
        check(linear_solver.solve(rhs,x).converged,"Combined convection-diffusion solve failed");
        x.copy_to_component(input.values.view().component(0),exec);x.commit();
        check(matrix.revision()==revision,"Frozen convection correction rebuilt the matrix");
    }
    const auto h=snapshot(input.values.view().component(0),exec);
    for(LocalIndex i=0;i<f.host.n_own;++i)near(h(static_cast<std::size_t>(i)),static_cast<double>(f.first+i)+0.5,2e-10,"Composed scalar equation lost its affine solution");
    Kokkos::View<double*,Space> residual("composed residual",f.host.n_own);Kokkos::deep_copy(exec,residual,0.0);status.begin_async();
    diffusion.add_residual_async(input.values.view().component(0),residual,boundary,status,law);
    convection.add_residual_async(input.values.view().component(0),residual,boundary,status,VelocityFlux{},reconstruction);
    source.add_async(residual,status,ts::UniformSource<Real>{1});(void)status.check(comm);
    const auto r=snapshot(residual,exec);for(std::size_t i=0;i<r.extent(0);++i)near(r(i),0.0,5e-10,"Combined convection-diffusion/source residual is nonzero");
}
void distributed_case(const Exec& exec,bool empty_last=false,bool globally_empty=false)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm,false,empty_last,globally_empty);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary,2,Real{2}+Real{2}*static_cast<Real>(f.total));
    Inputs<double,Kokkos::LayoutRight> input(a,exec,true);
    halo::Exchange<double,Exec> exchange(a.mesh.view(),a.mesh.halo_plan(),comm,53,exec,halo::Transfer::HostStaged);
    exchange.add_field(input.values.view());exchange.add_field(input.gradients.view());exchange.add_field(input.factors.view());exchange.prepare();exchange.exchange();
    const auto reconstruction=cv::make_reconstruction(a.plan,rec::Tvd{},input.data(),1);
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);status.begin_async();
    convection.add_system_async(target,boundary,status,VelocityFlux{},reconstruction);(void)status.check(comm);matrix.commit();rhs.commit();
    Kokkos::View<double*,Space> residual("distributed convection residual",f.host.n_own);Kokkos::deep_copy(exec,residual,0.0);status.begin_async();
    convection.add_residual_async(input.values.view().component(1),residual,boundary,status,VelocityFlux{},reconstruction);(void)status.check(comm);
    compare_system_residual(matrix,rhs,input.values.view().component(1),residual,exec);
    const auto h=snapshot(residual,exec);double local=0,global=0;
    for(std::size_t i=0;i<h.extent(0);++i){near(h(i),-2.0,2e-11,"Incorrect interface donor or stale reconstruction ghosts");local+=h(i);}
    mpi::check(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,comm.native()),"Convection conservation sum");
    near(global,-2.0*static_cast<double>(f.total),3e-10,"Internal/interface convective fluxes are not conservative");
}
void distributed(const Exec& e){distributed_case(e);distributed_case(e,true);distributed_case(e,true,true);}
void empty(const Exec& e){distributed_case(e,true,true);}
enum class Reject { MassFlux,Inflow,ShortFlux,ShortInflow,ShortGradient,ShortFactors,Component,Reconstruction,InPlace,Pending,Overflow,Underflow,Batch,Time };
void rejection(const Exec& exec,Reject which)
{
    const auto comm=mpi::CommView::world();const auto f=bar(comm);Attached a(f,comm,exec);
    Boundary boundary(f.host,a.mesh.view(),exec);conditions(boundary,4,7);
    if(which==Reject::Inflow)boundary.set(0,bc::NormalGradient<Real>{0});
    cv::ConvectionOperator convection(a.plan);ts::AssemblyStatus<Exec> status(exec);Inputs input(a,exec);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);
    if(which!=Reject::Pending)status.begin_async();
    if(which==Reject::MassFlux)convection.add_system_async(target,boundary,status,cv::UniformMassFlux<>{Kokkos::quiet_NaN_v<Real>});
    else if(which==Reject::ShortFlux){Kokkos::View<Real*,Space> flux("short flux",f.host.n_faces-1);convection.add_system_async(target,boundary,status,cv::FaceMassFlux{flux});}
    else if(which==Reject::ShortInflow){Kokkos::View<Real*,Space> values("short inflow",f.host.n_faces-1);convection.add_system_async(target,boundary,status,VelocityFlux{},cv::NoReconstruction{},cv::FaceInflowValues{values});}
    else if(which==Reject::ShortGradient){
        fields::FieldStorage<double,6,Exec> short_g("short gradients",static_cast<std::size_t>(f.host.n_cells-1),fields::FieldLocation::Custom,exec);
        const auto r=cv::make_reconstruction(a.plan,rec::SecondOrderUpwind{},rec::data(rec::field_pack(input.values.view()),rec::gradient_pack(short_g.view())));
        convection.add_system_async(target,boundary,status,VelocityFlux{},r);
    }else if(which==Reject::ShortFactors){
        fields::FieldStorage<double,1,Exec> short_f("short factors",static_cast<std::size_t>(f.host.n_cells-1),fields::FieldLocation::Custom,exec);
        const auto r=cv::make_reconstruction(a.plan,rec::LimitedLinear{},rec::data(rec::field_pack(input.values.view()),rec::gradient_pack(input.gradients.view()),rec::field_pack(short_f.view())));
        convection.add_system_async(target,boundary,status,VelocityFlux{},r);
    }else if(which==Reject::Component){const auto r=cv::make_reconstruction(a.plan,rec::SecondOrderUpwind{},input.data(),2);convection.add_system_async(target,boundary,status,VelocityFlux{},r);}
    else if(which==Reject::Reconstruction)convection.add_system_async(target,boundary,status,VelocityFlux{},InvalidReconstruction{});
    else if(which==Reject::InPlace)convection.add_residual_async(target.rhs,target.rhs,boundary,status,VelocityFlux{});
    else if(which==Reject::Overflow)convection.add_system_async(target,boundary,status,VelocityFlux{Kokkos::finite_max_v<Real>/Real{2}});
    else if(which==Reject::Underflow){
        cv::ConvectionPlan<Exec,double> wide_plan(f.host,a.mesh.view(),a.connectivity.view(),comm,exec);
        cv::ConvectionOperator<Exec,double> wide(wide_plan);
        linalg::CsrMatrix<float,Exec> small(a.assembly.pattern());linalg::Vector<float,Exec> small_rhs(small.distribution());
        const auto t=ts::make_system_view(small,small_rhs,a.assembly,exec);Kokkos::deep_copy(exec,t.values,0.0F);Kokkos::deep_copy(exec,t.rhs,0.0F);
        wide.add_system_async(t,boundary,status,cv::UniformMassFlux<double>{Kokkos::norm_min_v<double>});(void)status.check(comm);small.commit();small_rhs.commit();
    }else if(which==Reject::Batch){ts::SourceOperator source(a.mesh.view(),exec);source.add_async(target.rhs,status,ts::UniformSource<Real>{Kokkos::quiet_NaN_v<Real>});convection.add_system_async(target,boundary,status,VelocityFlux{});}
    else if(which==Reject::Time)convection.add_system_async(target,boundary,status,VelocityFlux{},cv::NoReconstruction{},cv::FixedValueInflow{},Kokkos::quiet_NaN_v<Real>);
    else convection.add_system_async(target,boundary,status,VelocityFlux{});
    (void)status.check(comm);matrix.commit();rhs.commit();
    check(false,"Invalid convection input was accepted");
}
#define CONVECTION_REJECT_CASE(name,reason) void name(const Exec& exec){rejection(exec,Reject::reason);}
CONVECTION_REJECT_CASE(reject_mass_flux,MassFlux)
CONVECTION_REJECT_CASE(reject_inflow,Inflow)
CONVECTION_REJECT_CASE(reject_short_flux,ShortFlux)
CONVECTION_REJECT_CASE(reject_short_inflow,ShortInflow)
CONVECTION_REJECT_CASE(reject_short_gradient,ShortGradient)
CONVECTION_REJECT_CASE(reject_short_factors,ShortFactors)
CONVECTION_REJECT_CASE(reject_component,Component)
CONVECTION_REJECT_CASE(reject_reconstruction,Reconstruction)
CONVECTION_REJECT_CASE(reject_in_place,InPlace)
CONVECTION_REJECT_CASE(reject_pending,Pending)
CONVECTION_REJECT_CASE(reject_overflow,Overflow)
CONVECTION_REJECT_CASE(reject_underflow,Underflow)
CONVECTION_REJECT_CASE(reject_batch,Batch)
CONVECTION_REJECT_CASE(reject_time,Time)
#undef CONVECTION_REJECT_CASE
template<auto Function>
void with_tpetra(const Exec& exec)
{
    mpi::guarded([&] {
        Tpetra::ScopeGuard tpetra(nullptr,nullptr,mpi::CommView::world().native());
        test::complete(exec,[&] { Function(exec); });
    });
}
int reject(int argc,char** argv)
{
    const std::array cases{
        test::Case{"reject_mass_flux",with_tpetra<reject_mass_flux>},test::Case{"reject_inflow",with_tpetra<reject_inflow>},
        test::Case{"reject_short_flux",with_tpetra<reject_short_flux>},test::Case{"reject_short_inflow",with_tpetra<reject_short_inflow>},
        test::Case{"reject_short_gradient",with_tpetra<reject_short_gradient>},test::Case{"reject_short_factors",with_tpetra<reject_short_factors>},
        test::Case{"reject_component",with_tpetra<reject_component>},test::Case{"reject_reconstruction",with_tpetra<reject_reconstruction>},
        test::Case{"reject_in_place",with_tpetra<reject_in_place>},test::Case{"reject_pending",with_tpetra<reject_pending>},
        test::Case{"reject_overflow",with_tpetra<reject_overflow>},test::Case{"reject_underflow",with_tpetra<reject_underflow>},
        test::Case{"reject_batch",with_tpetra<reject_batch>},test::Case{"reject_time",with_tpetra<reject_time>}};
    return test::run(argc,argv,"transport.scalar.convection",cases);
}

} // namespace

int main(int argc,char** argv)
{
    std::setvbuf(stdout,nullptr,_IONBF,0);
    const std::array cases{
        test::Case{"upwind",with_tpetra<upwind>},test::Case{"linear",with_tpetra<linear>},test::Case{"tvd",with_tpetra<tvd>},test::Case{"muscl",with_tpetra<muscl>},
        test::Case{"strided_float",with_tpetra<strided_float>},test::Case{"limiting",with_tpetra<limiting>},test::Case{"zero_flux",with_tpetra<zero_flux>},
        test::Case{"inflow",with_tpetra<inflow>},test::Case{"face_inputs",with_tpetra<face_inputs>},test::Case{"divergent",with_tpetra<divergent>},test::Case{"bsr",with_tpetra<bsr>},test::Case{"additive",with_tpetra<additive>},
        test::Case{"quick",with_tpetra<quick>},test::Case{"quadratic",with_tpetra<quadratic>},test::Case{"frozen_system",with_tpetra<frozen_system>},test::Case{"empty",with_tpetra<empty>}};
    const std::array mpi_cases{test::Case{"distributed",with_tpetra<distributed>}};
    if(argc==2&&std::string_view(argv[1])=="distributed")return test::run(argc,argv,"transport.scalar.convection",mpi_cases);
    if(argc==2&&std::string_view(argv[1]).starts_with("reject_"))return reject(argc,argv);
    return test::run(argc,argv,"transport.scalar.convection",cases);
}