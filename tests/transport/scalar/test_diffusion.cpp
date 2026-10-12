#include "common/test_support.hpp"
#include "cartesian_fixture.hpp"
#include "krona/mpi/communicator.hpp"

#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/transport/scalar/steady_diffusion.hpp>

#include <array>
#include <limits>
#include <string>

namespace {
using namespace krona::test;
namespace ts = krona::transport::scalar;
namespace bc = krona::boundary::scalar;
using Calc = krona::ComputeReal;
using Space = typename Exec::memory_space;
using Field = krona::fields::FieldStorage<Real, 1, Exec>;
using MeshStorage = krona::mesh::compute::MeshStorage<Exec>;
using MeshPart = krona::mesh::MeshPart;
using DeviceVector = Kokkos::View<Real*, Space>;

ts::SteadyDiffusionOptions options()
{
    ts::SteadyDiffusionOptions o;
    const bool single = std::is_same_v<krona::SolverReal, float>;
    o.linear.relative_tolerance = single ? 2.0e-6 : 1.0e-12;
    o.linear.absolute_tolerance = single ? 2.0e-5 : 1.0e-12;
    o.linear.maximum_iterations = 400; o.linear.restart_length = 80;
    return o;
}

double tolerance() { return std::is_same_v<Real, float> ? 8.0e-4 : 2.0e-10; }
std::size_t extended(const MeshPart& m)
{ return static_cast<std::size_t>(m.n_cells)+static_cast<std::size_t>(m.n_faces-m.n_inner_faces); }

template<class View>
auto snapshot(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    complete(exec, [&] { Kokkos::deep_copy(exec, host, view); });
    return host;
}

template<class Manager>
void insulated(Manager& boundary)
{ for (int p = 0; p < 6; ++p) boundary.set(p, bc::NormalGradient<Calc>{0}); }

struct AffineValue {
    Calc offset = 1;
    KOKKOS_INLINE_FUNCTION bc::Relation<Calc> operator()(const bc::Context<Calc>& c) const
    { return bc::FixedValue<Calc>{offset + Calc{2}*c.x - Calc{3}*c.y + Calc{4}*c.z}(c); }
};

struct LayeredDiffusion {
    krona::mesh::compute::MeshView<Space> mesh;
    KOKKOS_INLINE_FUNCTION Calc operator()(std::size_t face) const
    {
        const auto x = mesh.geometry.face_centroid_x(face);
        if (x == 0.5) return Calc{8}/Calc{5}; // Harmonic face value at the material interface.
        return x < 0.5 ? Calc{1} : Calc{4};
    }
};

struct NegativeBoundary {
    KOKKOS_INLINE_FUNCTION bc::Relation<Calc> operator()(const bc::Context<Calc>&) const
    { return {0, 0, -1, 0}; }
};

double affine(double x, double y, double z, double offset = 1)
{ return offset+2*x-3*y+4*z; }

void check_result(const krona::linalg::SolveResult& result)
{
    check(result.converged, "True algebraic residual meets the requested threshold");
    check(std::isfinite(result.residual_norm), "Finite residual norm");
    const auto o = options();
    check(result.residual_norm <= std::max(o.linear.absolute_tolerance, o.linear.relative_tolerance*result.rhs_norm),
          "Convergence report agrees with the true residual");
}

template<class Values, class Rhs>
void check_two_cell_matrix(const MeshPart& m, Values values, Rhs rhs, double d)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_own); ++i) {
        // Each cell has one shared face and one Dirichlet x face; all other faces are insulated.
        const auto row_begin = values.graph.row_map(i), row_end = values.graph.row_map(i+1);
        double diagonal = 0, off_diagonal = 0;
        for (auto k = row_begin; k < row_end; ++k) {
            const double a = static_cast<double>(values.values(k));
            if (a > 0) diagonal += a; else off_diagonal += a;
        }
        near(diagonal, 6*d, tolerance(), "Integrated diagonal includes Dirichlet area/h");
        near(off_diagonal, -2*d, tolerance(), "Interior off-diagonal has the negative sign");
        const double value = m.cell_centroid_x[i] < 0.5 ? 1 : 3;
        near(static_cast<double>(rhs(i, 0)), 4*d*value+1, tolerance(), "Source density is integrated by cell volume");
    }
}

void assembly_case(const Exec& exec, bool split)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(2, 1, 1, comm.native());
    if (split) krona::test::diffusion::split_face(m);
    MeshStorage mesh{m, {}, exec};
    bc::Manager<> boundary{m, mesh.view(), exec};
    insulated(boundary); boundary.set(0, bc::FixedValue<Calc>{1}); boundary.set(1, bc::FixedValue<Calc>{3});
    ts::DiffusionOperator<> op{m, mesh.view(), comm, exec};
    op.assemble_async(boundary, ts::UniformSource<Calc>{2}, bc::UniformDiffusivity<Calc>{3});
    check(op.pending(), "Assembly records an outstanding batch");
    op.commit(); op.require_ready();
    {
        auto values = op.matrix().native()->getLocalMatrixHost();
        auto rhs = op.rhs().native()->getLocalViewHost(Tpetra::Access::ReadOnly);
        check_two_cell_matrix(m, values, rhs, 3);
    }
    const auto* graph = op.matrix().pattern().native().getRawPtr();
    const auto revision = op.matrix().revision();
    op.assemble_async(boundary, ts::UniformSource<Calc>{2}, bc::UniformDiffusivity<Calc>{1});
    op.commit();
    check(op.matrix().pattern().native().getRawPtr() == graph, "Reassembly reuses the graph");
    equal(op.matrix().revision(), revision+1, "Reassembly updates the numeric revision");
    {
        auto values = op.matrix().native()->getLocalMatrixHost();
        auto rhs = op.rhs().native()->getLocalViewHost(Tpetra::Access::ReadOnly);
        check_two_cell_matrix(m, values, rhs, 1);
    }
}
void assembly(const Exec& exec) { assembly_case(exec, false); }
void duplicate_faces(const Exec& exec) { assembly_case(exec, true); }

void affine_case(const Exec& exec, int nx)
{   
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(nx, 3, 2, comm.native());
    MeshStorage mesh{m, {}, exec};
    bc::Manager<Exec, Calc, AffineValue> boundary{m, mesh.view(), exec};
    for (int p = 0; p < 6; ++p) boundary.set(p, AffineValue{});
    Field field{"phi", extended(m)+2, krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{0}); });
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, options()};
    for (int pass = 0; pass < 2; ++pass) {
        const Calc offset = pass == 0 ? Calc{1} : Calc{3};
        for (int p = 0; p < 6; ++p) boundary.set(p, AffineValue{offset});
        check_result(solver.solve(boundary));
        const auto h = snapshot(exec, field.view().component(0));
        for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_cells); ++i)
            near(static_cast<double>(h(i)), affine(m.cell_centroid_x[i], m.cell_centroid_y[i], m.cell_centroid_z[i], offset),
                 tolerance(), "Affine solution in owned and scattered MPI ghost cells");
        for (auto face = m.n_inner_faces; face < m.n_faces; ++face) {
            const auto f = static_cast<std::size_t>(face);
            const auto b = static_cast<std::size_t>(m.n_cells)+f-static_cast<std::size_t>(m.n_inner_faces);
            near(static_cast<double>(h(b)), affine(m.face_centroid_x[f], m.face_centroid_y[f], m.face_centroid_z[f], offset),
                 tolerance(), "Boundary tail stores face values");
        }
        equal(h(extended(m)), Real{0}, "Extra field tail is untouched");
        const auto balance = solver.balance(boundary);
        near(balance.source_integral, 0.0, tolerance(), "Zero integrated source");
        near(balance.defect, 0.0, 4*tolerance(), "Global outward flux balance");
    }
}
void affine_3d(const Exec& exec) { affine_case(exec, 6); }
void empty_rank(const Exec& exec) { affine_case(exec, 2); }

void mixed(const Exec& exec)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(10, 1, 1, comm.native());
    MeshStorage mesh{m, {}, exec};
    bc::Manager<> boundary{m, mesh.view(), exec}; insulated(boundary);
    boundary.set(0, bc::FixedValue<Calc>{1}); boundary.set(1, bc::FixedFlux<Calc>{-6});
    Field field{"phi", extended(m), krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{0}); });
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, options()};
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) boundary.set(1, bc::NormalGradient<Calc>{2});
        check_result(solver.solve(boundary, ts::UniformSource<Calc>{0}, bc::UniformDiffusivity<Calc>{3}));
        const auto h = snapshot(exec, field.view().component(0));
        for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_cells); ++i)
            near(static_cast<double>(h(i)), 1+2*m.cell_centroid_x[i], tolerance(), "Mixed Dirichlet/Neumann linear solution");
        const auto balance = solver.balance(boundary, ts::UniformSource<Calc>{0}, bc::UniformDiffusivity<Calc>{3});
        near(balance.defect, 0.0, 4*tolerance(), "Prescribed flux has the outward sign");
    }
}

void variable(const Exec& exec)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(12, 2, 1, comm.native());
    MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec}; insulated(boundary);
    boundary.set(0, bc::FixedValue<Calc>{1}); boundary.set(1, bc::FixedValue<Calc>{0});
    Field field{"phi", extended(m), krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{0}); });
    DeviceVector d{"D_faces", static_cast<std::size_t>(m.n_faces)};
    auto host_d = Kokkos::create_mirror(d);
    LayeredDiffusion provider{mesh.view()};
    for (std::size_t f = 0; f < host_d.extent(0); ++f) {
        const double x = m.face_centroid_x[f];
        host_d(f) = static_cast<Real>(x == 0.5 ? 1.6 : (x < 0.5 ? 1 : 4));
    }
    complete(exec, [&] { Kokkos::deep_copy(exec, d, host_d); });
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, options()};
    check_result(solver.solve(boundary, ts::UniformSource<Calc>{0}, bc::FaceDiffusivity{d}));
    auto h = snapshot(exec, field.view().component(0));
    for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_cells); ++i) {
        const double x = m.cell_centroid_x[i];
        const double expected = 1-1.6*(x < 0.5 ? x : (0.5+(x-0.5)/4));
        near(static_cast<double>(h(i)), expected, tolerance(), "Layered material solution with harmonic interface D");
    }
    check_result(solver.solve(boundary, ts::UniformSource<Calc>{0}, provider));
    const auto balance = solver.balance(boundary, ts::UniformSource<Calc>{0}, provider);
    near(balance.defect, 0.0, 4*tolerance(), "Variable diffusivity flux balance");
}

double quadratic_error(const Exec& exec, int nx)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(nx, 1, 1, comm.native());
    MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec}; insulated(boundary);
    boundary.set(0, bc::FixedValue<Calc>{0}); boundary.set(1, bc::FixedValue<Calc>{0});
    Field field{"phi", extended(m), krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{0}); });
    DeviceVector source{"source_density", static_cast<std::size_t>(m.n_own)};
    complete(exec, [&] { Kokkos::deep_copy(exec, source, Real{2}); });
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, options()};
    check_result(solver.solve(boundary, ts::CellSource{source}));
    const auto h = snapshot(exec, field.view().component(0));
    double local_error = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_own); ++i) {
        const double x = m.cell_centroid_x[i];
        local_error = std::max(local_error, std::abs(static_cast<double>(h(i))-x*(1-x)));
    }
    double error = 0;
    krona::mpi::check(MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(test error)");
    const auto balance = solver.balance(boundary, ts::CellSource{source});
    near(balance.source_integral, 2.0, tolerance(), "Source is a density, not an integrated RHS");
    near(balance.outward_boundary_flux, 2.0, 4*tolerance(), "Boundary flux balances nonzero source");
    near(error, 0.25/(nx*nx), std::is_same_v<Real, float> ? 2.0e-6 : 2.0e-11,
         "Quadratic solution has the expected two-point boundary discretization error");
    return error;
}
void refinement(const Exec& exec)
{
    const auto coarse = quadratic_error(exec, 8), fine = quadratic_error(exec, 16);
    near(coarse/fine, 4.0, std::is_same_v<Real, float> ? 0.02 : 1.0e-6, "Second-order convergence under refinement");
}

void strided(const Exec& exec)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(8, 1, 1, comm.native());
    MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec}; insulated(boundary);
    boundary.set(0, bc::FixedValue<Calc>{1}); boundary.set(1, bc::FixedValue<Calc>{3});
    krona::fields::FieldStorage<Real, 3, Exec, Kokkos::LayoutRight> field{
        "state", extended(m), krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{-7}); });
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(1), comm, exec, options()};
    check_result(solver.solve(boundary));
    const auto h = snapshot(exec, field.view().kokkos_view());
    for (std::size_t i = 0; i < extended(m); ++i) {
        equal(h(i, 0), Real{-7}, "First unselected component is untouched");
        equal(h(i, 2), Real{-7}, "Third unselected component is untouched");
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_cells); ++i)
        near(static_cast<double>(h(i, 1)), 1+2*m.cell_centroid_x[i], tolerance(), "Strided component is solved correctly");
}

void nonconvergence(const Exec& exec)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(12, 1, 1, comm.native());
    MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec}; insulated(boundary);
    boundary.set(0, bc::FixedValue<Calc>{1}); boundary.set(1, bc::FixedValue<Calc>{0});
    Field field{"phi", extended(m), krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{-7}); });
    auto o = options(); o.linear.maximum_iterations = 1; o.linear.restart_length = 1; o.linear.preconditioner.clear();
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, o};
    const auto result = solver.solve(boundary);
    check(!result.converged, "Insufficient iteration budget reports nonconvergence");
    const auto h = snapshot(exec, field.view().component(0));
    for (std::size_t i = 0; i < extended(m); ++i) equal(h(i), Real{-7}, "Failed solve does not publish a partial field");
}

void all_empty(const Exec& exec)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    MeshPart m; m.faces_grouped = true;
    krona::mpi::check(MPI_Comm_rank(comm.native(), &m.rank), "MPI_Comm_rank(empty)");
    krona::mpi::check(MPI_Comm_size(comm.native(), &m.nprocs), "MPI_Comm_size(empty)");
    MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec};
    Field field{"phi", 0, krona::fields::FieldLocation::Custom, exec};
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, options()};
    check(solver.solve(boundary).converged, "Globally empty system succeeds");
    near(solver.balance(boundary).defect, 0.0, 0.0, "Empty balance is zero");
}

void publication(const Exec& exec)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(1, 1, 1, comm.native());
    MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec};
    const Calc value = Calc{1}/Calc{3};
    for (int p = 0; p < 6; ++p) boundary.set(p, bc::FixedValue<Calc>{value});
    Field field{"phi", extended(m), krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{0}); });
    ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm, exec, options()};
    check_result(solver.solve(boundary));
    const auto h = snapshot(exec, field.view().component(0));
    double local_expected = 0;
    if (m.n_own != 0) {
        near(static_cast<double>(h(0)), static_cast<double>(value), tolerance(), "Published scalar is rounded to StorageReal");
        local_expected = 12*(static_cast<double>(static_cast<Calc>(h(0)))-static_cast<double>(value));
        if constexpr (std::is_same_v<Calc, double> && std::is_same_v<Real, float>)
            check(local_expected != 0, "Mixed publication has a measurable storage-rounding balance defect");
    }
    double expected = 0;
    krona::mpi::check(MPI_Allreduce(&local_expected, &expected, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce(publication)");
    const auto result = solver.balance(boundary);
    near(result.defect, expected, std::is_same_v<Calc, float> ? 2.0e-6 : 1.0e-14,
         "Physical balance uses the published field rather than the solver vector");
}

void subcomm(const Exec& exec)
{
    int rank = 0; krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank(subcomm)");
    MPI_Comm comm = MPI_COMM_NULL;
    const int color = rank%2;
    krona::mpi::check(MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm), "MPI_Comm_split(diffusion)");
    {
        krona::mpi::CommView comm_view(comm);
        auto m = krona::test::diffusion::cartesian(4, 1, 1, comm_view.native());
        MeshStorage mesh{m, {}, exec}; bc::Manager<> boundary{m, mesh.view(), exec}; insulated(boundary);
        boundary.set(0, bc::FixedValue<Calc>{static_cast<Calc>(color+1)});
        boundary.set(1, bc::FixedValue<Calc>{static_cast<Calc>(color+3)});
        Field field{"phi", extended(m), krona::fields::FieldLocation::Custom, exec};
        complete(exec, [&] { field.fill(exec, Real{0}); });
        ts::SteadyDiffusion<> solver{m, mesh.view(), field.view().component(0), comm_view, exec, options()};
        check_result(solver.solve(boundary));
        const auto h = snapshot(exec, field.view().component(0));
        for (std::size_t i = 0; i < static_cast<std::size_t>(m.n_own); ++i)
            near(static_cast<double>(h(i)), color+1+2*m.cell_centroid_x[i], tolerance(), "Independent subcommunicator solution");
    }
    krona::mpi::check(MPI_Comm_free(&comm), "MPI_Comm_free(diffusion subcomm)");
}

void rejection(const Exec& exec, std::string_view name)
{
    krona::mpi::CommView comm = krona::mpi::CommView::world();
    auto m = krona::test::diffusion::cartesian(6, 1, 1, comm.native());
    if (name == "nonorthogonal") {
        m.face_normal_x[0] *= std::sqrt(0.99); m.face_normal_y[0] = 0.1;
    }
    MeshStorage mesh{m, {}, exec};
    bc::Manager<Exec, Calc, NegativeBoundary> boundary{m, mesh.view(), exec}; insulated(boundary);
    if (name != "pure_neumann") boundary.set(0, bc::FixedValue<Calc>{0});
    if (name == "negative_boundary") boundary.set(1, NegativeBoundary{});
    ts::DiffusionOperator<> op{m, mesh.view(), comm, exec};
    if (name == "negative_diffusion") op.assemble_async(boundary, ts::UniformSource<Calc>{0}, bc::UniformDiffusivity<Calc>{-1});
    else if (name == "bad_source") op.assemble_async(boundary, ts::UniformSource<Calc>{std::numeric_limits<Calc>::quiet_NaN()});
    else if (name == "source_extent") {
        DeviceVector source{"short_source", 1}; op.assemble_async(boundary, ts::CellSource{source});
    } else op.assemble_async(boundary);
    if (name == "pending") op.assemble_async(boundary);
    op.commit();
    check(false, "Invalid input should have been rejected");
}
void pure_neumann(const Exec& e) { rejection(e, "pure_neumann"); }
void negative_diffusion(const Exec& e) { rejection(e, "negative_diffusion"); }
void bad_source(const Exec& e) { rejection(e, "bad_source"); }
void negative_boundary(const Exec& e) { rejection(e, "negative_boundary"); }
void source_extent(const Exec& e) { rejection(e, "source_extent"); }
void pending(const Exec& e) { rejection(e, "pending"); }
void nonorthogonal(const Exec& e) { rejection(e, "nonorthogonal"); }
} // namespace

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    try {
        Kokkos::ScopeGuard kokkos{argc, argv};
        Tpetra::ScopeGuard tpetra{&argc, &argv};
        const std::array cases{
            Case{"assembly", assembly}, Case{"duplicate_faces", duplicate_faces}, Case{"affine_3d", affine_3d},
            Case{"empty_rank", empty_rank}, Case{"mixed", mixed}, Case{"variable", variable},
            Case{"refinement", refinement}, Case{"strided", strided}, Case{"nonconvergence", nonconvergence},
            Case{"all_empty", all_empty}, Case{"publication", publication}, Case{"subcomm", subcomm},
            Case{"pure_neumann", pure_neumann},
            Case{"negative_diffusion", negative_diffusion}, Case{"bad_source", bad_source},
            Case{"negative_boundary", negative_boundary}, Case{"source_extent", source_extent},
            Case{"pending", pending}, Case{"nonorthogonal", nonorthogonal}};
        check(argc == 2, "Expected one diffusion test-case name");
        const std::string_view selected = argv[1]; Exec exec{};
        for (const auto& entry : cases) if (entry.name == selected) {
            std::cout << "RUN diffusion." << entry.name << " [" << Exec::name() << "]\n";
            entry.function(exec); exec.fence();
            std::cout << "PASS diffusion." << entry.name << '\n'; return EXIT_SUCCESS;
        }
        check(false, "Unknown diffusion test case");
    } catch (const std::exception& error) { krona::mpi::fatal(error.what()); }
    catch (...) { krona::mpi::fatal("Unhandled diffusion test exception"); }
    return EXIT_FAILURE;
}
