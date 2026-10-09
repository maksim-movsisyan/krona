#include "common/test_support.hpp"

#include <krona/linalg/field_transfer.hpp>
#include <krona/linalg/mesh_adapter.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/fields/field_storage.hpp>

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
using namespace krona::test;
namespace la = krona::linalg;
using GO = la::GlobalOrdinal;
using Scalar = krona::SolverReal;

std::pair<int, int> world()
{
    int rank = 0, ranks = 0;
    krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank(test)");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size(test)");
    return {rank, ranks};
}

template<class T>
void set_vector(la::Vector<T>& vector, const std::vector<T>& values, const Exec& exec)
{
    equal(values.size(), vector.local_size(), "Owned vector input size");
    Kokkos::View<T*, typename Exec::memory_space> input("test/input", values.size());
    auto host = Kokkos::create_mirror_view(input);
    for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
    Kokkos::deep_copy(exec, input, host);
    vector.copy_from_component(input, exec); vector.commit();
}

template<class T>
std::vector<T> get_vector(const la::Vector<T>& vector)
{
    auto host = std::as_const(*vector.native()).getLocalViewHost(Tpetra::Access::ReadOnly);
    std::vector<T> values(vector.local_size());
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i, 0);
    return values;
}

GO gid(int point) { return 17+11*static_cast<GO>(point); }
double exact(int point) { return 1+0.125*point; }
double right_hand_side(int point, int total, double shift = 0)
{
    return (4+shift)*exact(point) - (point > 0 ? exact(point-1) : 0) -
        (point+1 < total ? 0.5*exact(point+1) : 0);
}

struct Chain {
    std::vector<int> points;
    std::vector<GO> owned;
    la::CsrInput input;
    int total = 0;
};

Chain chain(int per_rank = 4, bool empty_rank = false)
{
    const auto [rank, ranks] = world();
    Chain data;
    data.total = empty_rank ? per_rank : per_rank*ranks;
    if (!empty_rank || rank == 0) for (int i = per_rank-1; i >= 0; --i) {
        const int point = empty_rank ? i : rank*per_rank+i;
        data.points.push_back(point); data.owned.push_back(gid(point));
    }
    data.input.row_offsets = {0};
    for (const auto point : data.points) {
        // Deliberately unsorted: upper, diagonal, lower. Native column IDs differ.
        if (point+1 < data.total) data.input.column_gids.push_back(gid(point+1));
        data.input.column_gids.push_back(gid(point));
        if (point > 0) data.input.column_gids.push_back(gid(point-1));
        data.input.row_offsets.push_back(data.input.column_gids.size());
    }
    return data;
}

template<class T>
std::vector<T> coefficients(const Chain& data, double shift = 0)
{
    std::vector<T> values;
    for (const auto point : data.points) {
        if (point+1 < data.total) values.push_back(static_cast<T>(-0.5));
        values.push_back(static_cast<T>(4+shift));
        if (point > 0) values.push_back(T{-1});
    }
    return values;
}

template<class T>
void solve_chain(const Exec& exec, bool empty = false, bool update = false, bool device = false)
{
    const auto data = chain(4, empty);
    la::Distribution<> distribution(data.owned, MPI_COMM_WORLD);
    la::CsrPattern<> pattern(distribution, data.input, exec);
    la::CsrMatrix<T> matrix(pattern);
    auto values = coefficients<T>(data);
    Kokkos::View<T*, typename Exec::memory_space> device_values("test/coefficients", values.size());
    if (device) {
        auto host = Kokkos::create_mirror_view(device_values);
        for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
        Kokkos::deep_copy(exec, device_values, host);
        matrix.set_values_async(device_values, exec); matrix.commit();
    } else matrix.set_values(values);
    la::Vector<T> x(distribution), rhs(distribution), product(distribution);
    std::vector<T> expected, b;
    for (const auto point : data.points) {
        expected.push_back(static_cast<T>(exact(point)));
        b.push_back(static_cast<T>(right_hand_side(point, data.total)));
    }
    set_vector(x, expected, exec); set_vector(rhs, b, exec);
    matrix.apply(x, product);
    const auto actual_product = get_vector(product);
    for (std::size_t i = 0; i < b.size(); ++i)
        near(actual_product[i], b[i], static_cast<T>(64*std::numeric_limits<T>::epsilon()), "Independent nonsymmetric SpMV reference");
    x.fill(static_cast<T>(0.25));
    la::SolverOptions options;
    options.relative_tolerance = std::is_same_v<T, float> ? 2.0e-5 : 1.0e-11;
    la::Solver<T> solver(matrix, options);
    auto result = solver.solve(rhs, x);
    check(result.converged, "Manufactured solution converges");
    check(result.residual_norm <= options.relative_tolerance*result.rhs_norm, "True residual meets the requested tolerance");
    auto solution = get_vector(x);
    for (std::size_t i = 0; i < solution.size(); ++i)
        near(solution[i], expected[i], static_cast<T>(std::is_same_v<T, float> ? 3.0e-4 : 1.0e-9), "Manufactured solution values");
    auto repeated = solver.solve(rhs, x);
    check(repeated.converged && repeated.iterations == 0, "Existing acceptable initial guess is preserved");
    if (update) {
        const auto* graph = matrix.native()->getCrsGraph().getRawPtr();
        const auto revision = matrix.revision();
        values = coefficients<T>(data, 3);
        matrix.set_values(values);
        equal(matrix.native()->getCrsGraph().getRawPtr(), graph, "Coefficient update retains the graph");
        check(matrix.revision() > revision, "Coefficient update invalidates numeric preconditioning");
        b.clear();
        for (const auto point : data.points) b.push_back(static_cast<T>(right_hand_side(point, data.total, 3)));
        set_vector(rhs, b, exec); x.fill(T{0});
        result = solver.solve(rhs, x);
        check(result.converged, "Updated matrix converges with the same solver");
        solution = get_vector(x);
        for (std::size_t i = 0; i < solution.size(); ++i)
            near(solution[i], expected[i], static_cast<T>(std::is_same_v<T, float> ? 3.0e-4 : 1.0e-9), "Updated matrix solution");
        const auto [rank, ranks] = world(); (void)ranks;
        if (rank == 0) matrix.set_values(coefficients<T>(data, 5));
        b.clear();
        for (const auto point : data.points)
            b.push_back(static_cast<T>(right_hand_side(point, data.total, rank == 0 ? 5 : 3)));
        set_vector(rhs, b, exec); x.fill(T{0});
        result = solver.solve(rhs, x);
        check(result.converged, "Local-only coefficient edits trigger a consistent global numeric refresh");
        solution = get_vector(x);
        for (std::size_t i = 0; i < solution.size(); ++i)
            near(solution[i], expected[i], static_cast<T>(std::is_same_v<T, float> ? 3.0e-4 : 1.0e-9), "Locally updated matrix solution");
    }
}

void distribution_case(const Exec&)
{
    const auto [rank, ranks] = world();
    std::array<GO, 2> gids{101+10*static_cast<GO>(rank), 100+10*static_cast<GO>(rank)};
    la::Distribution<> distribution(gids, MPI_COMM_WORLD);
    equal(distribution.local_size(), std::size_t{2}, "Owned count");
    equal(distribution.global_size(), static_cast<Tpetra::global_size_t>(2*ranks), "Global count");
    equal(distribution.global_id(0), gids[0], "Arbitrary GID order is preserved");
    equal(distribution.global_id(1), gids[1], "Second arbitrary GID");
}

void numbering_case(const Exec& exec)
{
    const auto [rank, ranks] = world(); equal(ranks, 2, "Two ranks for numbering");
    krona::mesh::MeshPart mesh;
    mesh.rank = rank; mesh.nprocs = ranks; mesh.n_own = 2; mesh.n_cells = 4;
    mesh.nb_ranks = {1-rank}; mesh.send_offsets = {0, 2}; mesh.recv_offsets = {0, 2};
    mesh.send_owned_local = {0, 1}; mesh.recv_ghost_local = {3, 2};
    mesh.cell_gid = {991, 17, 602, 1003};
    mesh.n_faces = 4; mesh.face_owner = {0, 0, 1, 0}; mesh.face_neigh = {1, 2, 3, -1};
    auto numbering = la::build_cell_numbering(mesh, MPI_COMM_WORLD);
    equal(numbering.cell_to_global[0], static_cast<GO>(2*rank), "Owned contiguous GID");
    equal(numbering.cell_to_global[2], static_cast<GO>(2*(1-rank)+1), "Permuted ghost index maps to its donor");
    equal(numbering.cell_to_global[3], static_cast<GO>(2*(1-rank)), "Ghost order is not sorted away");
    auto expanded = numbering.expanded_gids(3);
    equal(expanded[7], numbering.cell_to_global[2]*3+1, "Expanded ghost DOF numbering");
    auto input = la::make_face_csr(mesh, numbering);
    equal(input.row_offsets.size(), std::size_t{3}, "Owned-only CSR rows");
    equal(input.column_gids.size(), std::size_t{6}, "Diagonal and real-cell neighbours; no boundary unknown");
    la::Distribution<> distribution(numbering.owned_gids(), MPI_COMM_WORLD);
    la::CsrPattern<> pattern(distribution, input, exec);
    la::CsrMatrix<double> matrix(pattern);
    std::vector<double> values(input.column_gids.size(), 1);
    matrix.set_values(values);
    la::Vector<double> x(distribution), y(distribution);
    set_vector(x, {1+static_cast<double>(2*rank), 2+static_cast<double>(2*rank)}, exec);
    matrix.apply(x, y);
    auto output = get_vector(y);
    near(output[0], rank == 0 ? 7.0 : 9.0, 1.0e-13, "Mesh-to-algebra remote column apply");
    near(output[1], rank == 0 ? 6.0 : 8.0, 1.0e-13, "Second permuted ghost column apply");
}

void numbering_empty_case(const Exec&)
{
    const auto [rank, ranks] = world(); equal(ranks, 3, "Three ranks for empty numbering");
    krona::mesh::MeshPart mesh;
    mesh.rank = rank; mesh.nprocs = ranks;
    mesh.n_own = rank == 0 ? 2 : rank == 1 ? 1 : 0;
    mesh.n_cells = mesh.n_own;
    const auto numbering = la::build_cell_numbering(mesh, MPI_COMM_WORLD);
    equal(numbering.global_cells, GO{3}, "Global numbering includes empty ranks");
    equal(numbering.owned_begin, rank == 0 ? GO{0} : rank == 1 ? GO{2} : GO{3}, "Prefix on an empty rank");
}

void subcomm_case(const Exec& exec)
{
    const auto [rank, ranks] = world(); (void)ranks;
    MPI_Comm comm = MPI_COMM_NULL;
    krona::mpi::check(MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &comm), "MPI_Comm_split(test)");
    {
        krona::mesh::MeshPart mesh;
        mesh.rank = 0; mesh.nprocs = 1; mesh.n_own = 2; mesh.n_cells = 2;
        mesh.n_faces = 1; mesh.face_owner = {0}; mesh.face_neigh = {1};
        const auto numbering = la::build_cell_numbering(mesh, comm);
        equal(numbering.global_cells, GO{2}, "Numbering is relative to the supplied subcommunicator");
        la::Distribution<> distribution(numbering.owned_gids(), comm);
        auto input = la::make_face_csr(mesh, numbering);
        la::CsrPattern<> pattern(distribution, input, exec);
        la::CsrMatrix<double> matrix(pattern);
        const std::array<double, 4> values{2, -1, -1, 2}; matrix.set_values(values);
        la::Vector<double> rhs(distribution), x(distribution);
        set_vector(rhs, {1.0, 4.0}, exec);
        la::Solver<double> solver(matrix);
        const auto result = solver.solve(rhs, x);
        check(result.converged, "Independent subcommunicator solve");
        auto output = get_vector(x);
        near(output[0], 2.0, 1.0e-9, "Subcommunicator first unknown");
        near(output[1], 3.0, 1.0e-9, "Subcommunicator second unknown");
    }
    krona::mpi::check(MPI_Comm_free(&comm), "MPI_Comm_free(test)");
}

void csr_case(const Exec& exec) { solve_chain<Scalar>(exec); }
void device_case(const Exec& exec) { solve_chain<Scalar>(exec, false, false, true); }
void update_case(const Exec& exec) { solve_chain<Scalar>(exec, false, true); }
void empty_case(const Exec& exec) { solve_chain<Scalar>(exec, true); }
void precision_case(const Exec& exec) { solve_chain<float>(exec); solve_chain<double>(exec); }

template<class Layout>
void fields(const Exec& exec)
{
    const auto [rank, ranks] = world(); (void)ranks;
    std::array<GO, 6> gids{};
    for (std::size_t i = 0; i < gids.size(); ++i) gids[i] = 6*static_cast<GO>(rank)+static_cast<GO>(i);
    la::Distribution<> distribution(gids, MPI_COMM_WORLD);
    la::Vector<double> vector(distribution);
    krona::fields::FieldStorage<float, 3, Exec, Layout> input("input", 5, krona::fields::FieldLocation::Custom, exec);
    krona::fields::FieldStorage<float, 3, Exec, Layout> output("output", 5, krona::fields::FieldLocation::Custom, exec);
    auto host = input.create_host_buffer();
    for (std::size_t i = 0; i < 5; ++i) for (int c = 0; c < 3; ++c) host(i, c) = static_cast<float>(10*c)+static_cast<float>(i);
    Kokkos::deep_copy(exec, input.view().kokkos_view(), host);
    output.fill(exec, -999);
    la::copy_from_field(vector, input.view().as_const(), exec); vector.commit();
    la::copy_to_field(vector, output.view(), exec); exec.fence();
    auto result = output.create_host_buffer();
    Kokkos::deep_copy(exec, result, output.view().kokkos_view()); exec.fence();
    for (std::size_t i = 0; i < 5; ++i) for (int c = 0; c < 3; ++c)
        equal(result(i, c), i < 2 ? host(i, c) : -999.0F, "Field packing respects layout, mixed precision and extended tails");
    std::array<GO, 2> scalar_gids{2*static_cast<GO>(rank), 2*static_cast<GO>(rank)+1};
    la::Distribution<> scalar_distribution(scalar_gids, MPI_COMM_WORLD);
    la::Vector<double> scalar(scalar_distribution);
    scalar.copy_from_component(input.view().component(2), exec); scalar.commit();
    scalar.copy_to_component(output.view().component(1), exec); exec.fence();
    auto values = get_vector(scalar);
    near(values[0], 20.0, 0.0, "Strided component prefix");
    near(values[1], 21.0, 0.0, "Second strided component value");
}
void fields_case(const Exec& exec) { fields<Kokkos::LayoutLeft>(exec); fields<Kokkos::LayoutRight>(exec); }

void shared_case(const Exec& exec)
{
    const auto data = chain();
    la::Distribution<> distribution(data.owned, MPI_COMM_WORLD);
    la::CsrPattern<> pattern(distribution, data.input, exec);
    la::CsrMatrix<double> a(pattern), b(pattern);
    equal(a.native()->getCrsGraph().getRawPtr(), b.native()->getCrsGraph().getRawPtr(), "Matrices share the graph");
    a.set_values(coefficients<double>(data)); b.set_values(coefficients<double>(data, 3));
    la::Vector<double> x(distribution), ya(distribution), yb(distribution);
    x.fill(1); a.apply(x, ya); b.apply(x, yb);
    auto ha = get_vector(ya), hb = get_vector(yb);
    for (std::size_t i = 0; i < ha.size(); ++i) near(hb[i]-ha[i], 3.0, 1.0e-13, "Shared graph has independent coefficients");
}

template<class Values, class Graph>
struct NativeDiagonal {
    Values values;
    Graph graph;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t row) const
    {
        // This case uses diagonal-only sparsity, so each row has one entry.
        values(graph.row_map(row)) = 2.0;
    }
};
void native_case(const Exec& exec)
{
    const auto [rank, ranks] = world(); (void)ranks;
    std::array<GO, 2> gids{2*static_cast<GO>(rank), 2*static_cast<GO>(rank)+1};
    la::Distribution<> distribution(gids, MPI_COMM_WORLD);
    la::CsrInput input{{0, 1, 2}, {gids[0], gids[1]}};
    la::CsrPattern<> pattern(distribution, input, exec, false);
    check(!pattern.has_input_mapping(), "Native assembly omits coefficient permutation storage");
    la::CsrMatrix<double> matrix(pattern);
    auto values = matrix.edit_values(exec);
    auto graph = pattern.native()->getLocalGraphDevice();
    using Range = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("test/native_diagonal", Range(exec, 0, distribution.local_size()),
        NativeDiagonal<decltype(values), decltype(graph)>{values, graph});
    matrix.commit();
    la::Vector<double> x(distribution), y(distribution);
    x.fill(3); matrix.apply(x, y);
    const auto output = get_vector(y);
    for (const auto value : output) near(value, 6.0, 0.0, "Native device coefficient assembly");
}

void nonconvergence_case(const Exec& exec)
{
    const auto data = chain(20);
    la::Distribution<> distribution(data.owned, MPI_COMM_WORLD);
    la::CsrPattern<> pattern(distribution, data.input, exec);
    la::CsrMatrix<double> matrix(pattern); matrix.set_values(coefficients<double>(data));
    la::Vector<double> rhs(distribution), x(distribution);
    std::vector<double> b;
    for (const auto point : data.points) b.push_back(right_hand_side(point, data.total));
    set_vector(rhs, b, exec);
    la::SolverOptions options; options.maximum_iterations = 1; options.restart_length = 1;
    options.relative_tolerance = 1.0e-13; options.preconditioner.clear();
    la::Solver<double> solver(matrix, options);
    const auto result = solver.solve(rhs, x);
    check(!result.converged && result.iterations == 1, "Nonconvergence is reported without aborting");
    check(result.residual_norm > options.relative_tolerance*result.rhs_norm, "Nonconvergence uses the true residual");
}

void zero_rhs_case(const Exec& exec)
{
    const auto data = chain();
    la::Distribution<> distribution(data.owned, MPI_COMM_WORLD);
    la::CsrPattern<> pattern(distribution, data.input, exec);
    la::CsrMatrix<double> matrix(pattern); matrix.set_values(coefficients<double>(data));
    la::Vector<double> rhs(distribution), x(distribution);
    la::SolverOptions options; options.absolute_tolerance = 1.0e-10;
    la::Solver<double> solver(matrix, options);
    auto result = solver.solve(rhs, x);
    check(result.converged && result.iterations == 0 && result.relative_residual == 0, "Zero RHS and zero initial guess");
    x.fill(1);
    result = solver.solve(rhs, x);
    check(result.converged && result.residual_norm <= options.absolute_tolerance, "Zero RHS with a nonzero initial guess");
    rhs.fill(1.0e-15); x.fill(0);
    result = solver.solve(rhs, x);
    check(result.converged && result.iterations == 0, "Absolute tolerance accepts a small RHS");
    (void)exec;
}

void all_empty_case(const Exec& exec)
{
    la::Distribution<> distribution(std::span<const GO>{}, MPI_COMM_WORLD);
    la::CsrInput input{{0}, {}};
    la::CsrPattern<> pattern(distribution, input, exec);
    la::CsrMatrix<double> matrix(pattern);
    la::Vector<double> rhs(distribution), x(distribution);
    la::Solver<double> solver(matrix);
    const auto result = solver.solve(rhs, x);
    check(result.converged && result.iterations == 0, "Globally empty system is a collective no-op");
}

void duplicate_case(const Exec& exec)
{
    const auto [rank, ranks] = world(); (void)ranks;
    const std::array<GO, 1> owned{static_cast<GO>(rank)};
    la::Distribution<> distribution(owned, MPI_COMM_WORLD);
    la::CsrInput input{{0, 2}, {rank, rank}};
    la::CsrPattern<> pattern(distribution, input, exec);
    (void)pattern;
}
void pending_case(const Exec& exec)
{
    const auto data = chain();
    la::Distribution<> distribution(data.owned, MPI_COMM_WORLD);
    la::CsrPattern<> pattern(distribution, data.input, exec);
    la::CsrMatrix<double> matrix(pattern);
    auto values = matrix.edit_values(exec); (void)values;
    la::Vector<double> x(distribution), y(distribution);
    matrix.apply(x, y);
}
void missing_case(const Exec& exec)
{
    const auto [rank, ranks] = world(); (void)ranks;
    const std::array<GO, 1> owned{static_cast<GO>(rank)};
    la::Distribution<> distribution(owned, MPI_COMM_WORLD);
    la::CsrInput input{{0, 1}, {GO{1000000}}};
    la::CsrPattern<> pattern(distribution, input, exec); (void)pattern;
}
void offsets_case(const Exec& exec)
{
    const auto [rank, ranks] = world(); (void)ranks;
    const std::array<GO, 2> owned{2*static_cast<GO>(rank), 2*static_cast<GO>(rank)+1};
    la::Distribution<> distribution(owned, MPI_COMM_WORLD);
    la::CsrInput input{{0, 100, 1}, {owned[0]}};
    la::CsrPattern<> pattern(distribution, input, exec); (void)pattern;
}
} // namespace

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    try {
        Kokkos::ScopeGuard kokkos{argc, argv};
        Tpetra::ScopeGuard tpetra{&argc, &argv};
        const std::array cases{
            Case{"distribution", distribution_case}, Case{"numbering", numbering_case},
            Case{"numbering_empty", numbering_empty_case}, Case{"subcomm", subcomm_case}, Case{"csr", csr_case},
            Case{"device", device_case}, Case{"update", update_case}, Case{"empty", empty_case},
            Case{"precision", precision_case}, Case{"fields", fields_case}, Case{"shared", shared_case}, Case{"native", native_case},
            Case{"nonconvergence", nonconvergence_case}, Case{"zero_rhs", zero_rhs_case},
            Case{"all_empty", all_empty_case}, Case{"duplicate", duplicate_case}, Case{"pending", pending_case},
            Case{"missing", missing_case}, Case{"offsets", offsets_case}};
        check(argc == 2, "Expected one linalg test-case name");
        const std::string_view selected = argv[1];
        Exec exec{};
        for (const auto& entry : cases) if (entry.name == selected) {
            std::cout << "RUN linalg." << entry.name << " [" << Exec::name() << "]\n";
            entry.function(exec); exec.fence();
            std::cout << "PASS linalg." << entry.name << '\n';
            return EXIT_SUCCESS;
        }
        check(false, "Unknown linalg test case");
    } catch (const std::exception& error) { krona::mpi::fatal(error.what()); }
    catch (...) { krona::mpi::fatal("Unhandled linalg test exception"); }
    return EXIT_FAILURE;
}
