#include "common/test_support.hpp"

#include <krona/linalg/solver.hpp>
#include <krona/linalg/field_transfer.hpp>
#include <krona/linalg/mesh_adapter.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using namespace krona;
using namespace krona::linalg;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
int rank = 0, ranks = 1;

using krona::test::check;

template<class T>
auto device_copy(const std::vector<T>& input)
{
    Kokkos::View<T*, Space> result("test/input", input.size());
    auto host = Kokkos::create_mirror_view(result);
    for (std::size_t i = 0; i < input.size(); ++i) host(i) = input[i];
    Kokkos::deep_copy(result, host);
    return result;
}

template<class S>
void exact_values(Vector<S, Exec>& vector)
{
    std::vector<GlobalOrdinal> host_gids(vector.local_size());
    for (std::size_t i = 0; i < host_gids.size(); ++i)
        host_gids[i] = vector.distribution().global_id(static_cast<LocalOrdinal>(i));
    const auto gids = device_copy(host_gids);
    const auto values = vector.overwrite();
    Kokkos::parallel_for("test/exact", Kokkos::RangePolicy<Exec>(0, host_gids.size()),
        KOKKOS_LAMBDA(int i) { values(i, 0) = S(1) + S(0.125)*S(gids(i)%7); });
    vector.commit();
}

template<class S>
double error(const Vector<S, Exec>& vector)
{
    double local = 0;
    {
        auto values = std::as_const(*vector.native()).getLocalViewHost(Tpetra::Access::ReadOnly);
        for (std::size_t i = 0; i < vector.local_size(); ++i) {
            const auto gid = vector.distribution().global_id(static_cast<LocalOrdinal>(i));
            local = std::max(local, std::abs(double(values(i, 0)) - (1 + 0.125*double(gid%7))));
        }
    }
    double global = 0;
    mpi::check(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "test error reduction");
    return global;
}

std::vector<GlobalOrdinal> owned(int n = 16)
{
    std::vector<GlobalOrdinal> gids;
    // Cyclic, reversed local numbering exercises both map and CSR permutations.
    for (int i = n-1; i >= 0; --i) gids.push_back(rank + static_cast<GlobalOrdinal>(ranks)*i);
    return gids;
}

CsrInput input_for(const Distribution<Exec>& distribution)
{
    CsrInput input;
    input.row_offsets.push_back(0);
    for (std::size_t row = 0; row < distribution.local_size(); ++row) {
        const auto gid = distribution.global_id(static_cast<LocalOrdinal>(row));
        input.column_gids.push_back(gid);
        if (gid+1 < static_cast<GlobalOrdinal>(distribution.global_size())) input.column_gids.push_back(gid+1);
        if (gid > 0) input.column_gids.push_back(gid-1);
        input.row_offsets.push_back(input.column_gids.size());
    }
    return input;
}

template<class S>
std::vector<S> coefficients(const Distribution<Exec>& distribution, const CsrInput& input, bool symmetric)
{
    std::vector<S> result(input.column_gids.size());
    for (std::size_t row = 0; row < distribution.local_size(); ++row) {
        const auto gid = distribution.global_id(static_cast<LocalOrdinal>(row));
        for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k)
            result[k] = input.column_gids[k] == gid ? S(4) :
                (symmetric ? S(-1) : (input.column_gids[k] < gid ? S(-0.5) : S(-1.5)));
    }
    return result;
}

void mapping_and_assembly()
{
    const auto gids = owned();
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    const auto input = input_for(distribution);
    CsrPattern<Exec> pattern(distribution, input);
    check(pattern.local_nonzeros() == input.column_gids.size(), "test: wrong graph entry count");
    {
        const auto graph = pattern.native()->getLocalGraphHost();
        const auto permutation = pattern.host_input_to_entry();
        for (std::size_t row = 0; row < gids.size(); ++row)
            for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k) {
                const auto pos = permutation[k];
                check(pos >= graph.row_map(row) && pos < graph.row_map(row+1), "test: permutation leaves its row");
                check(pattern.native()->getColMap()->getGlobalElement(graph.entries(pos)) == input.column_gids[k],
                      "test: wrong input-to-native permutation");
            }
    }
    CsrMatrix<double, Exec> matrix(pattern);
    check(matrix.native()->getCrsGraph().getRawPtr() == pattern.native().getRawPtr(), "test: graph was copied");
    const auto values = coefficients<double>(distribution, input, false);
    matrix.set_values(values);
    Vector<double, Exec> x(distribution), host_result(distribution), device_result(distribution);
    exact_values(x);
    matrix.apply(x, host_result);
    const auto revision = matrix.revision();
    const auto device_values = device_copy(values);
    const auto exec = Kokkos::Experimental::partition_space(Exec{}, 1)[0];
    matrix.set_values_async(device_values, exec);
    matrix.commit();
    check(matrix.revision() == revision+1, "test: asynchronous coefficient edit did not advance revision");
    matrix.apply(x, device_result);
    device_result.native()->update(-1.0, *host_result.native(), 1.0);
    check(device_result.native()->norm2() < 1e-13, "test: host/device assembly disagree");
    {
        auto first = matrix.read_values();
        auto second = matrix.read_values();
        check(first.data() == second.data(), "test: read views do not share coefficients");
        matrix.commit();
    }
    check(matrix.revision() == revision+1, "test: coefficient read changed revision");
    CsrPattern<Exec> native_pattern(distribution, input, Exec{}, false);
    CsrMatrix<double, Exec> native_matrix(native_pattern);
    {
        auto target = native_matrix.edit_values(exec);
        auto source = matrix.read_values(exec);
        Kokkos::deep_copy(exec, target, source);
        native_matrix.commit(); matrix.commit();
    }
    native_matrix.apply(x, device_result);
    device_result.native()->update(-1.0, *host_result.native(), 1.0);
    check(device_result.native()->norm2() < 1e-13, "test: native-only assembly disagrees");
}

template<class S, class P>
void solve_case(const std::string& method, const std::string& preconditioner, bool symmetric, bool reuse = false)
{
    auto gids = owned();
    // Give Schwarz nontrivial local factors and an interface to adjacent ranks.
    if (preconditioner == "SCHWARZ")
        for (std::size_t i = 0; i < gids.size(); ++i)
            gids[i] = static_cast<GlobalOrdinal>(rank)*static_cast<GlobalOrdinal>(gids.size()) + static_cast<GlobalOrdinal>(gids.size()-1-i);
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    const auto input = input_for(distribution);
    CsrPattern<Exec> pattern(distribution, input);
    CsrMatrix<S, Exec> matrix(pattern);
    auto values = coefficients<S>(distribution, input, symmetric);
    matrix.set_values(values);
    Vector<S, Exec> exact(distribution), rhs(distribution), solution(distribution);
    exact_values(exact);
    matrix.apply(exact, rhs);
    solution.fill(S(0.25));
    SolverOptions options;
    options.method = method;
    options.preconditioner.type = preconditioner;
    options.relative_tolerance = std::is_same_v<S, float> ? 3e-5 : 1e-10;
    if constexpr (std::is_same_v<S, float>) options.relative_tolerance.reset();
    options.restart_length = 5;
    options.maximum_iterations = 300;
    if (preconditioner == "SCHWARZ") {
        options.preconditioner.parameters.set("schwarz: overlap level", 0);
        options.preconditioner.parameters.set("inner preconditioner name", "RILUK");
        options.preconditioner.parameters.sublist("inner preconditioner parameters").set("fact: iluk level-of-fill", 1);
    }
    if (preconditioner == "AMG") {
        options.preconditioner.parameters.set("coarse: max size", 4);
        options.preconditioner.parameters.set("coarse: type", "RELAXATION");
        options.preconditioner.parameters.sublist("coarse: params").set("relaxation: type", "Jacobi");
        options.preconditioner.parameters.sublist("coarse: params").set("relaxation: sweeps", 5);
        if (reuse) options.preconditioner.parameters.set("reuse: type", "full");
    }
    Solver<S, Exec, P> solver(matrix, options);
    if (rank == 0)
        std::fprintf(stderr, "[linalg] solve %s/%s/%zu/%zu, reuse=%s\n", method.c_str(),
                     preconditioner.c_str(), sizeof(S), sizeof(P), reuse ? "full" : "none");
    const auto result = solver.solve(rhs, solution);
    if constexpr (std::is_same_v<S, float>)
        check(result.residual_threshold == 1e-5*result.rhs_norm, "test: default tolerance uses policy instead of actual solver scalar");
    if (!result.converged)
        std::fprintf(stderr, "Rank %d failed %s/%s/%zu/%zu: it=%d r=%g target=%g krylov=%d\n", rank, method.c_str(),
                     preconditioner.c_str(), sizeof(S), sizeof(P), result.iterations, result.residual_norm,
                     result.residual_threshold, result.krylov_converged);
    check(result.converged && result.residual_norm <= result.residual_threshold, "test: true residual did not converge");
    check(error(solution) < (std::is_same_v<S, float> ? 3e-4 : 1e-8), "test: wrong solution");
    const auto generation = solver.preconditioner().generation();
    const auto old_inner = solver.preconditioner().native_inner().getRawPtr();
    // An exact initial guess must not start Belos or rebuild its preconditioner.
    const auto already_done = solver.solve(rhs, exact);
    check(already_done.converged && !already_done.iterations_started && already_done.iterations == 0,
          "test: exact initial guess was not accepted");
    check(solver.preconditioner().generation() == generation, "test: unnecessary preconditioner rebuild");
    // Only one process changes coefficients: preparation must still refresh on every process.
    if (rank == 0) {
        for (auto& value : values) value *= S(1.25);
        matrix.set_values(values);
    }
    matrix.apply(exact, rhs);
    solution.fill(S(0));
    const auto changed = solver.solve(rhs, solution);
    check(changed.converged && error(solution) < (std::is_same_v<S, float> ? 3e-4 : 1e-8),
          "test: stale preconditioner or coefficients after a rank-local edit");
    if (preconditioner != "NONE")
        check(solver.preconditioner().generation() == generation+1, "test: preconditioner revision was not refreshed collectively");
    if (reuse) check(solver.preconditioner().native_inner().getRawPtr() == old_inner, "test: AMG reuse replaced its operator");
    if (rank == 0) std::printf("PASS solve %s/%s/%zu/%zu\n", method.c_str(), preconditioner.c_str(), sizeof(S), sizeof(P));
}

void symmetric_indefinite()
{
    std::vector<GlobalOrdinal> gids;
    for (int i = 0; i < 4; ++i) gids.push_back(4*rank+i);
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    CsrInput input{{0, 1, 2, 3, 4}, gids};
    CsrPattern<Exec> pattern(distribution, input);
    CsrMatrix<double, Exec> matrix(pattern);
    std::vector<double> values;
    for (const auto gid : gids) values.push_back((gid%2 == 0 ? 1.0 : -1.0)*static_cast<double>(gid+2));
    matrix.set_values(values);
    Vector<double, Exec> exact(distribution), rhs(distribution), solution(distribution);
    exact_values(exact);
    matrix.apply(exact, rhs);
    SolverOptions options;
    options.method = "MINRES";
    options.preconditioner.type = "NONE";
    Solver<double, Exec, double> solver(matrix, options);
    const auto result = solver.solve(rhs, solution);
    check(result.converged && error(solution) < 1e-8, "test: MINRES failed on a symmetric indefinite operator");
}

void stopping_cases()
{
    const auto gids = owned();
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    const auto input = input_for(distribution);
    CsrPattern<Exec> pattern(distribution, input);
    CsrMatrix<double, Exec> matrix(pattern);
    matrix.set_values(coefficients<double>(distribution, input, true));
    Vector<double, Exec> rhs(distribution), solution(distribution), exact(distribution);
    exact_values(exact);
    SolverOptions options;
    options.preconditioner.type = "NONE";
    options.relative_tolerance = 1e-9;
    Solver<double, Exec, double> solver(matrix, options);
    // Homogeneous system with a nonzero initial guess exercises the zero-RHS scale.
    solution.fill(2.0);
    const auto zero_rhs = solver.solve(rhs, solution);
    check(zero_rhs.converged && zero_rhs.initial_residual_norm > 0 && zero_rhs.rhs_norm == 0,
          "test: homogeneous solve or zero-RHS reference failed");
    check(zero_rhs.relative_residual <= 1e-9, "test: zero-RHS relative residual is inconsistent");
    matrix.apply(exact, rhs);
    options.maximum_iterations = 0;
    Solver<double, Exec, double> no_iterations(matrix, options);
    solution.fill(0);
    const auto stopped = no_iterations.solve(rhs, solution);
    check(!stopped.converged && !stopped.iterations_started && stopped.iterations == 0,
          "test: zero iteration budget falsely converged");
    options.maximum_iterations = 1;
    Solver<double, Exec, double> one_iteration(matrix, options);
    const auto limited = one_iteration.solve(rhs, solution);
    check(!limited.converged && limited.iterations_started && limited.iterations <= 1,
          "test: iteration limit or independent residual check failed");
    options.maximum_iterations = 300;
    options.relative_tolerance = 0;
    options.absolute_tolerance = 1e-8;
    Solver<double, Exec, double> absolute(matrix, options);
    solution.fill(0);
    const auto result = absolute.solve(rhs, solution);
    check(result.converged && result.residual_norm <= 1e-8, "test: absolute-only tolerance failed");
}

void empty_cases()
{
    for (const bool all_empty : {false, true}) {
        std::vector<GlobalOrdinal> gids;
        if (!all_empty && rank == 0) gids = {0};
        Distribution<Exec> distribution(gids, mpi::CommView::world());
        CsrInput input;
        input.row_offsets = gids.empty() ? std::vector<std::size_t>{0} : std::vector<std::size_t>{0, 1};
        input.column_gids = gids;
        CsrPattern<Exec> pattern(distribution, input);
        CsrMatrix<double, Exec> matrix(pattern);
        matrix.set_values(std::vector<double>(gids.size(), 2.0));
        Vector<double, Exec> rhs(distribution), solution(distribution);
        rhs.fill(2.0);
        SolverOptions options;
        options.method = "CG";
        options.preconditioner.type = "NONE";
        Solver<double, Exec, double> solver(matrix, options);
        const auto result = solver.solve(rhs, solution);
        check(result.converged, "test: solve with empty local/global distribution failed");
        if (all_empty) check(result.iterations == 0, "test: empty system performed iterations");
    }
}

template<class Layout>
void field_transfers()
{
    const auto exec = Kokkos::Experimental::partition_space(Exec{}, 1)[0];
    std::vector<GlobalOrdinal> gids(6);
    for (std::size_t i = 0; i < gids.size(); ++i) gids[i] = static_cast<GlobalOrdinal>(rank)*6+static_cast<GlobalOrdinal>(i);
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    Vector<double, Exec> vector(distribution);
    Kokkos::View<float*[3], Layout, Space> source("source", 5), target("target", 5), strided("strided", 9);
    Kokkos::deep_copy(target, -7.0f); Kokkos::deep_copy(strided, -9.0f);
    Kokkos::parallel_for("test/fields", Kokkos::RangePolicy<Exec>(exec, 0, 5),
        KOKKOS_LAMBDA(int cell) { for (int c = 0; c < 3; ++c) source(cell, c) = float(10*cell+c); });
    using Field = fields::FieldView<float, 3, Space, Layout>;
    Field from(typename Field::view_type(source.data(), 5));
    Field to(typename Field::view_type(target.data(), 5));
    copy_from_field(vector, from, exec);
    vector.commit();
    copy_to_field(vector, to, exec);
    vector.copy_to_component(Kokkos::subview(strided, Kokkos::ALL(), 1), exec);
    vector.fill_async(99.0, exec); // Same-instance overwrite follows the queued reads.
    vector.commit();
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, target);
    const auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, strided);
    for (int cell = 0; cell < 5; ++cell) for (int c = 0; c < 3; ++c)
        check(h(cell, c) == (cell < 2 ? float(10*cell+c) : -7.0f), "test: field transfer or extended tail corruption");
    for (int i = 0; i < 9; ++i) for (int c = 0; c < 3; ++c)
        check(s(i, c) == (c == 1 && i < 6 ? float(10*(i/3)+i%3) : -9.0f), "test: strided component or tail corruption");
    vector.copy_from_component(Kokkos::subview(strided, Kokkos::ALL(), 1), exec);
    vector.commit();
    const auto local = std::as_const(*vector.native()).getLocalViewHost(Tpetra::Access::ReadOnly);
    for (int i = 0; i < 6; ++i) check(local(i, 0) == double(10*(i/3)+i%3), "test: strided component import failed");
}

void mesh_adapter()
{
    mesh::MeshPart mesh;
    mesh.rank = rank; mesh.nprocs = ranks; mesh.n_own = 2; mesh.n_cells = 2*ranks;
    mesh.face_owner = {0, 0, 0, 1}; mesh.face_neigh = {1, 1, -1, -1};
    for (int peer = 0, slot = 2; peer < ranks; ++peer) if (peer != rank) {
        mesh.nb_ranks.push_back(peer);
        mesh.send_owned_local.push_back(1); mesh.send_owned_local.push_back(0);
        mesh.recv_ghost_local.push_back(slot+1); mesh.recv_ghost_local.push_back(slot);
        mesh.send_offsets.push_back(mesh.send_owned_local.size());
        mesh.recv_offsets.push_back(mesh.recv_ghost_local.size());
        mesh.face_owner.push_back(0); mesh.face_neigh.push_back(slot);
        mesh.face_owner.push_back(1); mesh.face_neigh.push_back(slot+1);
        slot += 2;
    }
    mesh.n_faces = static_cast<LocalIndex>(mesh.face_owner.size());
    const auto numbering = build_cell_numbering(mesh, mpi::CommView::world());
    check(numbering.n_owned == 2 && numbering.owned_begin == rank*2 && numbering.global_cells == ranks*2,
          "test: wrong prefix numbering");
    for (int peer = 0, slot = 2; peer < ranks; ++peer) if (peer != rank) {
        check(numbering.cell_to_global[static_cast<std::size_t>(slot)] ==
            peer*2 && numbering.cell_to_global[static_cast<std::size_t>(slot)+1] == peer*2+1,
              "test: halo numbering did not respect receive permutation");
        slot += 2;
    }
    const auto expanded = numbering.expanded_gids(3);
    for (std::size_t cell = 0; cell < numbering.cell_to_global.size(); ++cell) for (int c = 0; c < 3; ++c)
        check(expanded[cell*3+c] == numbering.cell_to_global[cell]*3+c, "test: wrong multicomponent numbering");
    const auto input = make_face_csr(mesh, numbering);
    for (std::size_t row = 0; row < 2; ++row) {
        std::vector<GlobalOrdinal> expected{rank*2, rank*2+1};
        for (int peer = 0; peer < ranks; ++peer) if (peer != rank) expected.push_back(peer*2+static_cast<int>(row));
        std::sort(expected.begin(), expected.end());
        const std::vector<GlobalOrdinal> actual(input.column_gids.begin()+input.row_offsets[row],
                                               input.column_gids.begin()+input.row_offsets[row+1]);
        check(actual == expected, "test: face CSR omitted entries, kept duplicates or added boundary slots");
    }
}

// A simple independent native operator checks the precision bridge's full apply contract.
template<class S>
class ScaleOperator final : public Types<S, Exec>::operator_type {
    using T = Types<S, Exec>;
    Teuchos::RCP<const typename T::map_type> map_;
public:
    mutable int calls = 0;
    explicit ScaleOperator(Teuchos::RCP<const typename T::map_type> map) : map_(std::move(map)) {}
    Teuchos::RCP<const typename T::map_type> getDomainMap() const override { return map_; }
    Teuchos::RCP<const typename T::map_type> getRangeMap() const override { return map_; }
    bool hasTransposeApply() const override { return true; }
    void apply(const typename T::multivector_type& x, typename T::multivector_type& y,
               Teuchos::ETransp = Teuchos::NO_TRANS, S alpha = S(1), S beta = S(0)) const override
    { ++calls; y.update(S(3)*alpha, x, beta); }
};

void precision_bridge_and_factory()
{
    const auto gids = owned(3);
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    auto inner = Teuchos::rcp(new ScaleOperator<float>(distribution.native()));
    detail::PrecisionOperator<double, float, Exec> bridge(inner);
    using MV = Types<double, Exec>::multivector_type;
    MV x(distribution.native(), 4), y(distribution.native(), 4);
    x.putScalar(2); y.putScalar(5);
    const double alpha = 1.0000000001, beta = 0.2500000001;
    bridge.apply(x, y, Teuchos::NO_TRANS, alpha, beta);
    {
        auto view = std::as_const(y).getLocalViewHost(Tpetra::Access::ReadOnly);
        for (std::size_t row = 0; row < gids.size(); ++row) for (int c = 0; c < 4; ++c)
            check(std::abs(view(row, c) - (6*alpha+5*beta)) < 1e-13, "test: precision bridge lost alpha/beta precision or old y");
    }
    // Exercise nonconstant-stride columns, a different width, and transpose workspaces.
    Teuchos::Array<std::size_t> selected_x{3, 1}, selected_y{2, 0};
    const auto xs = std::as_const(x).subView(selected_x());
    auto ys = y.subViewNonConst(selected_y());
    bridge.apply(*xs, *ys, Teuchos::TRANS, 1.0, 0.0);
    {
        auto view = std::as_const(y).getLocalViewHost(Tpetra::Access::ReadOnly);
        for (std::size_t row = 0; row < gids.size(); ++row)
            check(view(row, 0) == 6 && view(row, 2) == 6, "test: nonconstant-stride precision conversion failed");
    }
    x.putScalar(std::numeric_limits<double>::quiet_NaN());
    y.putScalar(std::numeric_limits<double>::quiet_NaN());
    const auto calls = inner->calls;
    bridge.apply(x, y, Teuchos::NO_TRANS, 0.0, 0.0);
    check(inner->calls == calls, "test: alpha=0 unnecessarily applied inner operator");
    Teuchos::Array<double> norms(4);
    y.normInf(norms());
    check(norms[0] == 0, "test: alpha=beta=0 read NaN input/output");

    const auto input = input_for(distribution);
    CsrPattern<Exec> pattern(distribution, input);
    CsrMatrix<double, Exec> matrix(pattern);
    matrix.set_values(coefficients<double>(distribution, input, true));
    PreconditionerOptions options;
    options.type = "NONE"; // A supplied factory overrides this selector.
    int preparations = 0;
    using Prec = Preconditioner<double, Exec, float>;
    Prec prec(matrix, options, [&](const Teuchos::RCP<const Prec::matrix_type>& low,
                                  const Teuchos::RCP<Prec::inner_operator_type>& previous) {
        ++preparations;
        check(low->getCrsGraph().getRawPtr() == pattern.native().getRawPtr(), "test: low-precision matrix copied graph");
        if (!previous.is_null()) return previous;
        return Teuchos::rcp_implicit_cast<Prec::inner_operator_type>(Teuchos::rcp(new ScaleOperator<float>(distribution.native())));
    });
    prec.prepare(); prec.prepare();
    check(preparations == 1 && prec.generation() == 1, "test: custom factory rebuilt unchanged preconditioner");
    if (rank == 0) matrix.mark_modified();
    prec.prepare();
    check(preparations == 2 && prec.generation() == 2, "test: custom factory did not refresh collectively");
    using SamePrec = Preconditioner<double, Exec, double>;
    SamePrec same(matrix, options, [&](const Teuchos::RCP<const SamePrec::matrix_type>& shared,
                                      const Teuchos::RCP<SamePrec::inner_operator_type>&) {
        check(shared.getRawPtr() == matrix.native().getRawPtr(), "test: same precision allocated another matrix");
        return Teuchos::rcp_implicit_cast<SamePrec::inner_operator_type>(Teuchos::rcp(new ScaleOperator<double>(distribution.native())));
    });
    same.prepare();
}

void rejected_case(const std::string& mode)
{
    std::vector<GlobalOrdinal> gids{rank};
    Distribution<Exec> distribution(gids, mpi::CommView::world());
    CsrInput input{{0, 1}, gids};
    if (mode == "duplicate_column") { input.row_offsets = {0, 2}; input.column_gids = {rank, rank}; }
    if (mode == "missing_owner") input.column_gids = {static_cast<GlobalOrdinal>(ranks+10)};
    CsrPattern<Exec> pattern(distribution, input);
    CsrMatrix<double, Exec> matrix(pattern);
    matrix.set_values(std::vector<double>{2.0});
    Vector<double, Exec> rhs(distribution), solution(distribution);
    if (mode == "pending_read") { const auto view = rhs.read(); (void)view; matrix.apply(rhs, solution); }
    if (mode == "pending_write") { const auto view = solution.edit(); (void)view; matrix.apply(rhs, solution); }
    if (mode == "pending_matrix_read") { const auto view = matrix.read_values(); (void)view; matrix.apply(rhs, solution); }
    SolverOptions options; options.preconditioner.type = "RELAXATION";
    if (mode == "bicgstab_left") { options.method = "BICGSTAB"; options.side = PreconditioningSide::Left; }
    if (mode == "bad_method") options.method = "NO_SUCH_METHOD";
    if (mode == "precision_overflow") matrix.set_values(std::vector<double>{std::numeric_limits<double>::max()});
    Solver<double, Exec, float> solver(matrix, options);
    solver.prepare();
    check(false, "test: rejection case unexpectedly succeeded");
}

void initialization()
{
    check(Tpetra::isInitialized(), "Tpetra must be initialized inside a linalg case");
    check(Kokkos::is_initialized(), "Kokkos must remain initialized");
    int provided = MPI_THREAD_SINGLE;
    mpi::check(MPI_Query_thread(&provided), "MPI_Query_thread");
    check(provided >= MPI_THREAD_FUNNELED, "Linalg tests require MPI_THREAD_FUNNELED");
    check(Tpetra::getDefaultComm()->getSize() == ranks, "Tpetra/MPI communicator size mismatch");
}

void gmres() { solve_case<double, double>("GMRES", "NONE", false); }
void fgmres() { solve_case<double, double>("FGMRES", "NONE", false); }
void bicgstab() { solve_case<double, double>("BICGSTAB", "NONE", false); }
void cg() { solve_case<double, double>("CG", "NONE", true); }
void minres() { solve_case<double, double>("MINRES", "NONE", true); }

void ifpack()
{
    if (ranks == 1) solve_case<double, double>("GMRES", "RILUK", true);
    for (const auto* type : {"RELAXATION", "SCHWARZ", "CHEBYSHEV"})
        solve_case<double, double>("GMRES", type, true);
    solve_case<double, double>("CG", "RELAXATION", true);
    solve_case<double, double>("BICGSTAB", "RELAXATION", false);
}

void amg() { solve_case<double, double>("GMRES", "AMG", true); }
void amg_reuse() { solve_case<double, double>("GMRES", "AMG", true, true); }

void mixed_precision()
{
    solve_case<double, float>("FGMRES", "RELAXATION", false);
    solve_case<double, float>("FGMRES", "SCHWARZ", false);
    solve_case<double, float>("CG", "RELAXATION", true);
    solve_case<double, float>("GMRES", "AMG", true);
}

void float_solver()
{
    solve_case<float, float>("GMRES", "RELAXATION", false);
    solve_case<float, double>("GMRES", "RELAXATION", false);
}

// test::run owns MPI (FUNNELED) and Kokkos, just as in the halo tests.
// Tpetra borrows both runtimes; its guard is destroyed before returning to test::run.
template<auto Function>
void with_tpetra(const Exec& exec)
{
    const auto comm = mpi::CommView::world();
    rank = comm.rank();
    ranks = comm.size();
    mpi::guarded([&] {
        if (rank == 0) std::cerr << "[linalg] Before Tpetra initialization\n" << std::flush;
        {
            // MPI/Kokkos are already initialized, so Tpetra does not parse arguments.
            Tpetra::ScopeGuard tpetra(nullptr, nullptr, comm.native());
            if (rank == 0) std::cerr << "[linalg] After Tpetra initialization\n" << std::flush;
            krona::test::complete(exec, [] { Function(); });
            Kokkos::fence("KRONA linalg test: case complete");
        }
        // Finalizing Tpetra must not finalize the runtimes owned by test::run.
        int finalized = 0;
        mpi::check(MPI_Finalized(&finalized), "MPI_Finalized");
        check(finalized == 0 && Kokkos::is_initialized(), "Tpetra finalized an externally owned runtime");
    });
}

template<std::size_t Index>
void rejection()
{
    constexpr std::array names{"duplicate_column", "missing_owner", "pending_read", "pending_write",
        "pending_matrix_read", "bicgstab_left", "bad_method", "precision_overflow"};
    rejected_case(names[Index]);
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << std::unitbuf;
    constexpr std::array positive_cases{
        krona::test::Case{"initialization", with_tpetra<initialization>},
        krona::test::Case{"mapping", with_tpetra<mapping_and_assembly>},
        krona::test::Case{"stopping", with_tpetra<stopping_cases>},
        krona::test::Case{"empty", with_tpetra<empty_cases>},
        krona::test::Case{"minres_indefinite", with_tpetra<symmetric_indefinite>},
        krona::test::Case{"fields_left", with_tpetra<field_transfers<Kokkos::LayoutLeft>>},
        krona::test::Case{"fields_right", with_tpetra<field_transfers<Kokkos::LayoutRight>>},
        krona::test::Case{"mesh_adapter", with_tpetra<mesh_adapter>},
        krona::test::Case{"precision_bridge", with_tpetra<precision_bridge_and_factory>},
        krona::test::Case{"gmres", with_tpetra<gmres>},
        krona::test::Case{"fgmres", with_tpetra<fgmres>},
        krona::test::Case{"bicgstab", with_tpetra<bicgstab>},
        krona::test::Case{"cg", with_tpetra<cg>},
        krona::test::Case{"minres", with_tpetra<minres>},
        krona::test::Case{"ifpack", with_tpetra<ifpack>},
        krona::test::Case{"amg", with_tpetra<amg>},
        krona::test::Case{"amg_reuse", with_tpetra<amg_reuse>},
        krona::test::Case{"mixed_precision", with_tpetra<mixed_precision>},
        krona::test::Case{"float_solver", with_tpetra<float_solver>}};
    constexpr std::array negative_cases{
        krona::test::Case{"duplicate_column", with_tpetra<rejection<0>>},
        krona::test::Case{"missing_owner", with_tpetra<rejection<1>>},
        krona::test::Case{"pending_read", with_tpetra<rejection<2>>},
        krona::test::Case{"pending_write", with_tpetra<rejection<3>>},
        krona::test::Case{"pending_matrix_read", with_tpetra<rejection<4>>},
        krona::test::Case{"bicgstab_left", with_tpetra<rejection<5>>},
        krona::test::Case{"bad_method", with_tpetra<rejection<6>>},
        krona::test::Case{"precision_overflow", with_tpetra<rejection<7>>}};
    if (argc == 2) {
        const std::string_view selected = argv[1];
        for (const auto& entry : negative_cases)
            if (selected == entry.name) return krona::test::run(argc, argv, "linalg", negative_cases);
        if (selected == "all") argc = 1;
    }
    return krona::test::run(argc, argv, "linalg", positive_cases);
}
