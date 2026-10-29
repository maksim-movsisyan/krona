#include "common/test_support.hpp"

#include <krona/linalg/solver.hpp>
#include <krona/linalg/field_transfer.hpp>
#include <krona/linalg/mesh_adapter.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
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
    Solver<decltype(matrix), P> solver(matrix, options);
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
        for (std::size_t row = 0; row < gids.size(); ++row)
            for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k)
                if (input.column_gids[k] == gids[row]) values[k] *= S(1.25);
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
    Solver<decltype(matrix), double> solver(matrix, options);
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
    Solver<decltype(matrix), double> solver(matrix, options);
    // Homogeneous system with a nonzero initial guess exercises the zero-RHS scale.
    solution.fill(2.0);
    const auto zero_rhs = solver.solve(rhs, solution);
    check(zero_rhs.converged && zero_rhs.initial_residual_norm > 0 && zero_rhs.rhs_norm == 0,
          "test: homogeneous solve or zero-RHS reference failed");
    check(zero_rhs.relative_residual <= 1e-9, "test: zero-RHS relative residual is inconsistent");
    matrix.apply(exact, rhs);
    options.maximum_iterations = 0;
    Solver<decltype(matrix), double> no_iterations(matrix, options);
    solution.fill(0);
    const auto stopped = no_iterations.solve(rhs, solution);
    check(!stopped.converged && !stopped.iterations_started && stopped.iterations == 0,
          "test: zero iteration budget falsely converged");
    options.maximum_iterations = 1;
    Solver<decltype(matrix), double> one_iteration(matrix, options);
    const auto limited = one_iteration.solve(rhs, solution);
    check(!limited.converged && limited.iterations_started && limited.iterations <= 1,
          "test: iteration limit or independent residual check failed");
    options.maximum_iterations = 300;
    options.relative_tolerance = 0;
    options.absolute_tolerance = 1e-8;
    Solver<decltype(matrix), double> absolute(matrix, options);
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
        Solver<decltype(matrix), double> solver(matrix, options);
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
    mesh::host::MeshPart mesh;
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
    using Prec = Preconditioner<decltype(matrix), float>;
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
    using SamePrec = Preconditioner<decltype(matrix), double>;
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
    if (mode.starts_with("bsr_") || mode.starts_with("pack_") || mode == "expanded_gid_overflow") {
        if (mode == "expanded_gid_overflow") {
            const std::vector<GlobalOrdinal> huge{std::numeric_limits<GlobalOrdinal>::max()-2};
            Distribution<Exec> large(huge, mpi::CommView::world());
            (void)large.expanded(5);
        }
        if (mode == "bsr_zero_block") { BsrMatrix<double, Exec> invalid(pattern, 0); }
        if (mode == "bsr_different_blocks") { BsrMatrix<double, Exec> invalid(pattern, rank+2); }
        BsrMatrix<double, Exec> block_matrix(pattern, 3);
        block_matrix.set_values(std::vector<double>{2, 0, 0, 0, 2, 0, 0, 0, 2});
        Vector<double, Exec> block_rhs(block_matrix.distribution()), block_solution(block_matrix.distribution());
        if (mode == "bsr_wrong_map") {
            Vector<double, Exec> wrong(distribution); block_matrix.apply(wrong, block_solution);
        }
        if (mode == "bsr_pending_read") { const auto view = block_matrix.read_blocks(); (void)view; block_matrix.apply(block_rhs, block_solution); }
        if (mode == "bsr_wrong_values") block_matrix.set_values(std::vector<double>{1});
        if (mode == "bsr_precision_overflow") {
            block_matrix.set_values(std::vector<double>(9, std::numeric_limits<double>::max()));
            PreconditionerOptions options; options.type = "RELAXATION";
            Preconditioner<decltype(block_matrix), float> prec(block_matrix, options); prec.prepare();
        }
        if (mode == "bsr_overlap" || mode == "bsr_kokkos_amg") {
            PreconditionerOptions options;
            options.type = mode == "bsr_overlap" ? "SCHWARZ" : "AMG";
            if (mode == "bsr_overlap") options.parameters.set("schwarz: overlap level", 1);
            else options.parameters.set("use kokkos refactor", true);
            Preconditioner prec(block_matrix, options); prec.prepare();
        }
        if (mode.starts_with("pack_")) {
            Kokkos::View<double*, Space> values("short", mode == "pack_short" ? 0 : 1);
            if (mode == "pack_short") copy_from_fields(block_rhs, Exec{}, values, values, values);
            else copy_from_fields(block_rhs, Exec{}, values, values);
        }
        check(false, "test: new rejection case unexpectedly succeeded");
    }
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
    Solver<decltype(matrix), float> solver(matrix, options);
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

template<class S>
std::vector<S> block_coefficients(const Distribution<Exec>& blocks, const CsrInput& input,
                                  int width, bool symmetric)
{
    std::vector<S> values(input.column_gids.size()*static_cast<std::size_t>(width*width));
    for (std::size_t row = 0; row < blocks.local_size(); ++row) {
        const auto gid = blocks.global_id(static_cast<LocalOrdinal>(row));
        for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k)
            for (int a = 0; a < width; ++a) for (int b = 0; b < width; ++b) {
                const double coupling = symmetric ? 0.025*double(a+b+1) : 0.02*double((a+1)*(b+2));
                const double value = input.column_gids[k] == gid ?
                    (a == b ? 8.0+0.25*double(a) : coupling) :
                    -0.04*double(a+b+1)*(symmetric ? 1.0 : (input.column_gids[k] < gid ? 0.7 : 1.3));
                values[(k*static_cast<std::size_t>(width)+static_cast<std::size_t>(a))*static_cast<std::size_t>(width)+
                       static_cast<std::size_t>(b)] = static_cast<S>(value);
            }
    }
    return values;
}

void bsr_mapping()
{
    constexpr int width = 3;
    const auto gids = owned(4);
    Distribution<Exec> blocks(gids, mpi::CommView::world());
    const auto input = input_for(blocks);
    CsrPattern<Exec> graph(blocks, input);
    BsrPattern<Exec> pattern(graph, width);
    BsrMatrix<double, Exec> matrix(pattern), copied(pattern);
    check(copied.distribution().native().getRawPtr() == matrix.distribution().native().getRawPtr(),
          "test: matrices with one BSR pattern do not share their point map");
    check(matrix.distribution().local_size() == blocks.local_size()*width &&
          matrix.distribution().global_size() == blocks.global_size()*width, "test: wrong BSR point-map size");
    for (std::size_t row = 0; row < gids.size(); ++row) for (int c = 0; c < width; ++c)
        check(matrix.distribution().global_id(static_cast<LocalOrdinal>(row*width+static_cast<std::size_t>(c))) ==
              gids[row]*width+c, "test: BSR point-map order/GIDs changed");
    const auto native_graph = matrix.native()->getLocalMatrixDevice().graph;
    const auto original_graph = graph.native()->getLocalGraphDevice();
    check(native_graph.entries.data() == original_graph.entries.data() &&
          native_graph.row_map.data() == original_graph.row_map.data(), "test: BSR duplicated the device graph");
    check(matrix.native()->getLocalNumEntries() == graph.local_nonzeros() &&
          matrix.local_values() == graph.local_nonzeros()*width*width, "test: BSR graph was expanded to scalar CSR");
    check(matrix.native()->getDomainMap()->isSameAs(*matrix.distribution().native()) &&
          matrix.native()->getRowMap()->isSameAs(*blocks.native()), "test: BSR block/point maps are confused");

    const auto values = block_coefficients<double>(blocks, input, width, false);
    matrix.set_values(values);
    Vector<double, Exec> x(pattern.distribution()), y(pattern.distribution()), reference(pattern.distribution());
    exact_values(x);
    const auto check_product = [&](double alpha, double beta, double initial) {
        y.fill(initial); matrix.apply(x, y, alpha, beta);
        auto local = std::as_const(*y.native()).getLocalViewHost(Tpetra::Access::ReadOnly);
        for (std::size_t row = 0; row < gids.size(); ++row) for (int a = 0; a < width; ++a) {
            double expected = beta*initial;
            for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k)
                for (int b = 0; b < width; ++b) {
                    const auto point_gid = input.column_gids[k]*width+b;
                    expected += alpha*values[(k*width+static_cast<std::size_t>(a))*width+static_cast<std::size_t>(b)]*
                                (1.0+0.125*double(point_gid%7));
                }
            check(std::abs(local(row*width+static_cast<std::size_t>(a), 0)-expected) < 1e-11,
                  "test: BSR apply differs from independent dense-block reference");
        }
    };
    check_product(1.7, -0.3, 2.0);

    // Independently expanded point CSR must implement the same operator.
    CsrInput point_input; point_input.row_offsets.push_back(0);
    std::vector<double> point_values;
    for (std::size_t row = 0; row < gids.size(); ++row) for (int a = 0; a < width; ++a) {
        for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k)
            for (int b = 0; b < width; ++b) {
                point_input.column_gids.push_back(input.column_gids[k]*width+b);
                point_values.push_back(values[(k*width+static_cast<std::size_t>(a))*width+static_cast<std::size_t>(b)]);
            }
        point_input.row_offsets.push_back(point_values.size());
    }
    CsrPattern<Exec> point_pattern(pattern.distribution(), point_input);
    CsrMatrix<double, Exec> point_matrix(point_pattern);
    point_matrix.set_values(point_values); point_matrix.apply(x, reference);
    matrix.apply(x, y);
    y.native()->update(-1.0, *reference.native(), 1.0);
    check(y.native()->norm2() < 1e-11, "test: BSR and expanded CSR operators differ");

    const auto exec = Kokkos::Experimental::partition_space(Exec{}, 1)[0];
    const auto flat = device_copy(values);
    const auto revision = matrix.revision();
    matrix.set_values_async(flat, exec); matrix.commit();
    check(matrix.revision() == revision+1, "test: BSR async assembly did not advance revision");
    check_product(1, 0, 0);
    Kokkos::View<double***, Kokkos::LayoutLeft, Space> tensor("test/blocks", input.column_gids.size(), width, width);
    auto host = Kokkos::create_mirror_view(tensor);
    for (std::size_t k = 0; k < input.column_gids.size(); ++k)
        for (int a = 0; a < width; ++a) for (int b = 0; b < width; ++b)
            host(k, a, b) = values[(k*width+static_cast<std::size_t>(a))*width+static_cast<std::size_t>(b)];
    Kokkos::deep_copy(exec, tensor, host);
    matrix.set_values_async(tensor, exec); matrix.commit();
    check_product(1, 0, 0);
    const auto after_write = matrix.revision();
    const auto read = matrix.read_blocks(exec);
    const auto target = copied.overwrite_blocks(exec);
    Kokkos::deep_copy(exec, target, read);
    copied.commit(); matrix.commit();
    check(matrix.revision() == after_write, "test: reading BSR coefficients changed revision");
    copied.apply(x, y); matrix.apply(x, reference);
    y.native()->update(-1.0, *reference.native(), 1.0);
    check(y.native()->norm2() < 1e-11, "test: BSR native block assembly differs");
    matrix.mark_modified();
    check(matrix.revision() == after_write+1, "test: BSR native edit did not advance revision");

    BsrMatrix<double, Exec> one(graph, 1);
    CsrMatrix<double, Exec> scalar(graph);
    const auto scalar_values = coefficients<double>(blocks, input, false);
    one.set_values(scalar_values); scalar.set_values(scalar_values);
    Vector<double, Exec> sx(blocks), sy(blocks), sr(blocks);
    exact_values(sx); one.apply(sx, sy); scalar.apply(sx, sr);
    sy.native()->update(-1.0, *sr.native(), 1.0);
    check(sy.native()->norm2() < 1e-13, "test: block size one differs from CSR");
}

template<class S, class P>
void bsr_solve_case(const std::string& method, const std::string& type, bool symmetric, bool reuse = false)
{
    constexpr int width = 5;
    auto gids = owned();
    // Uncoupled AMG needs local neighbours to exercise actual multilevel coarsening.
    if (type == "AMG" || type == "SCHWARZ")
        for (std::size_t i = 0; i < gids.size(); ++i)
            gids[i] = static_cast<GlobalOrdinal>(rank)*static_cast<GlobalOrdinal>(gids.size())+
                static_cast<GlobalOrdinal>(type == "AMG" ? i : gids.size()-1-i);
    Distribution<Exec> blocks(gids, mpi::CommView::world());
    const auto input = input_for(blocks);
    CsrPattern<Exec> graph(blocks, input);
    BsrPattern<Exec> pattern(graph, width);
    BsrMatrix<S, Exec> matrix(pattern);
    auto values = block_coefficients<S>(blocks, input, width, symmetric);
    matrix.set_values(values);
    Vector<S, Exec> exact(matrix.distribution()), rhs(matrix.distribution()), solution(matrix.distribution());
    exact_values(exact); matrix.apply(exact, rhs); solution.fill(S(0.25));
    SolverOptions options; options.method = method; options.preconditioner.type = type;
    options.relative_tolerance = std::is_same_v<S, float> ? 2e-5 : 1e-10;
    options.maximum_iterations = 300; options.restart_length = 15;
    if (type == "SCHWARZ") {
        options.preconditioner.parameters.sublist("inner preconditioner parameters").set("fact: iluk level-of-fill", 1);
    }
    if (type == "AMG") {
        options.preconditioner.parameters.set("coarse: max size", 4);
        options.preconditioner.parameters.set("max levels", 4);
        if (reuse) options.preconditioner.parameters.set("reuse: type", "full");
    }
    Solver<decltype(matrix), P> solver(matrix, options);
    if (rank == 0) std::fprintf(stderr, "[linalg] BSR solve %s/%s/%zu/%zu\n", method.c_str(), type.c_str(), sizeof(S), sizeof(P));
    const auto result = solver.solve(rhs, solution);
    if (!result.converged)
        std::fprintf(stderr, "BSR rank %d: %s/%s it=%d residual=%g threshold=%g krylov=%d error=%g\n",
            rank, method.c_str(), type.c_str(), result.iterations, result.residual_norm,
            result.residual_threshold, result.krylov_converged, error(solution));
    check(result.converged && error(solution) < (std::is_same_v<S, float> ? 5e-4 : 1e-8),
          "test: BSR solve failed independent solution/residual checks");
    const auto generation = solver.preconditioner().generation();
    const auto old = solver.preconditioner().native_inner().getRawPtr();
    if (type == "AMG") {
        using Amg = MueLu::TpetraOperator<P, LocalOrdinal, GlobalOrdinal, typename Types<P, Exec>::node_type>;
        const auto amg = Teuchos::rcp_dynamic_cast<Amg>(solver.preconditioner().native_inner(), true);
        check(amg->GetHierarchy()->GetNumLevels() > 1, "test: BSR AMG did not build multiple levels");
        using XMatrix = Xpetra::Matrix<P, LocalOrdinal, GlobalOrdinal, typename Types<P, Exec>::node_type>;
        for (int level = 0; level < amg->GetHierarchy()->GetNumLevels(); ++level) {
            const auto a = amg->GetHierarchy()->GetLevel(level)->template Get<Teuchos::RCP<XMatrix>>("A");
            check(a->GetStorageBlockSize() == width, "test: BSR AMG hierarchy lost block storage");
        }
    }
    if (rank == 0) {
        // Change only diagonal blocks, preserving symmetry for CG/MINRES.
        for (std::size_t row = 0; row < gids.size(); ++row)
            for (auto k = input.row_offsets[row]; k < input.row_offsets[row+1]; ++k)
                if (input.column_gids[k] == gids[row])
                    for (int a = 0; a < width*width; ++a) values[k*width*width+static_cast<std::size_t>(a)] *= S(1.25);
        matrix.set_values(values);
    }
    matrix.apply(exact, rhs); solution.fill(S(0));
    const auto changed = solver.solve(rhs, solution);
    check(changed.converged && error(solution) < (std::is_same_v<S, float> ? 5e-4 : 1e-8),
          "test: BSR rank-local revision left stale preconditioner coefficients");
    if (type != "NONE") check(solver.preconditioner().generation() == generation+1,
                              "test: BSR preconditioner revision was not refreshed collectively");
    if (reuse) check(solver.preconditioner().native_inner().getRawPtr() == old, "test: BSR AMG reuse replaced the operator");
}

void bsr_solvers()
{
    for (const auto* method : {"GMRES", "FGMRES", "BICGSTAB", "CG", "MINRES"})
        bsr_solve_case<double, double>(method, "NONE", std::string_view(method) == "CG" || std::string_view(method) == "MINRES");
    bsr_solve_case<double, double>("GMRES", "RELAXATION", false);
    bsr_solve_case<double, double>("CG", "RELAXATION", true);
    bsr_solve_case<double, double>("BICGSTAB", "RELAXATION", false);
}

void bsr_ifpack()
{
    bsr_solve_case<double, double>("GMRES", "SCHWARZ", false);
    bsr_solve_case<double, float>("FGMRES", "SCHWARZ", false);
}

void bsr_amg() { bsr_solve_case<double, double>("GMRES", "AMG", true); }
void bsr_amg_reuse() { bsr_solve_case<double, double>("GMRES", "AMG", true, true); }
void bsr_mixed_precision()
{
    bsr_solve_case<double, float>("FGMRES", "RELAXATION", false);
    bsr_solve_case<float, float>("GMRES", "RELAXATION", false);
    bsr_solve_case<float, double>("GMRES", "RELAXATION", false);
    bsr_solve_case<double, float>("FGMRES", "AMG", true);
}

void bsr_empty()
{
    for (const bool all_empty : {false, true}) {
        std::vector<GlobalOrdinal> gids;
        if (!all_empty && rank == 0) gids = {0};
        Distribution<Exec> blocks(gids, mpi::CommView::world());
        CsrInput input{gids.empty() ? std::vector<std::size_t>{0} : std::vector<std::size_t>{0, 1}, gids};
        CsrPattern<Exec> graph(blocks, input);
        BsrMatrix<double, Exec> matrix(graph, 3);
        matrix.set_values(block_coefficients<double>(blocks, input, 3, true));
        Vector<double, Exec> exact(matrix.distribution()), rhs(matrix.distribution()), solution(matrix.distribution());
        exact_values(exact); matrix.apply(exact, rhs);
        SolverOptions options; options.method = "CG"; options.preconditioner.type = "RELAXATION";
        Solver solver(matrix, options);
        check(solver.solve(rhs, solution).converged, "test: BSR solve failed on an empty local/global map");
    }
}

void bsr_factory()
{
    const auto gids = owned(3);
    Distribution<Exec> blocks(gids, mpi::CommView::world());
    const auto input = input_for(blocks);
    CsrPattern<Exec> graph(blocks, input);
    BsrMatrix<double, Exec> matrix(graph, 3);
    const auto values = block_coefficients<double>(blocks, input, 3, true);
    matrix.set_values(values);
    PreconditionerOptions options; options.type = "NONE";
    using Low = Preconditioner<decltype(matrix), float>;
    int calls = 0;
    Low low(matrix, options, [&](const Teuchos::RCP<const Low::matrix_type>& native,
                                 const Teuchos::RCP<Low::inner_operator_type>& previous) {
        static_assert(std::is_same_v<Low::matrix_type, Types<float, Exec>::block_matrix_type>);
        ++calls;
        check(native->getBlockSize() == 3 && native->getLocalNumEntries() == graph.local_nonzeros(),
              "test: mixed precision expanded BSR coefficients to CSR");
        check(native->getLocalMatrixDevice().graph.entries.data() == graph.native()->getLocalGraphDevice().entries.data(),
              "test: mixed-precision BSR did not share the device graph");
        const auto source = matrix.native()->getValuesHost();
        const auto target = native->getValuesHost();
        for (std::size_t i = 0; i < source.extent(0); ++i)
            check(target(i) == static_cast<float>(source(i)), "test: BSR precision conversion changed coefficient ordering");
        if (!previous.is_null()) return previous;
        return Teuchos::rcp_implicit_cast<Low::inner_operator_type>(
            Teuchos::rcp(new ScaleOperator<float>(matrix.distribution().native())));
    });
    low.prepare(); low.prepare();
    check(calls == 1, "test: BSR custom factory rebuilt unchanged coefficients");
    if (rank == 0) matrix.mark_modified();
    low.prepare();
    check(calls == 2, "test: BSR custom factory ignored a collective revision");
    using Same = Preconditioner<decltype(matrix), double>;
    Same same(matrix, options, [&](const Teuchos::RCP<const Same::matrix_type>& native,
                                   const Teuchos::RCP<Same::inner_operator_type>&) {
        check(native.getRawPtr() == matrix.native().getRawPtr(), "test: same-precision BSR allocated another matrix");
        return Teuchos::rcp_implicit_cast<Same::inner_operator_type>(
            Teuchos::rcp(new ScaleOperator<double>(matrix.distribution().native())));
    });
    same.prepare();
    check(same.native().getRawPtr() == same.native_inner().getRawPtr(), "test: same precision created an adapter");
}

template<class Layout>
void multiple_fields()
{
    const auto exec = Kokkos::Experimental::partition_space(Exec{}, 1)[0];
    const auto gids = owned(3);
    Distribution<Exec> cells(gids, mpi::CommView::world());
    const auto dofs = cells.expanded(5);
    Vector<double, Exec> vector(dofs);
    const auto owned_cells = gids.size(), extended = owned_cells+2;
    Kokkos::View<float*, Space> density("density", extended), density_out("density_out", extended);
    Kokkos::View<double*[3], Layout, Space> momentum("momentum", extended), momentum_out("momentum_out", extended);
    using EnergyLayout = std::conditional_t<std::is_same_v<Layout, Kokkos::LayoutLeft>, Kokkos::LayoutRight, Kokkos::LayoutLeft>;
    Kokkos::View<double*[2], EnergyLayout, Space> energy("energy", extended), energy_out("energy_out", extended);
    Kokkos::deep_copy(density_out, -7.0f); Kokkos::deep_copy(momentum_out, -8.0); Kokkos::deep_copy(energy_out, -9.0);
    Kokkos::parallel_for("test/multiple_fields", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>(exec, 0, extended),
        KOKKOS_LAMBDA(std::size_t cell) {
            density(cell) = static_cast<float>(100*cell);
            for (int c = 0; c < 3; ++c) momentum(cell, c) = double(100*cell)+c+1;
            energy(cell, 0) = -12; energy(cell, 1) = double(100*cell)+4;
        });
    using Rho = fields::FieldView<float, 1, Space>;
    using Mom = fields::FieldView<double, 3, Space, Layout>;
    Rho rho(typename Rho::view_type(density.data(), extended));
    Rho rho_out(typename Rho::view_type(density_out.data(), extended));
    Mom mom(typename Mom::view_type(momentum.data(), extended));
    Mom mom_out(typename Mom::view_type(momentum_out.data(), extended));
    const auto sources = field_pack(rho.as_const(), mom.as_const(), Kokkos::subview(energy, Kokkos::ALL(), 1));
    static_assert(decltype(sources)::components == 5);
    copy_from_fields(vector, sources, exec); vector.commit();
    {
        const auto h = std::as_const(*vector.native()).getLocalViewHost(Tpetra::Access::ReadOnly);
        for (std::size_t cell = 0; cell < owned_cells; ++cell) for (int c = 0; c < 5; ++c)
            check(h(cell*5+static_cast<std::size_t>(c), 0) == double(100*cell)+c,
                  "test: multiple-field packing has wrong AoS/component order");
    }
    copy_to_fields(vector, exec, rho_out, mom_out, Kokkos::subview(energy_out, Kokkos::ALL(), 1));
    vector.fill_async(99, exec); vector.commit(); // The read must finish before the overwrite.
    const auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, density_out);
    const auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, momentum_out);
    const auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, energy_out);
    for (std::size_t cell = 0; cell < extended; ++cell) {
        check(hr(cell) == (cell < owned_cells ? static_cast<float>(100*cell) : -7.0f), "test: density tail was changed");
        for (int c = 0; c < 3; ++c)
            check(hm(cell, c) == (cell < owned_cells ? double(100*cell)+c+1 : -8), "test: momentum pack/tail is wrong");
        check(he(cell, 0) == -9 && he(cell, 1) == (cell < owned_cells ? double(100*cell)+4 : -9),
              "test: strided energy unpack changed another component or extended tail");
    }
    // Five separate scalar components exercise the Navier-Stokes residual form.
    copy_from_fields(vector, exec, rho, mom.component(0), mom.component(1), mom.component(2),
                     Kokkos::subview(energy, Kokkos::ALL(), 1));
    vector.commit();
    const auto h = std::as_const(*vector.native()).getLocalViewHost(Tpetra::Access::ReadOnly);
    for (std::size_t i = 0; i < vector.local_size(); ++i)
        check(h(i, 0) == double(100*(i/5)+i%5), "test: scalar-component pack differs from mixed-field pack");
    Distribution<Exec> empty(std::span<const GlobalOrdinal>{}, mpi::CommView::world());
    const auto empty_dofs = empty.expanded(5);
    Vector<double, Exec> empty_vector(empty_dofs);
    copy_from_fields(empty_vector, sources, exec);
    copy_to_fields(empty_vector, exec, rho_out, mom_out, Kokkos::subview(energy_out, Kokkos::ALL(), 1));
    empty_vector.commit();
    const auto unchanged = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, density_out);
    for (std::size_t cell = 0; cell < extended; ++cell)
        check(unchanged(cell) == (cell < owned_cells ? static_cast<float>(100*cell) : -7.0f),
              "test: empty field pack changed destination data");
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
        "pending_matrix_read", "bicgstab_left", "bad_method", "precision_overflow",
        "bsr_zero_block", "bsr_wrong_map", "bsr_pending_read", "bsr_wrong_values",
        "bsr_precision_overflow", "pack_mismatch", "pack_short", "expanded_gid_overflow", "bsr_different_blocks",
        "bsr_overlap", "bsr_kokkos_amg"};
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
        krona::test::Case{"float_solver", with_tpetra<float_solver>},
        krona::test::Case{"bsr_mapping", with_tpetra<bsr_mapping>},
        krona::test::Case{"bsr_solvers", with_tpetra<bsr_solvers>},
        krona::test::Case{"bsr_ifpack", with_tpetra<bsr_ifpack>},
        krona::test::Case{"bsr_amg", with_tpetra<bsr_amg>},
        krona::test::Case{"bsr_amg_reuse", with_tpetra<bsr_amg_reuse>},
        krona::test::Case{"bsr_mixed_precision", with_tpetra<bsr_mixed_precision>},
        krona::test::Case{"bsr_empty", with_tpetra<bsr_empty>},
        krona::test::Case{"bsr_factory", with_tpetra<bsr_factory>},
        krona::test::Case{"multiple_fields_left", with_tpetra<multiple_fields<Kokkos::LayoutLeft>>},
        krona::test::Case{"multiple_fields_right", with_tpetra<multiple_fields<Kokkos::LayoutRight>>}};
    constexpr std::array negative_cases{
        krona::test::Case{"duplicate_column", with_tpetra<rejection<0>>},
        krona::test::Case{"missing_owner", with_tpetra<rejection<1>>},
        krona::test::Case{"pending_read", with_tpetra<rejection<2>>},
        krona::test::Case{"pending_write", with_tpetra<rejection<3>>},
        krona::test::Case{"pending_matrix_read", with_tpetra<rejection<4>>},
        krona::test::Case{"bicgstab_left", with_tpetra<rejection<5>>},
        krona::test::Case{"bad_method", with_tpetra<rejection<6>>},
        krona::test::Case{"precision_overflow", with_tpetra<rejection<7>>},
        krona::test::Case{"bsr_zero_block", with_tpetra<rejection<8>>},
        krona::test::Case{"bsr_wrong_map", with_tpetra<rejection<9>>},
        krona::test::Case{"bsr_pending_read", with_tpetra<rejection<10>>},
        krona::test::Case{"bsr_wrong_values", with_tpetra<rejection<11>>},
        krona::test::Case{"bsr_precision_overflow", with_tpetra<rejection<12>>},
        krona::test::Case{"pack_mismatch", with_tpetra<rejection<13>>},
        krona::test::Case{"pack_short", with_tpetra<rejection<14>>},
        krona::test::Case{"expanded_gid_overflow", with_tpetra<rejection<15>>},
        krona::test::Case{"bsr_different_blocks", with_tpetra<rejection<16>>},
        krona::test::Case{"bsr_overlap", with_tpetra<rejection<17>>},
        krona::test::Case{"bsr_kokkos_amg", with_tpetra<rejection<18>>}};
    if (argc == 2) {
        const std::string_view selected = argv[1];
        for (const auto& entry : negative_cases)
            if (selected == entry.name) return krona::test::run(argc, argv, "linalg", negative_cases);
        if (selected == "all") argc = 1;
    }
    return krona::test::run(argc, argv, "linalg", positive_cases);
}