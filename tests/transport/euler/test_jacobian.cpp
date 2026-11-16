#include <common/test_support.hpp>
#include "fixture.hpp"
#include <krona/transport/euler/assembly.hpp>
#include <Tpetra_Core.hpp>

namespace {
using namespace euler_test;
namespace la = krona::linalg;
using krona::test::check;
using krona::test::near;

using U = ri::Conservative<R>;
using Delta = ri::Conservative<double>;

template<class T>
auto components(ri::Conservative<T> q)
{ return std::array<T, 5>{q.mass, q.momentum.x, q.momentum.y, q.momentum.z, q.energy}; }

Delta direction(krona::GlobalIndex gid)
{
    const double scale = 1.0 + static_cast<double>(gid) / 4.0;
    return {0.2 * scale, {0.03, -0.09 * scale, 0.14}, 0.32 * scale};
}

void state_derivatives()
{
    krona::thermodynamics::IdealGas<double> eos;
    eos.gas_constant = 1.4;
    const auto state = ri::from_pT(eos, 1.2, 0.95, ri::Vector<double>{1.1, -0.3, 0.2});
    const auto to_w = ri::primitive_from_conservative_jacobian(state, eos);
    const auto to_u = ri::conservative_from_primitive_jacobian(state, eos);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) {
            double coefficient = 0;
            for (std::size_t k = 0; k < 5; ++k) coefficient += to_u(i, k) * to_w(k, j);
            near(coefficient, i == j ? 1.0 : 0.0, 1e-12, "analytic state transformations are inverses");
        }

    const ri::Vector<double> n{0.6, 0, 0.8};
    const auto du = direction(0);
    const auto u = ri::conservative(state);
    const double h = 1e-5;
    const auto plus = ri::from_conservative(eos, u + du * h);
    const auto minus = ri::from_conservative(eos, u - du * h);
    const auto fd = components((ri::physical_flux(plus, n) - ri::physical_flux(minus, n)) * (0.5 / h));
    const auto a = ri::physical_flux_jacobian(state, n, eos);
    const auto d = components(du);
    for (std::size_t row = 0; row < 5; ++row) {
        double product = 0;
        for (std::size_t col = 0; col < 5; ++col) product += a(row, col) * d[col];
        near(product, fd[row], 1e-8 * std::max(1.0, std::abs(fd[row])), "analytic Euler flux derivative");
    }
}

void set_halo(Grid& grid, krona::mpi::CommView comm)
{
    if (comm.size() == 1) return;
    grid.host.nb_ranks = {1 - comm.rank()};
    grid.host.send_owned_local = {comm.rank() == 0 ? 1 : 0};
    grid.host.recv_ghost_local = {grid.host.n_own};
    grid.host.send_offsets = {0, 1};
    grid.host.recv_offsets = {0, 1};
}

auto matrix_product(la::BsrMatrix<krona::SolverReal, Exec>& matrix, const Grid& grid, const Exec& exec)
{
    using Scalar = krona::SolverReal;
    matrix.commit();
    la::Vector<Scalar, Exec> x(matrix.distribution()), y(matrix.distribution());
    const auto x_view = x.overwrite(exec);
    auto host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, x_view);
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(grid.host.n_own); ++cell) {
        const auto d = components(direction(grid.host.cell_gid[cell]));
        for (std::size_t c = 0; c < 5; ++c) host(5 * cell + c, 0) = static_cast<Scalar>(d[c]);
    }
    Kokkos::deep_copy(exec, x_view, host);
    x.commit();
    matrix.apply(x, y);
    const auto y_view = y.read(exec);
    const auto result = Kokkos::create_mirror(Kokkos::WithoutInitializing, y_view);
    Kokkos::deep_copy(exec, result, y_view);
    y.commit();
    return result;
}

void inner_blocks(const Exec& exec, Grid& grid)
{
    auto comm = grid.plan.communicator();
    Fields<> fields(grid, exec);
    fields.initialize(grid, exec, true);
    const la::FaceAssemblyPlan assembly(grid.host, comm, exec);
    la::BsrMatrix matrix(assembly.pattern(), 5);
    const eu::Thermodynamics law{gas()};
    const eu::EulerOperator op{grid.plan, law};
    eu::AssemblyStatus<Exec> status(exec);
    const auto target = eu::make_jacobian_view(matrix, assembly, exec);
    Kokkos::deep_copy(exec, target.blocks, krona::SolverReal{0});
    status.begin_async();
    const auto inner = eu::FrozenRusanov{gas()};
    const auto split = grid.host.n_internal_faces;
    op.add_inner_jacobian_async(target, fields.primitive(), status, inner, {0, split});
    op.add_inner_jacobian_async(target, fields.primitive(), status, inner, {split, grid.host.n_inner_faces});
    (void)status.check(comm);
    const auto product = matrix_product(matrix, grid, exec);

    // Independent finite differences of the frozen Rusanov flux, with native remote-column import.
    const auto eos = krona::thermodynamics::IdealGas<double>{static_cast<double>(gas().gamma), 1.0};
    using D = ri::Conservative<double>;
    std::array<D, 2> expected{};
    for (std::size_t f = 0; f < static_cast<std::size_t>(grid.host.n_inner_faces); ++f) {
        const auto i = static_cast<std::size_t>(grid.host.face_owner[f]);
        const auto j = static_cast<std::size_t>(grid.host.face_neigh[f]);
        const auto state = [&](std::size_t cell) {
            const auto q = profile(grid.host.cell_centroid_x[cell], true);
            return ri::from_pT(eos, static_cast<double>(static_cast<S>(q.pressure)),
                static_cast<double>(static_cast<S>(q.temperature)),
                ri::Vector<double>{static_cast<double>(static_cast<S>(q.velocity.x)), 0, 0});
        };
        const auto l = state(i), r = state(j);
        const ri::Vector<double> n{grid.host.face_normal_x[f], grid.host.face_normal_y[f], grid.host.face_normal_z[f]};
        const double speed = std::max(std::abs(l.velocity.dot(n)) + l.sound_speed,
                                      std::abs(r.velocity.dot(n)) + r.sound_speed);
        const auto flux = [&](double step) {
            const auto perturb = [&](auto q, std::size_t cell) {
                return ri::from_conservative(eos, ri::conservative(q) + direction(grid.host.cell_gid[cell]) * step);
            };
            const auto a = perturb(l, i), b = perturb(r, j);
            return (ri::physical_flux(a, n) + ri::physical_flux(b, n) -
                    (ri::conservative(b) - ri::conservative(a)) * speed) * 0.5;
        };
        const double step = 1e-5;
        const auto derivative = (flux(step) - flux(-step)) * (grid.host.face_area[f] / (2 * step));
        expected[i] = expected[i] + derivative;
        if (j < expected.size()) expected[j] = expected[j] - derivative;
    }
    for (std::size_t cell = 0; cell < expected.size(); ++cell) {
        const auto e = components(expected[cell]);
        for (std::size_t c = 0; c < 5; ++c)
            near(product(5 * cell + c, 0), e[c], std::max(1e-8, tolerance()) * std::max(1.0, std::abs(e[c])),
                 "BSR inner Jacobian product");
    }
}

struct TimeDependentOutletPressure {
    KOKKOS_INLINE_FUNCTION R operator()(const krona::boundary::FaceContext<R>& c) const
    { return R{1} - R{1} / R{100} * c.time; }
};

struct ZeroInner {
    KOKKOS_INLINE_FUNCTION eu::FaceJacobians<R> operator()(ri::State<R>, ri::State<R>, ri::Vector<R>) const
    { return {}; }
};

void boundary_blocks(const Exec& exec, Grid& grid)
{
    const auto comm = grid.plan.communicator();
    Fields<> fields(grid, exec);
    const auto read_gas = gas();
    using CharacteristicModel = ph::Compressible<Gas, ph::PerfectGasCharacteristics>;
    const CharacteristicModel model{read_gas, {}};
    krona::boundary::BoundaryLayout<Exec, R> layout(grid.host, grid.storage.view(), exec);
    const auto outlet = ph::pressure_outlet<R>(TimeDependentOutletPressure{});
    ph::Manager<CharacteristicModel, Exec, R, decltype(outlet)> boundary(layout, model);
    boundary.set(0, ph::SupersonicInlet<R>{R{1}, R{1}, {R{3}, 0, 0}});
    boundary.set(1, outlet);
    boundary.set(2, ph::SlipWall<R>{});
    boundary.finalize();
    const eu::Thermodynamics law{read_gas};
    const eu::EulerOperator op{grid.plan, law, ri::Hllc<>{}};
    eu::AssemblyStatus<Exec> status(exec);
    const auto closure = eu::EvaluatedExterior<>{};
    const R time = 2;
    const auto n_cells = static_cast<std::size_t>(grid.host.n_cells);
    krona::fields::FieldStorage<S, 15, Exec> gradient("boundary_gradients", n_cells,
        krona::fields::FieldLocation::Custom, exec);
    auto gradient_host = gradient.create_host_buffer();
    for (std::size_t cell = 0; cell < n_cells; ++cell)
        for (std::size_t c = 0; c < 15; ++c)
            gradient_host(cell, c) = c == 0 ? S{1} / S{10} : (c == 12 ? S{1} / S{25} : S{0});
    krona::execution::complete(exec, [&] { Kokkos::deep_copy(exec, gradient.view().kokkos_view(), gradient_host); });
    const auto reconstruction = eu::Reconstruction{rec::SecondOrderUpwind{},
        rec::data(fields.primitive(), rec::gradient_pack(gradient.view())),
        rec::MeshStencil<Space, R>{grid.storage.view(), grid.connectivity.view()}};

    const auto fill = [&](R step) {
        auto p = fields.pressure.create_host_buffer();
        auto T = fields.temperature.create_host_buffer();
        auto u = fields.velocity.create_host_buffer();
        for (std::size_t cell = 0; cell < fields.pressure.size(); ++cell) {
            ph::Primitive<R, true> w{R{1}, {R{2} / R{5}, R{1} / R{10}, -R{1} / R{10}}, R{1}};
            if (cell < grid.host.cell_gid.size()) {
                const auto state = ri::conservative(law(w));
                const auto d = direction(grid.host.cell_gid[cell]);
                const U delta{static_cast<R>(d.mass), krona::math::cast<R>(d.momentum), static_cast<R>(d.energy)};
                const auto changed = state + delta * step;
                w = model.primitive(ph::Conservative<R>{changed.mass, changed.momentum, changed.energy});
            }
            p(cell) = static_cast<S>(w.pressure);
            T(cell) = static_cast<S>(w.temperature);
            u(cell, 0) = static_cast<S>(w.velocity.x);
            u(cell, 1) = static_cast<S>(w.velocity.y);
            u(cell, 2) = static_cast<S>(w.velocity.z);
        }
        krona::execution::complete(exec, [&] {
            Kokkos::deep_copy(exec, fields.pressure.view().kokkos_view(), p);
            Kokkos::deep_copy(exec, fields.temperature.view().kokkos_view(), T);
            Kokkos::deep_copy(exec, fields.velocity.view().kokkos_view(), u);
        });
    };

    fill(0);
    const la::FaceAssemblyPlan assembly(grid.host, comm, exec);
    la::BsrMatrix matrix(assembly.pattern(), 5);
    const auto target = eu::make_jacobian_view(matrix, assembly, exec);
    Kokkos::deep_copy(exec, target.blocks, krona::SolverReal{0});
    status.begin_async();
    op.add_jacobian_async(target, reconstruction, boundary, status, ZeroInner{}, closure, time);
    (void)status.check(comm);
    const auto product = matrix_product(matrix, grid, exec);

    const auto residual = [&](R step) {
        fill(step);
        fields.zero(exec);
        status.begin_async();
        op.add_boundary_residual_async(reconstruction, fields.output(), boundary, status, closure, time);
        (void)status.check(comm);
        return fields.snapshot();
    };
    const R h = static_cast<R>(std::cbrt(std::max(static_cast<double>(Kokkos::epsilon_v<R>),
                                                static_cast<double>(Kokkos::epsilon_v<S>))));
    const auto plus = residual(h), minus = residual(-h);
    for (std::size_t cell = 0; cell < plus.extent(0); ++cell)
        for (std::size_t c = 0; c < 5; ++c) {
            const double reference = (minus(cell, c) - plus(cell, c)) / (2.0 * static_cast<double>(h));
            near(product(5 * cell + c, 0), reference, std::max(2e-7, 4.0 * tolerance()) * std::max(1.0, std::abs(reference)),
                 "physical boundary blocks differentiate negative residual in conservative variables");
        }
}

void jacobian(const Exec& exec)
{
    const auto comm = krona::mpi::CommView::world();
    Tpetra::ScopeGuard tpetra(nullptr, nullptr, comm.native());
    check(comm.size() <= 2, "Jacobian fixture supports up to two ranks");
    state_derivatives();
    Grid grid(comm, exec, 2 * comm.rank(), 2, 2 * comm.size(), false, 1.25, 1.5, true);
    set_halo(grid, comm);
    inner_blocks(exec, grid);
    boundary_blocks(exec, grid);
}
} // namespace

int main(int argc, char** argv)
{
    const std::array cases{krona::test::Case{"assembly", jacobian}};
    return krona::test::run(argc, argv, "transport.euler.jacobian", cases);
}