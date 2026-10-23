#include "common/test_support.hpp"

#include <krona/transport/scalar/transient/operator.hpp>
#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/csr_matrix.hpp>

#include <array>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {
using namespace krona;
namespace tt = transport::scalar::transient;
namespace ts = transport::scalar;
namespace nt = numerics::time;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
template<class T> using View = Kokkos::View<T*, Space>;
using test::check;

template<class T>
View<T> input(const Exec& exec, std::initializer_list<T> values)
{ 
    View<T> result(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/input"), values.size());
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, result);
    std::size_t i = 0;
    for (const auto value : values) host(i++) = value;
    Kokkos::deep_copy(exec, result, host);
    exec.fence("Test input ready");
    return result;
}
template<class V> auto host_copy(V v)
{ return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v); }
struct Entries { View<std::size_t> diagonal; };

void coefficients(const Exec&)
{
    const auto e = nt::BackwardEuler{}.coefficients(0.25);
    test::equal(e.weights[0], 4.0, "Euler diagonal");
    test::equal(e.weights[1], -4.0, "Euler history");
    const auto b = nt::Bdf2{}.coefficients(0.25);
    test::equal(b.weights[0], 6.0, "BDF2 diagonal");
    test::equal(b.weights[1], -8.0, "BDF2 newest history");
    test::equal(b.weights[2], 2.0, "BDF2 older history");
    const auto v = nt::Bdf2{}.coefficients(0.3, 0.2);
    // Exact derivative of 2+3*t+4*t*t at t=0.5, for unequal steps.
    const double derivative = v.weights[0]*4.5+v.weights[1]*2.76+v.weights[2]*2.0;
    test::near(derivative, 7.0, 1.e-13, "Variable-step BDF2 differentiates a quadratic");
    bool rejected = false;
    try { static_cast<void>(nt::BackwardEuler{}.coefficients(0.0)); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Zero time step must fail");
    rejected = false;
    try { static_cast<void>(nt::Bdf2{}.coefficients(1.0, -1.0)); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Negative previous step must fail");
    rejected = false;
    try { static_cast<void>(nt::BackwardEuler{}.coefficients(std::numeric_limits<double>::denorm_min())); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Overflowing derivative coefficients must fail");
}

void additive(const Exec& exec)
{
    auto values = input<double>(exec, {10, 20, 30, 40, 50, 60});
    auto rhs = input<double>(exec, {1, 2, 3});
    auto phi = input<double>(exec, {4, 5, 6});
    auto old = input<double>(exec, {2, 3, 4});
    auto mass = input<double>(exec, {2, 4, 6});
    const Entries entries{input<std::size_t>(exec, {2, 0, 4})};
    const auto h = tt::time_history(tt::TimeLevel{tt::UniformMass{3.0}, old});
    const auto derivative = nt::BackwardEuler{}.coefficients(0.5);
    tt::TransientOperator<Exec, double> op(3, exec);
    ts::AssemblyStatus<Exec> status(exec);
    auto residual = input<double>(exec, {8, 9, 10});
    test::complete(exec, [&] {
        status.begin_async();
        op.add_system_async(ts::SystemView{values, rhs, entries}, status, derivative, tt::CellMass{mass}, h);
        op.add_residual_async(phi, residual, status, derivative, tt::CellMass{mass}, h);
        const auto report = status.check(mpi::CommView::world());
        check(!report.has_diffusion_anchor, "A temporal diagonal is not a diffusion boundary anchor");
    });
    const auto a = host_copy(values), b = host_copy(rhs), r = host_copy(residual);
    const auto m = host_copy(mass), x = host_copy(phi), p = host_copy(old);
    const auto map = host_copy(entries.diagonal);
    const std::array<double, 6> original{10, 20, 30, 40, 50, 60};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto entry = map(i);
        const double added_a = 2*m(i), added_b = 6*p(i);
        test::near(a(entry), original[entry]+added_a, 1.e-13, "Preserve existing matrix coefficients");
        test::near(b(i), static_cast<double>(i+1)+added_b, 1.e-13, "Preserve existing RHS");
        test::near(r(i), static_cast<double>(i+8)+added_b-added_a*x(i), 1.e-13, "Residual equals b_time-A_time*phi");
    }
    test::equal(a(1), 20.0, "Off-diagonal entry preserved");
    test::equal(a(3), 40.0, "Off-diagonal entry preserved");
    test::equal(a(5), 60.0, "Off-diagonal entry preserved");
}

void conservation(const Exec& exec)
{
    const auto old = input<double>(exec, {3, 6});
    const auto older = input<double>(exec, {1.5, 3});
    const auto current = input<double>(exec, {2, 4});
    const auto volumes = input<double>(exec, {1, 1});
    const auto density = input<double>(exec, {3, 3});
    auto residual = input<double>(exec, {0, 0});
    tt::TransientOperator<Exec, double> op(2, exec);
    ts::AssemblyStatus<Exec> status(exec);
    // All layers have M*phi={6,12}, although mass and scalar vary.
    const auto h = tt::time_history(tt::TimeLevel{tt::UniformMass{2.0}, old},
                                  tt::TimeLevel{tt::UniformMass{4.0}, older});
    test::complete(exec, [&] {
        status.begin_async();
        op.add_residual_async(current, residual, status, nt::Bdf2{}.coefficients(0.3, 0.2),
            tt::DensityVolumeMass{volumes, density}, h);
        static_cast<void>(status.check(mpi::CommView::world()));
    });
    const auto r = host_copy(residual);
    for (std::size_t i = 0; i < 2; ++i) test::near(r(i), 0.0, 1.e-12, "Conserve integrated quantity for changing mass");
}

void strided(const Exec& exec)
{
    Kokkos::View<float***, Kokkos::LayoutRight, Space> blocks("test/blocks", 2, 3, 3);
    Kokkos::View<float**, Kokkos::LayoutLeft, Space> rhs("test/rhs", 6, 1);
    Kokkos::deep_copy(exec, blocks, 7.f);
    Kokkos::deep_copy(exec, rhs, 11.f);
    const auto slice = Kokkos::subview(blocks, Kokkos::ALL(), 1, 1);
    using Rhs = Kokkos::View<float*, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Rhs b(rhs.data()+rhs.stride(0), Kokkos::LayoutStride(2, 3*rhs.stride(0)));
    const Entries entries{input<std::size_t>(exec, {0, 1})};
    const auto old = input<float>(exec, {2, 3});
    const auto volumes = input<float>(exec, {4, 5});
    ts::AssemblyStatus<Exec> status(exec);
    tt::TransientOperator<Exec, double> op(2, exec);
    test::complete(exec, [&] {
        status.begin_async();
        op.add_system_async(ts::SystemView{slice, b, entries}, status, nt::BackwardEuler{}.coefficients(0.5),
            tt::DensityVolumeMass{volumes, 2.f}, tt::time_history(tt::TimeLevel{tt::UniformMass{3.f}, old}));
        static_cast<void>(status.check(mpi::CommView::world()));
    });
    const auto a = host_copy(blocks);
    const auto r = host_copy(rhs);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t col = 0; col < 3; ++col)
                test::equal(a(i, row, col), row == 1 && col == 1 ? 7.f+4.f*static_cast<float>(i+4) : 7.f,
                            "Only selected diagonal component changes");
    for (std::size_t i = 0; i < 6; ++i)
        test::equal(r(i, 0), i == 1 ? 23.f : i == 4 ? 29.f : 11.f, "AoS RHS component stride");
}

double decay(const Exec& exec, int steps, bool bdf, bool variable = false)
{
    auto value = input<double>(exec, {1});
    auto previous = input<double>(exec, {1});
    auto older = input<double>(exec, {1});
    auto a = input<double>(exec, {0}), b = input<double>(exec, {0});
    const Entries entries{input<std::size_t>(exec, {0})};
    const double base_dt = 1.0/static_cast<double>(steps);
    double previous_dt = base_dt;
    ts::AssemblyStatus<Exec> status(exec);
    tt::TransientOperator<Exec, double> op(1, exec);
    for (int step = 0; step < steps; ++step) {
        const double dt = base_dt*(variable ? (step%2 == 0 ? 0.8 : 1.2) : 1.0);
        // q'=-q. The spatial reaction is A=1, b=0.
        Kokkos::deep_copy(exec, a, 1.0);
        Kokkos::deep_copy(exec, b, 0.0);
        status.begin_async();
        const auto newest = tt::TimeLevel{tt::UniformMass{1.0}, previous};
        if (bdf && step > 0)
            op.add_system_async(ts::SystemView{a, b, entries}, status, nt::Bdf2{}.coefficients(dt, previous_dt), tt::UniformMass{1.0},
                tt::time_history(newest, tt::TimeLevel{tt::UniformMass{1.0}, older}));
        else
            op.add_system_async(ts::SystemView{a, b, entries}, status, nt::BackwardEuler{}.coefficients(dt),
                tt::UniformMass{1.0}, tt::time_history(newest));
        static_cast<void>(status.check(mpi::CommView::world()));
        const double next = host_copy(b)(0)/host_copy(a)(0);
        Kokkos::deep_copy(exec, value, next);
        // Accept the completed step; never rotate during assembly/outer iterations.
        Kokkos::deep_copy(exec, older, previous);
        Kokkos::deep_copy(exec, previous, value);
        previous_dt = dt;
    }
    exec.fence("Decay history complete");
    return std::abs(host_copy(value)(0)-std::exp(-1.0));
}
void order(const Exec& exec)
{
    const double be_ratio = decay(exec, 20, false)/decay(exec, 40, false);
    const double bdf_ratio = decay(exec, 20, true)/decay(exec, 40, true);
    check(be_ratio > 1.9 && be_ratio < 2.1, "Backward Euler is first order");
    check(bdf_ratio > 3.8 && bdf_ratio < 4.3, "BDF2 with Euler startup is second order");
    const double variable_ratio = decay(exec, 20, true, true)/decay(exec, 40, true, true);
    check(variable_ratio > 3.7 && variable_ratio < 4.4, "Variable-step BDF2 is second order");
}

void empty(const Exec& exec)
{
    View<double> values("test/empty", 0);
    Entries entries{View<std::size_t>("test/empty_map", 0)};
    ts::AssemblyStatus<Exec> status(exec);
    tt::TransientOperator<Exec, double> op(0, exec);
    status.begin_async();
    const auto h = tt::time_history(tt::TimeLevel{tt::UniformMass{1.0}, values});
    op.add_system_async(ts::SystemView{values, values, entries}, status, nt::BackwardEuler{}.coefficients(1.0), tt::UniformMass{1.0}, h);
    op.add_residual_async(values, values, status, nt::BackwardEuler{}.coefficients(1.0), tt::UniformMass{1.0}, h);
    static_cast<void>(status.check(mpi::CommView::world()));
}

void native(const Exec& exec)
{
    Tpetra::ScopeGuard tpetra(nullptr, nullptr);
    const auto comm = mpi::CommView::world();
    const std::vector<linalg::GlobalOrdinal> gids{2*comm.rank(), 2*comm.rank()+1};
    const linalg::Distribution<Exec> distribution(gids, comm);
    const linalg::CsrInput graph{{0, 1, 2}, gids};
    const linalg::CsrPattern<Exec> pattern(distribution, graph, exec);
    const Entries entries{input<std::size_t>(exec, {0, 1})};
    const auto old = input<double>(exec, {2, 4});
    const auto history = tt::time_history(tt::TimeLevel{tt::UniformMass{3.0}, old});
    tt::TransientOperator<Exec, double> op(2, exec);
    ts::AssemblyStatus<Exec> status(exec);
    using NativeScalar = SolverReal;
    linalg::CsrMatrix<NativeScalar, Exec> csr(pattern);
    linalg::Vector<NativeScalar, Exec> csr_rhs(distribution);
    const auto cv = csr.edit_values(exec);
    const auto cr = Kokkos::subview(csr_rhs.edit(exec), Kokkos::ALL(), 0);
    Kokkos::deep_copy(exec, cv, 0.0); Kokkos::deep_copy(exec, cr, 0.0);
    status.begin_async();
    op.add_system_async(ts::SystemView{cv, cr, entries}, status, nt::BackwardEuler{}.coefficients(0.5), tt::UniformMass{2.0}, history);
    static_cast<void>(status.check(comm)); csr.commit(); csr_rhs.commit();
    const auto ca = host_copy(cv);
    const auto cb = host_copy(cr);
    for (std::size_t i = 0; i < 2; ++i) {
        test::equal(ca(i), 4.0, "Native CSR temporal diagonal");
        test::equal(cb(i), 12.0*static_cast<double>(i+1), "Native CSR temporal RHS");
    }
    linalg::BsrMatrix<NativeScalar, Exec> bsr(pattern, 3);
    linalg::Vector<NativeScalar, Exec> bsr_rhs(bsr.distribution());
    const auto blocks = bsr.edit_blocks(exec);
    const auto br = bsr_rhs.edit(exec);
    Kokkos::deep_copy(exec, blocks, 7.0); Kokkos::deep_copy(exec, br, 11.0);
    const auto bs = Kokkos::subview(blocks, Kokkos::ALL(), 1, 1);
    using Rhs = Kokkos::View<NativeScalar*, Kokkos::LayoutStride, typename decltype(br)::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Rhs rb(br.data()+br.stride(0), Kokkos::LayoutStride(2, 3*br.stride(0)));
    status.begin_async();
    op.add_system_async(ts::SystemView{bs, rb, entries}, status, nt::BackwardEuler{}.coefficients(0.5), tt::UniformMass{2.0}, history);
    static_cast<void>(status.check(comm)); bsr.commit(); bsr_rhs.commit();
    const auto ba = host_copy(blocks);
    const auto bb = host_copy(br);
    for (std::size_t i = 0; i < 2; ++i) {
        test::equal(ba(i, 1, 1), 11.0, "Native BSR selected diagonal");
        test::equal(ba(i, 0, 1), 7.0, "Native BSR cross component preserved");
        test::equal(bb(3*i+1, 0), 11.0+12.0*static_cast<double>(i+1), "Native BSR AoS RHS");
        test::equal(bb(3*i, 0), 11.0, "Native BSR other RHS preserved");
    }
}

void rejection(const Exec& exec, std::string_view name)
{
    auto a = input<float>(exec, {0}), b = input<float>(exec, {0});
    auto old = input<double>(exec, {name == "bad_state" ? std::numeric_limits<double>::quiet_NaN() : 2.0});
    Entries entries{input<std::size_t>(exec, {name == "bad_entry" ? 3u : 0u})};
    ts::AssemblyStatus<Exec> status(exec);
    tt::TransientOperator<Exec, double> op(1, exec);
    if (name != "not_started") status.begin_async();
    const auto derivative = nt::BackwardEuler{}.coefficients(name == "overflow" ? 1.e-100 : 1.0);
    const auto mass = tt::UniformMass{name == "bad_mass" ? -1.0 : 1.0};
    const auto history = tt::time_history(tt::TimeLevel{tt::UniformMass{1.0}, old});
    op.add_system_async(ts::SystemView{a, b, entries}, status, derivative, mass, history);
    static_cast<void>(status.check(mpi::CommView::world()));
    check(false, "Expected rejection");
}
void bad_mass(const Exec& e) { rejection(e, "bad_mass"); }
void bad_state(const Exec& e) { rejection(e, "bad_state"); }
void bad_entry(const Exec& e) { rejection(e, "bad_entry"); }
void overflow(const Exec& e) { rejection(e, "overflow"); }
void not_started(const Exec& e) { rejection(e, "not_started"); }
} // namespace

int main(int argc, char** argv)
{
    const std::array cases{
        test::Case{"coefficients", coefficients}, test::Case{"additive", additive},
        test::Case{"conservation", conservation}, test::Case{"strided", strided},
        test::Case{"order", order}, test::Case{"empty", empty}, test::Case{"native", native},
        test::Case{"bad_mass", bad_mass}, test::Case{"bad_state", bad_state},
        test::Case{"bad_entry", bad_entry}, test::Case{"overflow", overflow}, test::Case{"not_started", not_started}
    };
    // Failure cases are selected explicitly and run in separate processes.
    return test::run(argc, argv, "transport.scalar.transient",
        argc == 1 ? std::span<const test::Case>{cases.data(), 7} : std::span<const test::Case>{cases});
}
