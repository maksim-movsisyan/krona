#include <common/test_support.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/numerics/gradient/correction.hpp>

#include <array>
#include <string_view>
#include <type_traits>

namespace {
using namespace krona::test;
namespace gradient = krona::numerics::gradient;
namespace fields = krona::fields;
using Space = typename Exec::memory_space;
using PolicySize = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

struct BoundaryContext { int owner; double tx, ty, tz; };
struct InteriorContext { int owner, neighbour; double owner_weight; };
 
template<class View>
void initialize_pair(const Exec& exec, View g)
{
    Kokkos::parallel_for("test/correction_initialize", PolicySize(exec, 0, 2), KOKKOS_LAMBDA(std::size_t i) {
        g(i, 0) = i == 0 ? 2.0 : 6.0;
        g(i, 1) = i == 0 ? -3.0 : 5.0;
        g(i, 2) = i == 0 ? 4.0 : -2.0;
    });
}

void boundary(const Exec& exec)
{
    Kokkos::View<double*[3], Kokkos::LayoutRight, Space> data("correction/g", 2);
    const auto g = gradient::cell_gradients(Kokkos::subview(data, Kokkos::ALL(), 0),
        Kokkos::subview(data, Kokkos::ALL(), 1), Kokkos::subview(data, Kokkos::ALL(), 2));
    g.template validate<Exec>(2);
    static_assert(std::is_same_v<decltype(g.x(0)), const double&>);
    static_assert(std::is_trivially_copyable_v<decltype(g)>);
    const gradient::GradientCorrection correction{g};
    Kokkos::View<double*, Space> output("correction/boundary", 4);
    auto host = Kokkos::create_mirror(output);
    complete(exec, [&] {
        initialize_pair(exec, data);
        Kokkos::parallel_for("test/correction_boundary", PolicySize(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            output(0) = correction.increment(0, 0.25, -0.5, 0.125);
            output(1) = correction(BoundaryContext{0, 0.25, -0.5, 0.125});
            output(2) = correction.increment(0, -0.25, 0.5, -0.125);
            output(3) = correction.increment(0, 0.0, 0.0, 0.0);
        });
        Kokkos::deep_copy(exec, host, output);
    });
    equal(host(0), 2.5, "Tangential displacement dotted with the owner gradient");
    equal(host(1), 2.5, "Structural boundary context adapter");
    equal(host(2), -2.5, "Projection reverses with the geometric vector");
    equal(host(3), 0.0, "Zero displacement gives zero correction");
}

void interior(const Exec& exec)
{
    fields::FieldStorage<double, 3, Exec> storage("grad", 2, fields::FieldLocation::Custom, exec);
    const gradient::GradientCorrection correction{gradient::cell_gradients(storage.view())};
    Kokkos::View<double*, Space> output("correction/interior", 5);
    auto host = Kokkos::create_mirror(output);
    const auto values = storage.view();
    complete(exec, [&] {
        initialize_pair(exec, values);
        Kokkos::parallel_for("test/correction_interior", PolicySize(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            output(0) = correction.interior(0, 1, 0.25, 0.5, -0.25, 2.0);
            output(1) = correction.interior(InteriorContext{0, 1, 0.25}, 0.5, -0.25, 2.0);
            output(2) = correction.interior(1, 0, 0.75, -0.5, 0.25, -2.0);
            output(3) = correction.interior(0, 1, 1.0, 0.5, -0.25, 2.0);
            output(4) = correction.interior(0, 1, 0.0, 0.5, -0.25, 2.0);
        });
        Kokkos::deep_copy(exec, host, output);
    });
    equal(host(0), 0.75, "Owner weight is not replaced by a midpoint weight");
    equal(host(1), 0.75, "Structural interior context adapter");
    equal(host(2), -0.75, "Swapped endpoints and reversed vector preserve orientation");
    equal(host(3), 9.75, "Unit owner weight selects the owner gradient");
    equal(host(4), -2.25, "Zero owner weight selects the neighbour gradient");
}

template<class Layout>
void layout_case(const Exec& exec)
{
    fields::FieldStorage<double, 6, Exec, Layout> storage("grad_two", 4, fields::FieldLocation::Custom, exec);
    const auto values = storage.view();
    const auto first = gradient::cell_gradients(values, 0);
    const auto second = gradient::cell_gradients(values.as_const(), 1);
    first.template validate<Exec>(4);
    second.template validate<Exec>(4);
    equal(second.x.data, values.component(3).data(), "Component c uses axis 3*c");
    equal(second.x.stride, values.component(3).stride(0), "Native stride is preserved");
    const gradient::GradientCorrection correction{second};
    Kokkos::View<double*, Space> output("correction/layout", 4);
    auto host = Kokkos::create_mirror(output);
    complete(exec, [&] {
        Kokkos::parallel_for("test/correction_layout_fill", PolicySize(exec, 0, 4), KOKKOS_LAMBDA(std::size_t i) {
            for (int c = 0; c < 6; ++c) values(i, c) = 10.0*static_cast<double>(i)+static_cast<double>(c);
        });
        Kokkos::parallel_for("test/correction_layout", PolicySize(exec, 0, 4), KOKKOS_LAMBDA(std::size_t i) {
            output(i) = correction.increment(i, 1.0, 0.0, 0.0)-first(i)[0];
        });
        Kokkos::deep_copy(exec, host, output);
    });
    for (std::size_t i = 0; i < 4; ++i) equal(host(i), 3.0, "Both gradient layouts select the correct scalar");
}
void layouts(const Exec& exec) { layout_case<Kokkos::LayoutLeft>(exec); layout_case<Kokkos::LayoutRight>(exec); }

struct AnalyticGradients {
    KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(std::size_t cell) const
    { return {static_cast<double>(cell)+1.0, 2.0, -1.0}; }
};
void custom(const Exec& exec)
{
    const gradient::GradientCorrection correction{AnalyticGradients{}};
    const gradient::NoCorrection zero;
    Kokkos::View<double*, Space> output("correction/custom", 5);
    auto host = Kokkos::create_mirror(output);
    complete(exec, [&] {
        Kokkos::parallel_for("test/correction_custom", PolicySize(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            output(0) = correction.increment(0, 0.5, 2.0, 3.0);
            output(1) = correction.interior(0, 2, 0.75, 0.5, 2.0, 3.0);
            output(2) = zero.increment(0, 1.0, 2.0, 3.0);
            output(3) = zero(BoundaryContext{0, 1.0, 2.0, 3.0});
            output(4) = zero.interior(InteriorContext{0, 1, 0.25}, 1.0, 2.0, 3.0);
        });
        Kokkos::deep_copy(exec, host, output);
    });
    equal(host(0), 1.5, "Custom callable gradient provider");
    equal(host(1), 1.75, "Custom interpolated gradient provider");
    for (std::size_t i = 2; i < 5; ++i) equal(host(i), 0.0, "NoCorrection needs no gradient storage");
}

void extents(const Exec& exec)
{
    Kokkos::View<float*, Space> x("correction/x", 2);
    Kokkos::View<double*, Space> y("correction/y", 3), z("correction/z", 4);
    const auto gradients = gradient::cell_gradients(x, y, z);
    static_assert(std::is_same_v<typename decltype(gradients)::value_type, double>);
    equal(gradients.size(), std::size_t{2}, "Minimum component extent");
    gradients.template validate<Exec>(2);
    Kokkos::View<double*, Space> output("correction/mixed", 1);
    auto host = Kokkos::create_mirror(output);
    const gradient::GradientCorrection correction{gradients};
    complete(exec, [&] {
        Kokkos::deep_copy(exec, x, 1.5f);
        Kokkos::deep_copy(exec, y, 2.25);
        Kokkos::deep_copy(exec, z, -0.5);
        Kokkos::parallel_for("test/correction_mixed", PolicySize(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            output(0) = correction.increment(1, 2.0, 1.0, -1.0);
        });
        Kokkos::deep_copy(exec, host, output);
    });
    equal(host(0), 5.75, "Mixed floating-point component precision");
    Kokkos::View<float*, Space> empty;
    const auto zero = gradient::cell_gradients(empty, y, z);
    equal(zero.size(), std::size_t{0}, "An empty component produces an empty adapter");
    zero.template validate<Exec>(0);
}

void distributed(const Exec& exec)
{
    const auto comm = krona::mpi::CommView::world();
    equal(comm.size(), 2, "Correction halo case uses two ranks");
    const int rank = comm.rank();
    krona::mesh::host::MeshPart source;
    source.rank = rank; source.nprocs = 2; source.n_own = 1; source.n_cells = 2;
    source.nb_ranks = {1-rank}; source.send_offsets = {0, 1}; source.recv_offsets = {0, 1};
    source.send_owned_local = {0}; source.recv_ghost_local = {1};
    krona::mesh::device::MeshStorage<> mesh(source, {.data = krona::mesh::device::MeshData::Halo}, exec);
    fields::FieldStorage<double, 3, Exec> storage("grad_halo", 2, fields::FieldLocation::Custom, exec);
    const auto values = storage.view();
    const gradient::GradientCorrection correction{gradient::cell_gradients(values)};
    correction.gradients.template validate<Exec>(2);
    krona::halo::Exchange<double> exchange(mesh.view(), mesh.halo_plan(), comm, 949, exec);
    exchange.add_field(values); exchange.prepare();
    Kokkos::View<double*, Space> output("correction/halo", 1);
    auto host = Kokkos::create_mirror(output);
    complete(exec, [&] {
        Kokkos::parallel_for("test/correction_halo_fill", PolicySize(exec, 0, 2), KOKKOS_LAMBDA(std::size_t i) {
            for (int c = 0; c < 3; ++c) values(i, c) = i == 0 ? static_cast<double>(3*rank+c+1) : -999.0;
        });
        exchange.start(); exchange.finish_async();
        Kokkos::parallel_for("test/correction_halo_use", PolicySize(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            output(0) = correction.interior(0, 1, 0.25, 1.0, 0.5, -0.25);
        });
        Kokkos::deep_copy(exec, host, output);
    });
    exchange.wait();
    equal(host(0), rank == 0 ? 4.0625 : 2.1875, "Correction consumes actual exchanged ghost gradients");
}

void reject_short(const Exec&)
{
    Kokkos::View<double*, Space> data("correction/short", 1);
    gradient::cell_gradients(data, data, data).template validate<Exec>(2);
}
void reject_component(const Exec& exec)
{
    fields::FieldStorage<double, 3, Exec> storage("gradient", 1, fields::FieldLocation::Custom, exec);
    (void)gradient::cell_gradients(storage.view(), 1);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view{argv[1]}.starts_with("reject_")) {
        const std::array cases{Case{"reject_short", reject_short}, Case{"reject_component", reject_component}};
        return krona::test::run(argc, argv, "gradient.correction", cases);
    }
    const std::array cases{Case{"boundary", boundary}, Case{"interior", interior}, Case{"layouts", layouts},
        Case{"custom", custom}, Case{"extents", extents}, Case{"distributed", distributed}};
    return krona::test::run(argc, argv, "gradient.correction", cases);
}
