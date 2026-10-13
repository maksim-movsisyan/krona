#include "common/test_support.hpp"

#include <krona/boundary/scalar/config.hpp>
#include <krona/boundary/physical/fields.hpp>
#include <krona/boundary/physical/closures.hpp>
#include <krona/boundary/physical/corrections.hpp>
#include <krona/thermodynamics/ideal_gas.hpp>
#include <krona/boundary/physical/manager.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

namespace {
using namespace krona::test;
namespace bc = krona::boundary::physical;
using Calc = krona::ComputeReal;
using Space = typename Exec::memory_space;
using MeshPart = krona::mesh::MeshPart;
using MeshStorage = krona::mesh::compute::MeshStorage<Exec>;
using EOS = krona::thermodynamics::IdealGas<Calc>;
using Gas = bc::CompressibleClosure<EOS, bc::PerfectGasCharacteristics>;
using Buffer = Kokkos::View<Real**, Kokkos::LayoutRight, Space>;

template<class S> concept HasTemperature = requires(S state) { state.temperature; };
static_assert(!HasTemperature<bc::State<Calc, false>>);
static_assert(HasTemperature<bc::State<Calc, true>>);

/// @brief Compares analytical double references with compute/storage precision taken into account.
template<class T>
void expect_near(T actual, double expected, std::string_view message,
                 std::source_location where = std::source_location::current())
{
    const double epsilon = std::max(static_cast<double>(std::numeric_limits<T>::epsilon()),
                                   static_cast<double>(std::numeric_limits<Calc>::epsilon()));
    near(static_cast<double>(actual), expected, 4096 * epsilon * (1 + std::abs(expected)), message, where);
}

template<class View>
auto snapshot(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    complete(exec, [&] { Kokkos::deep_copy(exec, host, view); });
    return host;
}

bc::Configuration configuration()
{ return bc::read_config(KRONA_PHYSICAL_BOUNDARY_TEST_DATA); }

void physics(const Exec&)
{
    Gas gas;
    const double gamma = static_cast<double>(gas.eos.gamma), r = static_cast<double>(gas.eos.gas_constant);
    const double cp = gamma * r / (gamma - 1);
    bc::Context<Calc> c; c.normal = {1, 0, 0}; c.distance = Calc{0.5};
    const auto owner = gas.make_state(Calc{100000}, Calc{300}, {Calc{50}, Calc{10}, 0}).state;
    expect_near(owner.density, 100000 / (r * 300), "ideal-gas density");
    expect_near(gas.sound_speed(owner), std::sqrt(gamma * r * 300), "sound speed");

    const auto symmetry = bc::Symmetry{}(c, owner, gas);
    check(symmetry.valid(), "symmetry status");
    expect_near(symmetry.state.velocity.x, 0, "symmetry normal velocity");
    expect_near(symmetry.state.velocity.y, 10, "symmetry tangential velocity");
    const auto slip = bc::SlipWall<Calc>{{0, Calc{4}, 0}}(c, owner, gas);
    expect_near(slip.state.velocity.x, 0, "slip normal velocity");
    expect_near(slip.state.velocity.y, 10, "slip retains owner tangent");
    const auto wall = bc::IsothermalWall<Calc>{{0, Calc{4}, 0}, Calc{400}}(c, owner, gas);
    check(wall.valid(), "isothermal wall status");
    expect_near(wall.state.velocity.y, 4, "no-slip wall velocity");
    expect_near(wall.state.temperature, 400, "wall temperature");
    expect_near(wall.state.density, 100000 / (r * 400), "wall EOS density update");
    c.temperature_tangential_increment = Calc{7};
    expect_near(bc::AdiabaticWall<Calc>{}(c, owner, gas).state.temperature, 307, "nonorthogonal thermal correction");
    expect_near(bc::Symmetry{}(c, owner, gas).state.temperature, 307, "symmetry thermal correction");
    expect_near(bc::IsothermalWall<Calc>{{}, Calc{350}}(c, owner, gas).state.temperature,
         350, "fixed temperature ignores tangential correction");
    c.temperature_tangential_increment = 0;

    const auto out = bc::PressureOutlet<Calc>{Calc{105000}}(c, owner, gas);
    check(out.valid(), "subsonic pressure outlet");
    expect_near(out.state.pressure, 105000, "outlet static pressure");
    expect_near(out.state.temperature, 300 * std::pow(1.05, (gamma - 1) / gamma), "outlet isentropy");
    const double invariant = 50 + 2 * std::sqrt(gamma * r * 300) / (gamma - 1);
    expect_near(out.state.velocity.x + Calc{2} * gas.sound_speed(out.state) / (gas.eos.gamma - Calc{1}),
         invariant, "outlet outgoing characteristic");
    expect_near(out.state.velocity.y, 10, "outlet tangential velocity");

    const auto sup = gas.make_state(Calc{100000}, Calc{300}, {Calc{600}, 0, 0}).state;
    const auto sup_out = bc::SupersonicOutlet{}(c, sup, gas);
    check(sup_out.valid(), "supersonic outlet status");
    expect_near(bc::PressureOutlet<Calc>{Calc{90000}}(c, sup, gas).state.pressure, 100000,
         "pressure ignored on supersonic outlet");
    c.normal = {-1, 0, 0};
    check(bc::SupersonicInlet<Calc>{Calc{100000}, Calc{300}, {Calc{600}, 0, 0}}(c, owner, gas).valid(),
          "supersonic inlet state");
    check(bc::SupersonicInlet<Calc>{Calc{100000}, Calc{300}, {Calc{50}, 0, 0}}(c, owner, gas).status ==
          bc::Status::RegimeMismatch, "reject subsonic specified supersonic inlet");
    const auto in = bc::VelocityInlet<Calc>{{Calc{60}, 0, 0},
        {bc::ThermalKind::FixedTemperature, Calc{310}}}(c, owner, gas);
    check(in.valid(), "velocity-temperature inlet");
    expect_near(in.state.pressure, 100000 + static_cast<double>(owner.density) * std::sqrt(gamma * r * 300) * 10,
         "inlet outgoing linearized acoustic compatibility");

    // Independent stagnation data manufactured from a known static state and velocity.
    for (const bc::Vector3<Calc> direction : {bc::Vector3<Calc>{1, 0, 0},
                                             bc::Vector3<Calc>{static_cast<Calc>(0.6), static_cast<Calc>(0.8), 0}}) {
        const auto known = gas.make_state(Calc{100000}, Calc{300}, direction * Calc{100}).state;
        const double t0 = 300 + 10000 / (2 * cp);
        const double p0 = 100000 * std::pow(t0 / 300, gamma / (gamma - 1));
        const auto result = bc::TotalInlet<Calc>{static_cast<Calc>(p0), static_cast<Calc>(t0), direction}(c, known, gas);
        check(result.valid(), "stagnation inlet quadratic");
        expect_near(result.state.temperature, 300, "recovered static temperature");
        expect_near(result.state.pressure, 100000, "recovered static pressure");
        expect_near(result.state.velocity.x, 100 * static_cast<double>(direction.x), "recovered oblique inlet velocity");
        expect_near(result.state.velocity.y, 100 * static_cast<double>(direction.y), "recovered inlet tangent");
    }
    const auto still = gas.make_state(Calc{100000}, Calc{300}, {}).state;
    check(bc::TotalInlet<Calc>{Calc{100000}, Calc{300}, {1, 0, 0}}(c, still, gas).valid(),
          "zero-speed stagnation roundoff");
    check(bc::TotalInlet<Calc>{Calc{100000}, Calc{10}, {1, 0, 0}}(c, still, gas).status ==
          bc::Status::NoSubsonicSolution, "incompatible total temperature");

    c.normal = {1, 0, 0};
    auto back = owner; back.velocity.x = Calc{-5};
    check(bc::PressureOutlet<Calc>{Calc{100000}}(c, back, gas).status == bc::Status::Backflow,
          "explicit backflow diagnostic");
    const auto fallback = bc::InletOutlet<Calc>{Calc{100000}, {Calc{-10}, 0, 0},
        {bc::ThermalKind::FixedTemperature, Calc{320}}}(c, back, gas);
    check(fallback.valid(), "specified backflow state");
    expect_near(fallback.state.velocity.x, -10, "backflow velocity");
    expect_near(fallback.state.temperature, 320, "backflow temperature");

    bc::IncompressibleClosure<Calc> liquid{Calc{1000}};
    bc::State<Calc, false> q{Calc{1000}, Calc{-20}, {Calc{5}, Calc{2}, 0}};
    check(liquid.valid(q), "incompressible gauge pressure may be negative");
    expect_near(bc::AdiabaticWall<Calc>{}(c, q, liquid).state.velocity.x, 0, "athermal no-slip");
    check(bc::TotalInlet<Calc>{}(c, q, liquid).status == bc::Status::UnsupportedCondition,
          "stagnation inlet unsupported without acoustic model");
    check(bc::IsothermalWall<Calc>{{}, Calc{300}}(c, q, liquid).status ==
          bc::Status::UnsupportedCondition, "athermal model rejects temperature");
    expect_near(bc::PressureOutlet<Calc>{Calc{-10}}(c, q, liquid).state.pressure, -10,
         "incompressible prescribed gauge pressure");
    bc::IncompressibleClosure<Calc, true> thermal_liquid{Calc{1000}};
    const auto tq = thermal_liquid.make_state(Calc{-20}, Calc{300}, q.velocity).state;
    expect_near(bc::IsothermalWall<Calc>{{}, Calc{350}}(c, tq, thermal_liquid).state.temperature,
         350, "thermal incompressible model");
}

/// @brief One owned cell, one MPI ghost, and three physical patches.
MeshPart fixture()
{
    MeshPart m;
    m.n_own = 1; m.n_cells = 2; m.n_faces = 4; m.n_inner_faces = 1; m.n_internal_faces = 0;
    m.faces_grouped = true;
    m.n_bfaces_g = m.n_labeled_bfaces_g = 3;
    m.face_owner = {0, 0, 0, 0}; m.face_neigh = {1, -1, -1, -1}; m.face_patch = {-1, 0, 1, 2};
    m.patches.resize(3); m.patches[0].name = "inlet"; m.patches[1].name = "wall"; m.patches[2].name = "outlet";
    m.cell_centroid_x = {0, 2}; m.cell_centroid_y = {0, 0}; m.cell_centroid_z = {0, 0}; m.cell_volume = {1, 1};
    m.face_centroid_x = {1, -1, 0, 1}; m.face_centroid_y = {0, 0, 1, 0}; m.face_centroid_z = {0, 0, 0, 0};
    m.face_normal_x = {1, -1, 0, 1}; m.face_normal_y = {0, 0, 1, 0}; m.face_normal_z = {0, 0, 0, 0};
    m.face_area = {1, 1, 1, 1};
    return m;
}

struct Profile {
    template<class Model>
    KOKKOS_INLINE_FUNCTION bc::Evaluation<typename Model::state_type>
    operator()(const bc::Context<Calc>& c, const typename Model::state_type& owner, const Model& model) const
    { return bc::VelocityInlet<Calc>{{Calc{10} + c.time, 0, 0}}(c, owner, model); }
};

struct CountFaces {
    Kokkos::View<int*, Space> calls;
    Calc expected_increment = 0;
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<Calc>& c, const Condition&) const
    { calls(static_cast<std::size_t>(c.face)) = c.temperature_tangential_increment == expected_increment ? 1 : 2; }
};
struct AddTemperatureCorrection {
    Calc value = 0;
    KOKKOS_INLINE_FUNCTION bc::Context<Calc> operator()(bc::Context<Calc> c) const
    { c.temperature_tangential_increment = value; return c; }
};
struct CaptureStatuses {
    Kokkos::View<int*, Space> statuses;
    template<class S, class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<Calc>& c, const S&,
                                           const bc::Evaluation<S>& result, const Condition&) const
    { statuses(static_cast<std::size_t>(c.face)) = static_cast<int>(result.status); }
};
struct RecordGeometry {
    Kokkos::View<krona::GeometryReal*, Space> distance, position;
    template<class R, class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<R>& c, const Condition&) const
    { distance(static_cast<std::size_t>(c.face)) = c.distance; position(static_cast<std::size_t>(c.face)) = c.position.x; }
};

template<class View>
auto fields_from(View buffer)
{
    return bc::make_fields<Calc>(Kokkos::subview(buffer, Kokkos::ALL(), 0),
        Kokkos::subview(buffer, Kokkos::ALL(), 1), Kokkos::subview(buffer, Kokkos::ALL(), 2),
        Kokkos::subview(buffer, Kokkos::ALL(), 3), Kokkos::subview(buffer, Kokkos::ALL(), 4),
        Kokkos::subview(buffer, Kokkos::ALL(), 5));
}

template<class View>
void initialize(const Exec& exec, View buffer, const Gas& gas)
{
    const auto fields = fields_from(buffer);
    const auto owner = gas.make_state(Calc{100000}, Calc{300}, {Calc{20}, 0, 0}).state;
    complete(exec, [&] {
        Kokkos::deep_copy(exec, buffer, typename View::non_const_value_type{-999});
        Kokkos::parallel_for("test/physical/initialize", Policy(exec, 0, 2),
            KOKKOS_LAMBDA(Index i) { fields.store(static_cast<std::size_t>(i), owner); });
    });
}

void config(const Exec& exec)
{
    std::ifstream file{KRONA_PHYSICAL_BOUNDARY_TEST_DATA};
    check(file.is_open(), "Boundary configuration file is readable");
    const std::string text{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    const auto temperature = krona::boundary::scalar::parse_config(text, "temperature");
    const auto tracer = krona::boundary::scalar::parse_config(text, "tracer");
    equal(temperature.patches[1].condition.parameter, 350.0, "Scalar manager reads its own catalogue from the shared file");
    equal(tracer.patches[0].condition.parameter, 0.25, "A second scalar catalogue remains independent");
    auto parsed = bc::parse_config(text);
    check(!parsed.unlabelled.has_value(), "Scalar unlabelled table does not configure physical boundaries");
    equal(parsed.patches.size(), std::size_t{3}, "Three physical patches are parsed");
    check(parsed.patches[0].condition.kind == bc::Kind::VelocityInlet, "Velocity inlet type");
    equal(parsed.patches[0].condition.velocity[0], 20.0, "Inlet velocity parameter");
    check(parsed.patches[1].condition.temperature.has_value(), "Wall temperature is present");
    equal(*parsed.patches[1].condition.temperature, 350.0, "Wall temperature parameter");
    check(parsed.patches[2].condition.pressure.has_value(), "Outlet pressure is present");
    equal(*parsed.patches[2].condition.pressure, 100000.0, "Outlet pressure parameter");
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    std::swap(parsed.patches.front(), parsed.patches.back());
    bc::Manager<Exec, Calc> manager{m, mesh.view(), parsed, exec};
    equal(manager.catalog().patches()[0].range.begin, Index{1}, "Patch binding uses ID rather than TOML order");
    equal(manager.catalog().patches()[0].range.end, Index{2}, "Absolute patch end");
    check(manager.catalog().unlabelled().empty(), "MPI interface is not an unlabelled physical boundary");
    equal(manager.required_field_size(), std::size_t{5}, "Boundary storage follows owned and MPI ghost cells");
}

/// @brief Both layouts must preserve cells and unused storage while filling face-centroid values.
template<class Layout>
void layout(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    Gas gas;
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Kokkos::View<Real**, Layout, Space> buffer{"primitive fields", manager.required_field_size() + 2, 6};
    initialize(exec, buffer, gas);
    const auto fields = fields_from(buffer);
    const auto before = snapshot(exec, buffer);
    complete(exec, [&] { manager.apply_async(fields, gas); });
    const auto after = snapshot(exec, buffer);
    for (std::size_t i : {std::size_t{0}, std::size_t{1}, std::size_t{5}, std::size_t{6}})
        for (std::size_t j = 0; j < 6; ++j)
            equal(after(i, j), before(i, j), "Owned/MPI ghost/unused tail unchanged");
    expect_near(after(2, 1), 20, "Boundary slot starts at n_cells");
    expect_near(after(3, 1), 0, "Wall face velocity");
    expect_near(after(3, 5), 350, "Wall face temperature");
    expect_near(after(3, 0), 100000 / (static_cast<double>(gas.eos.gas_constant) * 350), "Wall density recomputed");
    expect_near(after(4, 4), 100000, "Outlet face pressure");
}

void fill(const Exec& exec) { layout<Kokkos::LayoutRight>(exec); }
void layouts(const Exec& exec) { layout<Kokkos::LayoutLeft>(exec); layout<Kokkos::LayoutRight>(exec); }

void dispatch(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    int patches = 0;
    manager.for_each_patch([&](krona::boundary::PatchRange range, const auto&) {
        check(!range.empty(), "Nonempty patch"); ++patches;
    });
    equal(patches, 3, "Host patch dispatch");
    bool visited = false;
    manager.visit(1, [&](krona::boundary::PatchRange range, const auto&) {
        visited = true; equal(range.begin, Index{2}, "Single patch visit range");
    });
    check(visited, "Single patch visit");
    Kokkos::View<int*, Space> calls{"face callback counts", 4};
    complete(exec, [&] {
        Kokkos::deep_copy(exec, calls, 0);
        manager.for_each_face_async(CountFaces{calls, Calc{7}}, Calc{0}, AddTemperatureCorrection{Calc{7}});
    });
    const auto counts = snapshot(exec, calls);
    equal(counts(0), 0, "MPI interface is not submitted");
    for (std::size_t f = 1; f < 4; ++f) equal(counts(f), 1, "Boundary-only device callback with transformed context");
}

void profile(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    Gas gas;
    bc::Manager<Exec, Calc, Profile> manager{m, mesh.view(), configuration(), exec};
    Buffer buffer{"profile fields", manager.required_field_size(), 6};
    initialize(exec, buffer, gas);
    const auto fields = fields_from(buffer);
    // Parameters are copied into submitted kernels and may change between submissions.
    complete(exec, [&] { manager.apply_async(fields, gas); });
    manager.set(0, Profile{});
    complete(exec, [&] { manager.apply_async(fields, gas, Calc{2}); });
    const auto result = snapshot(exec, buffer);
    expect_near(result(2, 1), 12, "Custom time-dependent condition");
}

void statuses(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    Gas gas;
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Buffer buffer{"status fields", manager.required_field_size(), 6};
    initialize(exec, buffer, gas);
    const auto fields = fields_from(buffer);
    manager.set(0, bc::Extrapolate{});
    const auto owner = gas.make_state(Calc{100000}, Calc{300}, {Calc{-5}, 0, 0}).state;
    Kokkos::View<int*, Space> results{"statuses", 4};
    complete(exec, [&] {
        Kokkos::parallel_for("test/physical/reverse_flow", Policy(exec, 0, 1),
            KOKKOS_LAMBDA(Index) { fields.store(0, owner); });
        Kokkos::deep_copy(exec, results, -1);
        manager.evaluate_async(fields, gas, CaptureStatuses{results});
    });
    const auto status = snapshot(exec, results);
    equal(status(0), -1, "Internal face status untouched");
    equal(status(1), static_cast<int>(bc::Status::Success), "Extrapolation status");
    equal(status(2), static_cast<int>(bc::Status::Success), "Wall status");
    equal(status(3), static_cast<int>(bc::Status::Backflow), "Failed evaluation returned to device sink");
}

void incompressible(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    Gas gas;
    Buffer buffer{"athermal fields", 5, 6};
    initialize(exec, buffer, gas);
    bc::Manager<Exec, Calc> thermal{m, mesh.view(), configuration(), exec};
    complete(exec, [&] { thermal.apply_async(fields_from(buffer), gas); });
    bc::IncompressibleClosure<Calc> liquid{Calc{1000}};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), exec};
    manager.set(0, bc::VelocityInlet<Calc>{{Calc{20}, 0, 0}});
    manager.set(1, bc::AdiabaticWall<Calc>{});
    manager.set(2, bc::PressureOutlet<Calc>{Calc{-10}});
    const auto fields = bc::make_incompressible_fields<Calc>(liquid.density,
        Kokkos::subview(buffer, Kokkos::ALL(), 1), Kokkos::subview(buffer, Kokkos::ALL(), 2),
        Kokkos::subview(buffer, Kokkos::ALL(), 3), Kokkos::subview(buffer, Kokkos::ALL(), 4));
    static_assert(!decltype(fields)::has_temperature);
    complete(exec, [&] {
        Kokkos::parallel_for("test/physical/athermal_owner", Policy(exec, 0, 1),
            KOKKOS_LAMBDA(Index) { fields.store(0, bc::State<Calc, false>{Calc{1000}, Calc{-20}, {Calc{20}, 0, 0}}); });
        manager.apply_async(fields, liquid);
    });
    const auto result = snapshot(exec, buffer);
    expect_near(result(4, 4), -10, "Athermal gauge pressure outlet");
    expect_near(result(3, 1), 0, "Athermal wall kernel");
    expect_near(result(3, 5), 350, "Athermal adapter does not write temperature");
}

void unlabelled(const Exec& exec)
{
    auto m = fixture();
    m.face_patch[3] = -1;
    m.n_labeled_bfaces_g = 2;
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), exec};
    manager.set(0, bc::Extrapolate{}); manager.set(1, bc::Symmetry{}); manager.set(2, bc::Extrapolate{});
    manager.set_unlabelled(bc::Extrapolate{});
    int count = 0;
    manager.for_each_patch([&](krona::boundary::PatchRange, const auto&) { ++count; });
    equal(count, 3, "Nonempty unlabelled interval replaces empty local patch");
    bool labelled_visited = false, unlabelled_visited = false;
    manager.visit(2, [&](krona::boundary::PatchRange range, const auto&) {
        labelled_visited = true; check(range.empty(), "Empty labelled visit");
    });
    manager.visit_unlabelled([&](krona::boundary::PatchRange range, const auto&) {
        unlabelled_visited = true; equal(range.begin, Index{3}, "Unlabelled visit");
    });
    check(labelled_visited && unlabelled_visited, "Both visit callbacks execute");
}

void empty(const Exec& exec)
{
    auto m = fixture();
    m.n_own = m.n_cells = m.n_faces = m.n_inner_faces = m.n_internal_faces = 0;
    m.face_owner.clear(); m.face_neigh.clear(); m.face_patch.clear();
    m.cell_centroid_x.clear(); m.cell_centroid_y.clear(); m.cell_centroid_z.clear(); m.cell_volume.clear();
    m.face_centroid_x.clear(); m.face_centroid_y.clear(); m.face_centroid_z.clear();
    m.face_normal_x.clear(); m.face_normal_y.clear(); m.face_normal_z.clear(); m.face_area.clear();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Buffer buffer{"empty fields", 0, 6};
    complete(exec, [&] { manager.apply_async(fields_from(buffer), Gas{}); });
    equal(manager.required_field_size(), std::size_t{0}, "Empty partition has no boundary storage");
    int calls = 0;
    manager.for_each_patch([&](krona::boundary::PatchRange, const auto&) { ++calls; });
    equal(calls, 0, "Empty local patches do not launch work");

    MeshPart no_patches;
    no_patches.faces_grouped = true;
    MeshStorage no_patch_mesh{no_patches, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> no_patch_manager{no_patches, no_patch_mesh.view(), exec};
    Kokkos::View<Real*, Space> none;
    const auto fields = bc::make_incompressible_fields<Calc>(Calc{1}, none, none, none, none);
    complete(exec, [&] { no_patch_manager.apply_async(fields, bc::IncompressibleClosure<Calc>{}); });
}

void mixed_storage(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    Gas gas;
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Kokkos::View<float**, Kokkos::LayoutRight, Space> buffer{"float storage", 5, 6};
    initialize(exec, buffer, gas);
    complete(exec, [&] { manager.apply_async(fields_from(buffer), gas); });
    const auto result = snapshot(exec, buffer);
    const double expected = 100000 / (static_cast<double>(gas.eos.gas_constant) * 350);
    const double epsilon = std::max(static_cast<double>(std::numeric_limits<float>::epsilon()),
                                   static_cast<double>(std::numeric_limits<Calc>::epsilon()));
    near(static_cast<double>(result(3, 0)), expected, 64 * epsilon * (1 + expected), "Mixed field storage precision");
    equal(result(3, 5), 350.0f, "Mixed storage wall temperature");
    equal(result(3, 1), 0.0f, "Mixed storage wall velocity");
}

void large_offset_geometry(const Exec& exec)
{
    if constexpr (std::is_same_v<krona::GeometryReal, double>) {
        auto m = fixture();
        constexpr double offset = 1099511627776.0; // 2^40: unit distances survive in double, not float.
        for (auto& x : m.cell_centroid_x) x = static_cast<krona::GeometryReal>(static_cast<double>(x) + offset);
        for (auto& x : m.face_centroid_x) x = static_cast<krona::GeometryReal>(static_cast<double>(x) + offset);
        MeshStorage mesh{m, {.data = bc::requirements()}, exec};
        bc::Manager<Exec, float> manager{m, mesh.view(), exec};
        manager.set(0, bc::Extrapolate{}); manager.set(1, bc::Symmetry{}); manager.set(2, bc::Extrapolate{});
        Kokkos::View<krona::GeometryReal*, Space> distance{"geometry distance", 4}, position{"geometry position", 4};
        complete(exec, [&] { manager.for_each_face_async(RecordGeometry{distance, position}); });
        const auto d = snapshot(exec, distance), x = snapshot(exec, position);
        for (std::size_t f = 1; f < 4; ++f) equal(d(f), 1.0, "Geometry precision independent of gas precision");
        equal(x(1), offset - 1, "Inlet position retains GeometryReal precision");
        equal(x(3), offset + 1, "Outlet position retains GeometryReal precision");
    }
}

void thermodynamics(const Exec&)
{
    EOS eos;
    check(eos.valid_parameters(), "Ideal gas parameter domain");
    const Calc rho = eos.density(Calc{100000}, Calc{300});
    expect_near(eos.pressure(rho, Calc{300}), 100000, "EOS pressure/density inverse");
    expect_near(eos.temperature_from_energy(eos.internal_energy(Calc{300})), 300, "Internal-energy inverse");
    expect_near(eos.temperature_from_enthalpy(eos.enthalpy(Calc{300})), 300, "Enthalpy inverse");
    expect_near(eos.cp() - eos.cv(), static_cast<double>(eos.gas_constant), "Perfect-gas caloric identity");
}

// Architectural test double: no gamma, gas_constant, cp or boundary methods.
struct AlternativeEOS {
    using real_type = Calc;
    KOKKOS_INLINE_FUNCTION bool valid(Calc p, Calc t) const
    { return Kokkos::isfinite(p) && p > 0 && Kokkos::isfinite(t) && t > 0; }
    KOKKOS_INLINE_FUNCTION Calc density(Calc p, Calc t) const { return p / (Calc{200} * t) + Calc{1}; }
    KOKKOS_INLINE_FUNCTION Calc sound_speed(Calc rho, Calc) const { return Kokkos::sqrt(Calc{100000} / rho); }
};

void generic_eos(const Exec& exec)
{
    bc::CompressibleClosure<AlternativeEOS> closure;
    const auto owner = closure.make_state(Calc{100000}, Calc{300}, {Calc{20}, 0, 0}).state;
    bc::Context<Calc> c; c.normal = {1, 0, 0};
    const auto out = bc::PressureOutlet<Calc>{Calc{90000}}(c, owner, closure);
    check(out.valid(), "Generic prescribed outlet accepts an independent thermodynamic API");
    expect_near(out.state.pressure, 90000, "Generic outlet pressure");
    expect_near(out.state.temperature, 300, "Prescribed policy extrapolates temperature");
    expect_near(out.state.density, 2.5, "Generic outlet closes density through the supplied EOS");
    const auto wall = bc::IsothermalWall<Calc>{{}, Calc{400}}(c, owner, closure);
    check(wall.valid(), "Generic EOS is also used for wall completion");
    expect_near(wall.state.density, 2.25, "Generic isothermal wall density");
    check(bc::TotalInlet<Calc>{Calc{110000}, Calc{310}, {-1, 0, 0}}(c, owner, closure).status ==
          bc::Status::UnsupportedCondition, "Stagnation closure is selected explicitly");
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Buffer buffer{"alternative EOS fields", manager.required_field_size(), 6};
    const auto fields = fields_from(buffer);
    complete(exec, [&] {
        Kokkos::deep_copy(exec, buffer, Real{-999});
        Kokkos::parallel_for("test/physical/alternative_eos", Policy(exec, 0, 2),
            KOKKOS_LAMBDA(Index i) { fields.store(static_cast<std::size_t>(i), owner); });
        manager.apply_async(fields, closure);
    });
    const auto result = snapshot(exec, buffer);
    expect_near(result(3, 0), 100000.0 / (200 * 350) + 1, "Manager uses the independent EOS in device kernels");

}

void thermal_correction(const Exec& exec)
{
    auto m = fixture();
    m.face_centroid_x[2] = static_cast<krona::GeometryReal>(0.2);
    m.face_centroid_z[2] = static_cast<krona::GeometryReal>(-0.1);
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), exec};
    manager.set(0, bc::Extrapolate{}); manager.set(1, bc::AdiabaticWall<Calc>{}); manager.set(2, bc::Extrapolate{});
    Gas gas;
    Buffer buffer{"corrected fields", manager.required_field_size(), 6};
    initialize(exec, buffer, gas);
    Kokkos::View<Calc*, Space> gx{"gx", 1}, gy{"gy", 1}, gz{"gz", 1};
    const auto correction = bc::TemperatureGradientCorrection{gx, gy, gz};
    complete(exec, [&] {
        Kokkos::deep_copy(exec, gx, Calc{2}); Kokkos::deep_copy(exec, gy, Calc{0}); Kokkos::deep_copy(exec, gz, Calc{-4});
        manager.apply_async(fields_from(buffer), gas, Calc{0}, correction);
    });
    const auto corrected = snapshot(exec, buffer);
    expect_near(corrected(3, 5), 300.8, "Adiabatic correction uses the tangential displacement");
    manager.set(1, bc::IsothermalWall<Calc>{{}, Calc{350}});
    complete(exec, [&] { manager.apply_async(fields_from(buffer), gas, Calc{0}, correction); });
    const auto fixed = snapshot(exec, buffer);
    expect_near(fixed(3, 5), 350, "Isothermal wall ignores tangential correction");
}

struct AssembleWallDiagonal {
    Kokkos::View<Calc*, Space> diagonal;
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<Calc>& c, const Condition&) const
    {
        if constexpr (std::is_same_v<Condition, bc::IsothermalWall<Calc>>)
            Kokkos::atomic_add(&diagonal(static_cast<std::size_t>(c.owner)),
                              static_cast<Calc>(c.area / c.distance));
    }
};

void assembly(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Kokkos::View<Calc*, Space> diagonal{"diagonal", 2};
    complete(exec, [&] {
        Kokkos::deep_copy(exec, diagonal, Calc{0});
        manager.for_each_face_async(AssembleWallDiagonal{diagonal});
    });
    const auto result = snapshot(exec, diagonal);
    expect_near(result(0), 1, "Custom typed face kernel assembles an owned-cell coefficient without a flow model");
    expect_near(result(1), 0, "Custom assembly leaves MPI ghost rows untouched");
}

void distributed(const Exec& exec)
{
    int rank = 0, ranks = 0;
    krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    equal(ranks, 2, "Distributed boundary test requires two ranks");
    if (rank == 0) fill(exec); else empty(exec);
    int local = 1, total = 0;
    krona::mpi::check(MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce");
    equal(total, 2, "Rank with no local boundary faces participates normally");
}

template<class F>
void reject_with_manager(const Exec& exec, F mutation)
{
    auto m = fixture();
    auto config = configuration();
    Gas gas;
    mutation(m, config, gas);
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), config, exec};
    Buffer buffer{"rejection fields", manager.required_field_size(), 6};
    initialize(exec, buffer, gas);
    complete(exec, [&] { manager.apply_async(fields_from(buffer), gas); });
    check(false, "Invalid manager setup was accepted");
}

void reject_wrong_name(const Exec& exec)
{ reject_with_manager(exec, [](auto&, auto& c, auto&) { c.patches[1].name = "not_wall"; }); }
void reject_duplicate(const Exec& exec)
{ reject_with_manager(exec, [](auto&, auto& c, auto&) { c.patches[1].patch_id = 0; }); }
void reject_invalid_geometry(const Exec& exec)
{ reject_with_manager(exec, [](auto& m, auto&, auto&) { m.face_normal_x[1] = -2; }); }
void reject_invalid_model(const Exec& exec)
{ reject_with_manager(exec, [](auto&, auto&, auto& gas) { gas.eos.gamma = Calc{1}; }); }
void reject_missing(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), exec};
    manager.for_each_patch([](krona::boundary::PatchRange, const auto&) {});
    check(false, "Missing condition was accepted");
}
void reject_extent(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    Buffer buffer{"short fields", 1, 6};
    complete(exec, [&] { manager.apply_async(fields_from(buffer), Gas{}); });
    check(false, "Short field extent was accepted");
}
void reject_temperature_without_energy(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), exec};
    manager.set(0, bc::Extrapolate{}); manager.set(2, bc::Extrapolate{});
    manager.set(1, bc::IsothermalWall<Calc>{{}, Calc{300}});
    Buffer buffer{"athermal rejected fields", manager.required_field_size(), 6};
    const auto fields = bc::make_incompressible_fields<Calc>(Calc{1000},
        Kokkos::subview(buffer, Kokkos::ALL(), 1), Kokkos::subview(buffer, Kokkos::ALL(), 2),
        Kokkos::subview(buffer, Kokkos::ALL(), 3), Kokkos::subview(buffer, Kokkos::ALL(), 4));
    complete(exec, [&] { manager.apply_async(fields, bc::IncompressibleClosure<Calc>{Calc{1000}}); });
    check(false, "Temperature for an athermal model was accepted");
}
void reject_typo(const Exec&)
{
    (void)bc::parse_config("boundary_condition=[]\n[unlabelled]\ntype='SYMMETRY'\n[unlabelled.physical]\ntemperature=300\n");
    check(false, "Inapplicable TOML parameter was accepted");
}
void reject_malformed_toml(const Exec&)
{
    (void)bc::parse_config("[boundary_condition\n");
    check(false, "Malformed TOML was accepted");
}

void reject_unknown_type(const Exec&)
{
    (void)bc::parse_config("[[boundary_condition]]\npatch_id=0\nname='inlet'\ntype='TYPO'\n");
    check(false, "Unknown physical type was accepted");
}
void reject_parser_duplicate(const Exec&)
{
    const std::string entry = "[[boundary_condition]]\npatch_id=0\nname='inlet'\ntype='SYMMETRY'\n";
    (void)bc::parse_config(entry + entry);
    check(false, "Duplicate TOML patch_id was accepted");
}
void reject_unlabelled(const Exec& exec)
{
    auto m = fixture(); m.face_patch[3] = -1; m.n_labeled_bfaces_g = 2;
    MeshStorage mesh{m, {.data = bc::requirements()}, exec};
    bc::Manager<Exec, Calc> manager{m, mesh.view(), configuration(), exec};
    check(false, "Unconfigured physical unlabelled faces were accepted");
}

constexpr std::array cases{
    Case{"thermodynamics", thermodynamics}, Case{"generic_eos", generic_eos},
    Case{"thermal_correction", thermal_correction}, Case{"assembly", assembly},
    Case{"physics", physics}, Case{"config", config}, Case{"fill", fill}, Case{"layouts", layouts},
    Case{"dispatch", dispatch}, Case{"profile", profile}, Case{"statuses", statuses},
    Case{"incompressible", incompressible}, Case{"unlabelled", unlabelled}, Case{"empty", empty},
    Case{"mixed_storage", mixed_storage}, Case{"large_offset_geometry", large_offset_geometry},
    Case{"distributed", distributed},
    Case{"reject_missing", reject_missing}, Case{"reject_wrong_name", reject_wrong_name},
    Case{"reject_duplicate", reject_duplicate}, Case{"reject_invalid_geometry", reject_invalid_geometry},
    Case{"reject_invalid_model", reject_invalid_model}, Case{"reject_extent", reject_extent},
    Case{"reject_temperature_without_energy", reject_temperature_without_energy},
    Case{"reject_unknown_type", reject_unknown_type}, Case{"reject_parser_duplicate", reject_parser_duplicate},
    Case{"reject_unlabelled", reject_unlabelled},
    Case{"reject_typo", reject_typo}, Case{"reject_malformed_toml", reject_malformed_toml}};
} // namespace

int main(int argc, char** argv)
{ return krona::test::run(argc, argv, "boundary.physical", cases); }
