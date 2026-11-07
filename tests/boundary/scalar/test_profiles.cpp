#include "common/test_support.hpp"

#include <krona/boundary/scalar/manager.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/transport/diffusion/isotropic.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace {
using namespace krona::test;
namespace bc = krona::boundary;
namespace sc = bc::scalar;
namespace pr = bc::profile;
namespace grad = krona::numerics::gradient;
using Space = Exec::memory_space;
using Array = Kokkos::View<double*, Space>;

/// @brief Returns a completed independent host snapshot on the supplied execution instance.
template<class View>
auto snapshot(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view); exec.fence(); return host;
}


/// @brief Builds two owners with labelled, empty and unlabelled boundary ranges.
krona::mesh::host::MeshPart fixture()
{
    krona::mesh::host::MeshPart m;
    m.n_own = m.n_cells = 2;
    m.n_faces = 4; m.n_internal_faces = m.n_inner_faces = 0; m.faces_grouped = true;
    m.n_bfaces_g = 4; m.n_labeled_bfaces_g = 3;
    m.patches = {{"first", "", ""}, {"second", "", ""}, {"empty", "", ""}};
    m.face_owner = {0, 1, 0, 1}; m.face_neigh = {-1, -1, -1, -1}; m.face_patch = {0, 0, 1, -1};
    m.cell_centroid_x = {0, 2}; m.cell_centroid_y = m.cell_centroid_z = {0, 0}; m.cell_volume = {1, 1};
    m.face_centroid_x = {1, 3, 2, 4}; m.face_centroid_y = {0.5, -0.5, 0.25, 0.75};
    m.face_centroid_z = {0, 0, 0, 0}; m.face_normal_x = {1, 1, 1, 1};
    m.face_normal_y = m.face_normal_z = {0, 0, 0, 0}; m.face_area = {2, 3, 4, 5};
    return m;
}

/// @brief Supplies an analytic value depending on face position and evaluation time.
template<class R>
struct SpaceTime {
    R base = 0;
    KOKKOS_INLINE_FUNCTION R operator()(const bc::FaceContext<R>& c) const
    { return base+R{2}*c.x+R{3}*c.time; }
};

/// @brief Converts outward flux into normal gradient using a known coefficient of two.
struct Law {
    template<class R>
    KOKKOS_INLINE_FUNCTION R normal_gradient(const bc::FaceContext<R>&, R flux) const
    { return -flux/R{2}; }
};

/// @brief Stores all affine relation coefficients for independent comparison on the host.
struct Capture {
    Kokkos::View<double*[4], Space> relations;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<double>& c, const sc::Relation<double>& r) const
    {
        relations(c.boundary_index, 0) = r.value_owner;
        relations(c.boundary_index, 1) = r.value_constant;
        relations(c.boundary_index, 2) = r.gradient_owner;
        relations(c.boundary_index, 3) = r.gradient_constant;
    }
};

/// @brief Checks analytic profile values converted to affine value/gradient relations.
template<class R>
void direct_relations(const Exec& exec)
{
    Kokkos::View<R*[6], Space> output("direct", 1);
    Kokkos::parallel_for("test/profile_relations", Kokkos::RangePolicy<Exec>(exec, 0, 1), KOKKOS_LAMBDA(int) {
        bc::FaceContext<R> c; c.x = 1; c.time = 2; c.distance = 2;
        const auto d = sc::make_relation(sc::fixed_value<R>(SpaceTime<R>{10}), c, R{3});
        const auto n = sc::make_relation(sc::normal_gradient<R>(SpaceTime<R>{10}), c, R{3});
        const auto q = sc::make_relation(sc::fixed_flux<R>(SpaceTime<R>{10}), c, R{3}, Law{});
        output(0, 0) = d.face_value(20); output(0, 1) = d.normal_gradient(20);
        output(0, 2) = n.face_value(20); output(0, 3) = n.normal_gradient(20);
        output(0, 4) = q.face_value(20); output(0, 5) = q.normal_gradient(20);
    });
    const auto h = snapshot(exec, output);
    const R expected[]{18, R{-2.5}, 59, 18, 5, -9};
    for (std::size_t i = 0; i < 6; ++i) near(h(0, i), expected[i], static_cast<R>(1e-5), "Profiled affine relation");
}

/// @brief Exercises profiled relation factories with float and double arithmetic.
void relations_case(const Exec& exec) { direct_relations<float>(exec); direct_relations<double>(exec); }

/// @brief Checks space/time profiles, tangential corrections and independent manager/status lifecycles.
void analytic_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    using Value = decltype(sc::fixed_value<double>(SpaceTime<double>{}));
    using Gradient = decltype(sc::normal_gradient<double>(SpaceTime<double>{}));
    using Flux = decltype(sc::fixed_flux<double>(SpaceTime<double>{}));
    sc::Manager<Exec, double, Value, Gradient, Flux> manager(layout);
    manager.set(0, sc::fixed_value<double>(SpaceTime<double>{10}));
    manager.set(1, sc::normal_gradient<double>(SpaceTime<double>{10}));
    manager.set(2, sc::NormalGradient<double>{0});
    manager.set_unlabelled(sc::fixed_flux<double>(SpaceTime<double>{10}));
    check(!manager.finalized(), "Assignments invalidate finalization"); manager.finalize();
    Array field("field", 6), gx("gx", 2), gy("gy", 2), gz("gz", 2);
    Kokkos::deep_copy(exec, field, 20.0); Kokkos::deep_copy(exec, gx, 0.0);
    Kokkos::deep_copy(exec, gy, 4.0); Kokkos::deep_copy(exec, gz, 0.0);
    const auto correction = grad::GradientCorrection{grad::cell_gradients(gx, gy, gz)};
    Kokkos::View<double*[4], Space> result("result", 4);
    sc::Status<Exec> status(exec); status.begin_async();
    manager.apply_component_async(field, status, Law{}, correction, 2.0);

    // Evaluate relations separately from field filling and compare both with analytic results.
    manager.evaluate_async(Capture{result}, status, Law{}, correction, 2.0); status.check();
    const auto h = snapshot(exec, field); const auto r = snapshot(exec, result);
    const double expected[]{20, 20, 18, 22, 61, -1};
    for (std::size_t i = 0; i < 6; ++i) near(h(i), expected[i], 1e-12, "Analytic profile and correction");
    near(r(0, 3), 16.0, 1e-12, "Dirichlet gradient includes tangential correction");
    near(r(2, 1), 41.0, 1e-12, "Prescribed derivative relation");
    near(r(3, 3), -12.0, 1e-12, "Outward heat flux converted by supplied law");
    status.begin_async(); manager.apply_component_async(field, status, Law{}, correction, 3.0); status.check();
    const auto later = snapshot(exec, field);
    near(later(2), 21.0, 1e-12, "Time changes without refinalization");
    check(manager.finalized() && !status.pending(), "Configuration and status have independent lifecycles");
}

/// @brief Checks const strided patch arrays, borrowed updates and preservation of provider storage.
void arrays_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    Kokkos::View<float*[2], Kokkos::LayoutRight, Space> storage("strided", 2);
    auto data = Kokkos::subview(storage, Kokkos::ALL(), 1);
    Kokkos::View<const float*, Kokkos::LayoutStride, Space> readonly = data;
    const auto range = layout.catalog().patches()[0].range;
    const auto values = pr::patch_values(range, readonly);
    static_assert(std::is_same_v<typename decltype(values.values)::value_type, const float>);
    using Condition = decltype(sc::fixed_value<double>(values));
    sc::Manager<Exec, double, Condition> manager(layout);
    manager.set(0, sc::fixed_value<double>(values));
    manager.set(1, sc::NormalGradient<double>{0}); manager.set(2, sc::NormalGradient<double>{0});
    manager.set_unlabelled(sc::NormalGradient<double>{0}); manager.finalize();
    auto hs = Kokkos::create_mirror(storage); hs(0, 0) = hs(1, 0) = -123; hs(0, 1) = 11; hs(1, 1) = 17;
    Kokkos::deep_copy(exec, storage, hs);
    Array field("field", 6); Kokkos::deep_copy(exec, field, 20.0);
    sc::Status<Exec> status(exec); status.begin_async(); manager.apply_component_async(field, status); status.check();
    auto h = snapshot(exec, field); near(h(2), 11.0, 1e-12, "First local patch value");
    near(h(3), 17.0, 1e-12, "Strided second local patch value");

    // Changing borrowed values must become visible without rebinding or refinalizing.
    Kokkos::deep_copy(exec, data, 29.0f);
    status.begin_async(); manager.apply_component_async(field, status); status.check();
    h = snapshot(exec, field); near(h(2), 29.0, 1e-12, "Borrowed buffer updated without rebinding");
    check(manager.finalized(), "Updating values does not invalidate finalization");
    near(snapshot(exec, storage)(0, 0), -123.0f, 1e-6f, "Provider never writes the input storage");
}

/// @brief Checks array-based gradient/flux data on labelled, unlabelled and empty patches.
void array_kinds_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    Array data("data", 2), one("one", 1), empty("empty", 0);
    const auto gradient = sc::normal_gradient<double>(pr::patch_values(layout.catalog().patches()[0].range, data));
    const auto flux = sc::fixed_flux<double>(pr::patch_values(layout.catalog().patches()[1].range, one));
    const auto empty_flux = sc::fixed_flux<double>(pr::patch_values(layout.catalog().patches()[2].range, empty));
    const auto unlabelled = sc::fixed_flux<double>(pr::patch_values(layout.catalog().unlabelled(), one));
    sc::Manager<Exec, double, decltype(gradient), decltype(flux)> manager(layout);
    manager.set(0, gradient); manager.set(1, flux); manager.set(2, empty_flux);
    manager.set_unlabelled(unlabelled); manager.finalize();
    Kokkos::deep_copy(exec, data, 3.0); Kokkos::deep_copy(exec, one, 8.0);
    Array field("field", 6); Kokkos::deep_copy(exec, field, 20.0);
    sc::Status<Exec> status(exec); status.begin_async(); manager.apply_component_async(field, status, Law{}); status.check();
    const auto h = snapshot(exec, field); const double expected[]{20, 20, 23, 23, 12, 12};
    for (std::size_t i = 0; i < 6; ++i) near(h(i), expected[i], 1e-12, "Array gradient/flux and unlabelled data");
}

/// @brief Aborts if a prescribed-value path evaluates its unused correction.
struct NeverReadCorrection {
    KOKKOS_INLINE_FUNCTION double operator()(const bc::FaceContext<double>&) const
    { Kokkos::abort("Dirichlet must not read a correction"); return 0; }
};

/// @brief Checks prescribed values without reading NaN owner values or unused corrections.
void independent_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    using C = decltype(sc::fixed_value<double>(SpaceTime<double>{}));
    sc::Manager<Exec, double, C> manager(layout);
    for (krona::mesh::PatchId p = 0; p < 3; ++p) manager.set(p, C{SpaceTime<double>{10}});
    manager.set_unlabelled(C{SpaceTime<double>{10}}); manager.finalize();
    Array field("field", 6); Kokkos::deep_copy(exec, field, std::numeric_limits<double>::quiet_NaN());
    sc::Status<Exec> status(exec); status.begin_async();

    // A prescribed value is independent of both the unavailable owner and tangential correction.
    manager.apply_component_async(field, status, sc::NoFluxLaw{}, NeverReadCorrection{}, 2.0); status.check();
    const auto h = snapshot(exec, field);
    const double expected[]{18, 22, 20, 24};
    for (std::size_t i = 0; i < 4; ++i) near(h(i+2), expected[i], 1e-12, "Profiled Dirichlet ignores NaN owners/correction");
}

/// @brief Checks two managers share layout/catalog storage while retaining independent conditions.
void shared_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    sc::Manager<Exec, double> a(layout), b(layout);
    for (krona::mesh::PatchId p = 0; p < 3; ++p) { a.set(p, sc::FixedValue<double>{7}); b.set(p, sc::FixedValue<double>{9}); }
    a.set_unlabelled(sc::FixedValue<double>{7}); b.set_unlabelled(sc::FixedValue<double>{9}); a.finalize(); b.finalize();
    check(&a.catalog() == &b.catalog() && &a.layout() == &layout, "Managers share exactly one catalog/layout");
    Array q("q", 6), r("r", 6); sc::Status<Exec> status(exec); status.begin_async();
    a.apply_component_async(q, status); b.apply_component_async(r, status); status.check();
    near(snapshot(exec, q)(2), 7.0, 1e-12, "First manager owns its conditions");
    near(snapshot(exec, r)(2), 9.0, 1e-12, "Second manager owns its conditions");
    a.set(0, sc::FixedValue<double>{5}); check(!a.finalized() && b.finalized(), "Changing one manager leaves the other finalized");
    a.finalize();
}

/// @brief Checks constant and borrowed vector-valued profile providers inside a device kernel.
void vectors_case(const Exec& exec)
{
    using V = krona::math::Vector3<double>;
    Kokkos::View<V*, Space> input("velocity_profile", 2), output("velocity_result", 2);
    const pr::Constant constant{V{1, 2, 3}};
    const bc::PatchRange range{0, 10, 12}; const auto values = pr::patch_values(range, input);
    pr::validate<Exec>(constant, range); pr::validate<Exec>(values, range);
    Kokkos::deep_copy(exec, input, V{4, 5, 6});
    Kokkos::parallel_for("test/vector_providers", Kokkos::RangePolicy<Exec>(exec, 0, 2), KOKKOS_LAMBDA(int i) {
        bc::FaceContext<double> c; c.face = 10+i;
        output(i) = i == 0 ? pr::sample(constant, c) : pr::sample(values, c);
    });
    const auto h = snapshot(exec, output);
    near(h(0).y, 2.0, 1e-12, "Constant vector provider"); near(h(1).z, 6.0, 1e-12, "Borrowed vector profile");
}

/// @brief Counts profile evaluations independently for every boundary face.
struct Counted {
    Kokkos::View<int*, Space> calls;
    KOKKOS_INLINE_FUNCTION double operator()(const bc::FaceContext<double>& c) const
    { ++calls(c.boundary_index); return 7; }
};

/// @brief Checks one provider sample per face for each apply/evaluate operation.
void once_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    const auto condition = sc::fixed_value<double>(Counted{Kokkos::View<int*, Space>("calls", 4)});
    sc::Manager<Exec, double, decltype(condition)> manager(layout);
    for (krona::mesh::PatchId p = 0; p < 3; ++p) manager.set(p, condition);
    manager.set_unlabelled(condition); manager.finalize();
    Array field("field", 6); Kokkos::View<double*[4], Space> result("result", 4);
    sc::Status<Exec> status(exec); status.begin_async(); manager.apply_component_async(field, status);
    manager.evaluate_async(Capture{result}, status); status.check();
    const auto h = snapshot(exec, condition.value.calls);
    for (int i = 0; i < 4; ++i) equal(h(i), 2, "Provider sampled once per face per operation");
}

/// @brief Checks rank-local patch arrays when ranks have different local patch sizes.
void distributed_case(const Exec& exec)
{
    int rank = 0, size = 0; krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "rank");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &size), "size"); equal(size, 2, "Requires two ranks");
    auto host = fixture(); host.n_bfaces_g = 8; host.n_labeled_bfaces_g = 6;

    // Different per-rank patch sizes expose assumptions about global profile-array indexing.
    if (rank == 1) { host.face_patch = {0, 1, 1, -1}; }
    krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    const auto range = layout.catalog().patches()[0].range;
    Array values("local_profile", static_cast<std::size_t>(range.end-range.begin));
    const auto condition = sc::fixed_value<double>(pr::patch_values(range, values));
    sc::Manager<Exec, double, decltype(condition)> manager(layout);
    manager.set(0, condition); manager.set(1, sc::NormalGradient<double>{0}); manager.set(2, sc::NormalGradient<double>{0});
    manager.set_unlabelled(sc::NormalGradient<double>{0}); manager.finalize();
    Kokkos::deep_copy(exec, values, 100.0+rank); Array field("field", 6); Kokkos::deep_copy(exec, field, 20.0);
    sc::Status<Exec> status(exec); status.begin_async(); manager.apply_component_async(field, status); status.check();
    const auto h = snapshot(exec, field);
    for (auto f = range.begin; f < range.end; ++f) near(h(static_cast<std::size_t>(f)+2), 100.0+rank, 1e-12, "Rank-local patch order");
}

/// @brief Checks captured device-lambda providers through rebinding and configuration reset.
void lambda_case(const Exec& exec)
{
    const auto host = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(host, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(host, mesh.view(), exec);
    const double base = 7;
    const auto provider = KOKKOS_LAMBDA(const bc::FaceContext<double>& c) { return base+c.time; };
    const auto condition = sc::fixed_value<double>(provider);
    sc::Manager<Exec, double, decltype(condition)> manager(layout);
    for (krona::mesh::PatchId p = 0; p < 3; ++p) manager.set(p, condition);
    manager.set_unlabelled(condition); manager.finalize();
    Array field("field", 6); sc::Status<Exec> status(exec); status.begin_async();
    manager.apply_component_async(field, status, sc::NoFluxLaw{}, grad::NoCorrection{}, 2.0); status.check();
    near(snapshot(exec, field)(2), 9.0, 1e-12, "Device lambda with a capture");

    // Rebinding and config assignment must not require the lambda's deleted assignment operator.
    manager.set(0, condition); manager.finalize();
    sc::Configuration config;
    for (std::size_t p = 0; p < layout.catalog().patches().size(); ++p)
        config.patches.push_back({static_cast<std::int64_t>(p), layout.catalog().patches()[p].name,
                                  {sc::Kind::FixedValue, 11}});
    config.unlabelled = sc::ConditionConfig{sc::Kind::NormalGradient, 0};
    manager.configure(config); manager.finalize();
    status.begin_async(); manager.apply_component_async(field, status); status.check();
    near(snapshot(exec, field)(2), 11.0, 1e-12, "Config reset with a lambda alternative present");
}

/// @brief Rejects visitation after assignments invalidate manager finalization.
void dirty_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec); sc::Manager<Exec, double> manager(layout);
    for (krona::mesh::PatchId p = 0; p < 3; ++p) manager.set(p, sc::FixedValue<double>{7});
    manager.set_unlabelled(sc::FixedValue<double>{7}); manager.finalize(); manager.set(0, sc::FixedValue<double>{9});
    manager.for_each_patch([](bc::PatchRange, const auto&) {});
}

/// @brief Rejects a profile array shorter than its assigned patch range.
void short_array_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec);
    Array values("short", 1); const auto c = sc::fixed_value<double>(pr::patch_values(layout.catalog().patches()[0].range, values));
    sc::Manager<Exec, double, decltype(c)> manager(layout); manager.set(0, c);
}

/// @brief Rejects a profile bound to a different patch range.
void wrong_range_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec);
    Array values("one", 1); const auto c = sc::fixed_value<double>(pr::patch_values(layout.catalog().unlabelled(), values));
    sc::Manager<Exec, double, decltype(c)> manager(layout); manager.set(1, c);
}

/// @brief Supplies a nonfinite value to test profile-result validation.
struct Invalid {
    KOKKOS_INLINE_FUNCTION double operator()(const bc::FaceContext<double>&) const
    { return Kokkos::quiet_NaN_v<double>; }
};

/// @brief Rejects a nonfinite sampled normal-gradient profile.
void nonfinite_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec);
    const auto c = sc::normal_gradient<double>(Invalid{}); sc::Manager<Exec, double, decltype(c)> manager(layout);
    manager.set(0, c); manager.set(1, sc::NormalGradient<double>{0}); manager.set(2, sc::NormalGradient<double>{0});
    manager.set_unlabelled(sc::NormalGradient<double>{0}); manager.finalize();
    Array field("field", 6); Kokkos::deep_copy(exec, field, 20.0);
    sc::Status<Exec> status(exec); status.begin_async(); manager.apply_component_async(field, status); status.check();
}

/// @brief Ignores its flux argument to verify validation occurs before constitutive conversion.
struct IgnoringLaw {
    KOKKOS_INLINE_FUNCTION double normal_gradient(const bc::FaceContext<double>&, double) const { return 0; }
};

/// @brief Rejects nonfinite prescribed flux even if a custom law would ignore its value.
void nonfinite_flux_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec);
    const auto c = sc::fixed_flux<double>(Invalid{}); sc::Manager<Exec, double, decltype(c)> manager(layout);
    manager.set(0, c); manager.set(1, sc::NormalGradient<double>{0}); manager.set(2, sc::NormalGradient<double>{0});
    manager.set_unlabelled(sc::NormalGradient<double>{0}); manager.finalize();
    Array field("field", 6); Kokkos::deep_copy(exec, field, 20.0);
    sc::Status<Exec> status(exec); status.begin_async();
    manager.apply_component_async(field, status, IgnoringLaw{}); status.check();
}

/// @brief Rejects a boundary value unrepresentable in float destination storage.
void overflow_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec); sc::Manager<Exec, double> manager(layout);
    for (krona::mesh::PatchId p = 0; p < 3; ++p) manager.set(p, sc::FixedValue<double>{1e100});
    manager.set_unlabelled(sc::FixedValue<double>{1e100}); manager.finalize();
    Kokkos::View<float*, Space> field("float", 6); sc::Status<Exec> status(exec); status.begin_async();
    manager.apply_component_async(field, status); status.check();
}

/// @brief Rejects device-status access before beginning an operation.
void status_not_begun_case(const Exec& exec)
{
    sc::Status<Exec> status(exec); (void)status.device_flags(exec);
}

/// @brief Rejects profiled fixed-flux application without a transport law.
void profile_flux_without_law_case(const Exec& exec)
{
    const auto h = fixture(); krona::mesh::device::MeshStorage<Exec> mesh(h, {}, exec);
    bc::BoundaryLayout<Exec, double> layout(h, mesh.view(), exec);
    const auto c = sc::fixed_flux<double>(SpaceTime<double>{1}); sc::Manager<Exec, double, decltype(c)> manager(layout);
    manager.set(0, c); manager.set(1, sc::NormalGradient<double>{0}); manager.set(2, sc::NormalGradient<double>{0});
    manager.set_unlabelled(sc::NormalGradient<double>{0}); manager.finalize();
    Array field("field", 6); sc::Status<Exec> status(exec); status.begin_async(); manager.apply_component_async(field, status);
}
} // namespace

/// @brief Runs scalar profile cases, including individually selected contract rejections.
int main(int argc, char** argv)
{
    constexpr std::array cases{Case{"relations", relations_case}, Case{"analytic", analytic_case},
        Case{"arrays", arrays_case}, Case{"array_kinds", array_kinds_case}, Case{"independent", independent_case},
        Case{"shared", shared_case}, Case{"lambda", lambda_case}, Case{"vectors", vectors_case}, Case{"once", once_case}, Case{"distributed", distributed_case},
        Case{"dirty", dirty_case}, Case{"short_array", short_array_case}, Case{"wrong_range", wrong_range_case},
        Case{"nonfinite", nonfinite_case}, Case{"nonfinite_flux", nonfinite_flux_case}, Case{"overflow", overflow_case}, Case{"status_not_begun", status_not_begun_case},
        Case{"profile_flux_without_law", profile_flux_without_law_case}};
    return krona::test::run(argc, argv, "scalar_profiles", cases);
}
