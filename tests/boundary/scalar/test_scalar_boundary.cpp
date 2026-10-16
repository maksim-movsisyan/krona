#include "common/test_support.hpp"

#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/transport/isotropic_diffusion.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

namespace {
using namespace krona::test;
namespace bc = krona::boundary;
namespace scalar = bc::scalar;
namespace transport = krona::transport;
using Real = krona::ComputeReal;
using Memory = typename Exec::memory_space;
using Array = Kokkos::View<Real*, Memory>;

krona::mesh::MeshPart fixture()
{
    krona::mesh::MeshPart m;
    m.n_own = 2; m.n_cells = 3;
    m.n_faces = 5; m.n_internal_faces = 0; m.n_inner_faces = 1;
    m.faces_grouped = true;
    m.n_bfaces_g = 4; m.n_labeled_bfaces_g = 3;
    m.face_owner = {1, 0, 1, 0, 1};
    m.face_neigh = {2, -1, -1, -1, -1};
    m.face_patch = {-1, 0, 0, 1, -1};
    m.patches = {{"value", "wall", "BCWall"}, {"gradient", "", "BCOutflow"},
                 {"empty", "", "BCWall"}};
    m.cell_centroid_x = {0, 2, 4};
    m.cell_centroid_y = {0, 0, 0}; m.cell_centroid_z = {0, 0, 0};
    m.cell_volume = {1, 1, 1};
    m.face_centroid_x = {3, 1, 3, krona::GeometryReal{0.5}, 4};
    m.face_centroid_y = {0, krona::GeometryReal{0.25}, krona::GeometryReal{-0.5},
                         krona::GeometryReal{0.5}, krona::GeometryReal{0.75}};
    m.face_centroid_z = {0, 0, 0, 0, 0};
    m.face_normal_x = {1, 1, 1, 1, 1};
    m.face_normal_y = {0, 0, 0, 0, 0}; m.face_normal_z = {0, 0, 0, 0, 0};
    m.face_area = {1, 2, 3, 4, 5};
    return m;
}

template<class Manager>
void configure(Manager& manager, bool flux)
{
    manager.set(0, scalar::FixedValue<Real>{7});
    manager.set(1, scalar::NormalGradient<Real>{3});
    manager.set(2, scalar::NormalGradient<Real>{0});
    if (flux) manager.set_unlabelled(scalar::FixedFlux<Real>{8});
    else manager.set_unlabelled(scalar::NormalGradient<Real>{0});
}

constexpr Real tolerance = Real{128} *
    (std::numeric_limits<Real>::epsilon() +
     static_cast<Real>(std::numeric_limits<krona::GeometryReal>::epsilon()));

struct CountingCoefficient {
    Array values;
    Kokkos::View<int*, Memory> calls;
    KOKKOS_INLINE_FUNCTION Real operator()(const bc::FaceContext<Real>& c) const
    {
        ++calls(static_cast<std::size_t>(c.face));
        return values(static_cast<std::size_t>(c.owner));
    }
};

struct CaptureRelation {
    Kokkos::View<Real*[4], Memory> values;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c,
                                           const scalar::Relation<Real>& r) const
    {
        values(c.boundary_index, 0) = r.value_owner;
        values(c.boundary_index, 1) = r.value_constant;
        values(c.boundary_index, 2) = r.gradient_owner;
        values(c.boundary_index, 3) = r.gradient_constant;
    }
};

void reconstruction_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    configure(manager, true);
    Kokkos::View<Real*[2], Kokkos::LayoutRight, Memory> field("field", 7);
    auto host = Kokkos::create_mirror(field);
    Kokkos::deep_copy(host, Real{-777});
    host(0, 0) = 10; host(1, 0) = 20; host(2, 0) = 99;
    Kokkos::deep_copy(exec, field, host);

    Array coefficient("coefficient", 2), gx("gx", 2), gy("gy", 2), gz("gz", 2);
    auto hc = Kokkos::create_mirror(coefficient); hc(0) = 2; hc(1) = 4;
    Kokkos::deep_copy(exec, coefficient, hc);
    Kokkos::deep_copy(exec, gx, Real{-100});
    Kokkos::deep_copy(exec, gy, Real{4}); Kokkos::deep_copy(exec, gz, Real{0});
    Kokkos::View<int*, Memory> calls("calls", 5);
    Kokkos::deep_copy(exec, calls, 0);
    const auto law = transport::IsotropicDiffusion{CountingCoefficient{coefficient, calls}};
    const auto correction = bc::GradientCorrection{gx, gy, gz};
    manager.apply_component_async(Kokkos::subview(field, Kokkos::ALL(), 0), law, correction);
    Kokkos::View<Real*[4], Memory> relations("relations", 4);
    manager.evaluate_async(CaptureRelation{relations}, law, correction);

    Kokkos::deep_copy(exec, host, field);
    auto hr = Kokkos::create_mirror(relations); Kokkos::deep_copy(exec, hr, relations);
    auto calls_host = Kokkos::create_mirror(calls); Kokkos::deep_copy(exec, calls_host, calls);
    exec.fence();
    constexpr std::array<Real, 7> expected{10, 20, 99, 7, 7, Real{13.5}, 19};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        near(host(i, 0), expected[i], tolerance, "Corrected boundary values and preserved cells");
        equal(host(i, 1), Real{-777}, "Unselected AoS component is preserved");
    }
    constexpr Real coefficients[4][4]{{0, 7, -1, 6}, {0, 7, -1, 9},
                                     {1, Real{3.5}, 0, 3}, {1, -1, 0, -2}};
    for (std::size_t f = 0; f < 4; ++f)
        for (std::size_t c = 0; c < 4; ++c)
            near(hr(f, c), coefficients[f][c], tolerance, "Affine value and gradient coefficients");
    for (std::size_t f = 0; f < 5; ++f)
        equal(calls_host(f), f == 4 ? 2 : 0, "Only FixedFlux accesses the material coefficient");

    const auto owner_law = transport::IsotropicDiffusion{transport::OwnerCellCoefficient{coefficient}};
    manager.apply_component_async(Kokkos::subview(field, Kokkos::ALL(), 0), owner_law, correction);
    Kokkos::deep_copy(exec, host, field); exec.fence();
    near(host(6, 0), Real{19}, tolerance, "Owner-cell coefficient adapter");
}

struct PrescribedFluxSink {
    Array integrated;
    template<class C>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c, const C& condition) const
    {
        if constexpr (std::is_same_v<C, scalar::FixedFlux<Real>>)
            integrated(c.boundary_index) = condition.flux*c.area;
        else integrated(c.boundary_index) = Real{-1};
    }
};

void visitation_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    configure(manager, true);
    int patches = 0;
    krona::LocalIndex faces = 0;
    manager.for_each_patch([&](bc::PatchRange range, const auto&) {
        ++patches; faces += range.end - range.begin;
    });
    equal(patches, 3, "Empty labelled patches do not dispatch");
    equal(faces, 4, "Labelled and unlabelled ranges cover the boundary");
    Array flux("integrated_flux", 4);
    // Raw visitation intentionally needs no transport law, even with FixedFlux.
    manager.for_each_face_async(PrescribedFluxSink{flux});
    auto host = Kokkos::create_mirror(flux); Kokkos::deep_copy(exec, host, flux); exec.fence();
    for (std::size_t i = 0; i < 3; ++i) equal(host(i), Real{-1}, "Typed non-flux condition");
    equal(host(3), Real{40}, "Prescribed flux is assembled directly with area and sign");
}

struct TimeValue {
    KOKKOS_INLINE_FUNCTION scalar::Relation<Real> operator()(const bc::FaceContext<Real>& c, Real t) const
    { return scalar::FixedValue<Real>{c.x + Real{2}*c.time}(c, t); }
};

void custom_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real, TimeValue> manager(m, mesh.view(), exec);
    manager.set(0, TimeValue{});
    manager.set(1, scalar::NormalGradient<Real>{0});
    manager.set(2, scalar::NormalGradient<Real>{0});
    manager.set_unlabelled(scalar::NormalGradient<Real>{0});
    krona::fields::FieldStorage<Real, 1, Exec> a("a", 7, krona::fields::FieldLocation::Custom, exec);
    krona::fields::FieldStorage<Real, 1, Exec> b("b", 7, krona::fields::FieldLocation::Custom, exec);
    a.fill(exec, Real{10}); b.fill(exec, Real{10});
    manager.apply_async(a.view(), scalar::NoFluxLaw{}, bc::NoCorrection{}, Real{1.5});
    manager.set(0, scalar::FixedValue<Real>{9});
    manager.apply_async(b.view());
    auto ha = a.create_host_buffer(), hb = b.create_host_buffer();
    Kokkos::deep_copy(exec, ha, a.view().kokkos_view());
    Kokkos::deep_copy(exec, hb, b.view().kokkos_view()); exec.fence();
    near(ha(3), Real{4}, tolerance, "Custom condition reads face position and time");
    near(ha(4), Real{6}, tolerance, "Queued condition owns its original parameter snapshot");
    near(hb(3), Real{9}, tolerance, "A later set affects subsequent application");
    near(hb(4), Real{9}, tolerance, "A later set covers the complete patch");
    equal(ha(0), Real{10}, "Owned cell is preserved");
    equal(ha(2), Real{10}, "MPI ghost is preserved");
}

void relations_case(const Exec& exec)
{
    Kokkos::View<Real*[5], Memory> result("relations", 1);
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{4}};
    const auto fractional = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{Real{0.5}}};
    Kokkos::parallel_for("boundary/relations", Kokkos::RangePolicy<Exec>(exec, 0, 1),
        KOKKOS_LAMBDA(int) {
            bc::FaceContext<Real> c; c.distance = 2;
            const auto r = scalar::make_relation(scalar::FixedFlux<Real>{8}, c, Real{3}, law);
            result(0, 0) = r.face_value(20);
            result(0, 1) = r.normal_gradient(20);
            result(0, 2) = law.outward_flux(c, r.normal_gradient(20));
            result(0, 3) = scalar::NormalGradient<Real>{0}(c, Real{3}).face_value(20);
            result(0, 4) = fractional.normal_gradient(c, 1);
        });
    auto host = Kokkos::create_mirror(result); Kokkos::deep_copy(exec, host, result); exec.fence();
    near(host(0, 0), Real{19}, tolerance, "Flux to face value");
    near(host(0, 1), Real{-2}, tolerance, "Outward flux to normal gradient sign");
    near(host(0, 2), Real{8}, tolerance, "Constitutive round trip excludes area");
    near(host(0, 3), Real{23}, tolerance, "Zero normal gradient retains tangential reconstruction");
    near(host(0, 4), Real{-2}, tolerance, "Integer arguments preserve floating coefficient arithmetic");
}

void empty_case(const Exec& exec)
{
    krona::mesh::MeshPart m; m.faces_grouped = true;
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    krona::fields::FieldStorage<Real, 1, Exec> field("empty", 0, krona::fields::FieldLocation::Custom, exec);
    manager.apply_async(field.view());
    int count = 0;
    manager.for_each_patch([&](bc::PatchRange, const auto&) { ++count; });
    equal(count, 0, "Empty partition has no patch dispatch");
    equal(manager.required_field_size(), std::size_t{0}, "Empty extended field size");
    exec.fence();
}

void independent_value_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    for (krona::mesh::PatchId p = 0; p < 3; ++p)
        manager.set(p, scalar::FixedValue<Real>{7});
    manager.set_unlabelled(scalar::FixedValue<Real>{7});
    Array field("field", 7);
    Kokkos::deep_copy(exec, field, std::numeric_limits<Real>::quiet_NaN());
    manager.apply_component_async(field);
    auto host = Kokkos::create_mirror(field); Kokkos::deep_copy(exec, host, field); exec.fence();
    for (std::size_t i = 3; i < 7; ++i)
        equal(host(i), Real{7}, "Fixed value does not depend on an unavailable owner value");
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    bc::FaceContext<Real> c; c.distance = 1;
    equal(scalar::FixedValue<Real>{7}(c).face_value(nan), Real{7}, "Zero value coefficient ignores owner NaN");
    equal(scalar::NormalGradient<Real>{3}(c).normal_gradient(nan), Real{3}, "Prescribed derivative ignores owner NaN");
}

template<class Q, class X, class Y, class Z>
struct ReconstructBoundaryGradient {
    Q q;
    X gx;
    Y gy;
    Z gz;

    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c,
                                           const scalar::Relation<Real>& r) const
    {
        const auto p = static_cast<std::size_t>(c.owner);
        const Real px = static_cast<Real>(gx(p));
        const Real py = static_cast<Real>(gy(p));
        const Real pz = static_cast<Real>(gz(p));
        const Real owner_value = r.gradient_owner == Real{0} ? Real{0} : static_cast<Real>(q(p));
        const Real normal = r.normal_gradient(owner_value);
        const Real delta = normal - (c.nx*px + c.ny*py + c.nz*pz);
        gx(c.extended_index) = static_cast<typename X::non_const_value_type>(px + c.nx*delta);
        gy(c.extended_index) = static_cast<typename Y::non_const_value_type>(py + c.ny*delta);
        gz(c.extended_index) = static_cast<typename Z::non_const_value_type>(pz + c.nz*delta);
    }
};

void boundary_gradients_case(const Exec& exec)
{
    auto m = fixture();
    m.face_normal_x[3] = krona::GeometryReal{2}/3;
    m.face_normal_y[3] = krona::GeometryReal{2}/3;
    m.face_normal_z[3] = krona::GeometryReal{1}/3;
    m.face_centroid_x[3] = 1;
    m.face_centroid_y[3] = 0;
    m.face_centroid_z[3] = 1;
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    configure(manager, true);
    Array q("q", 7), coefficient("coefficient", 2);
    auto hq = Kokkos::create_mirror(q);
    Kokkos::deep_copy(hq, Real{-777});
    hq(0) = 10; hq(1) = 20; hq(2) = 99;
    Kokkos::deep_copy(exec, q, hq);
    auto hc = Kokkos::create_mirror(coefficient); hc(0) = 2; hc(1) = 4;
    Kokkos::deep_copy(exec, coefficient, hc);
    Kokkos::View<Real*[3], Kokkos::LayoutRight, Memory> gradient("gradient", 7);
    auto hg = Kokkos::create_mirror(gradient);
    Kokkos::deep_copy(hg, Real{-777});
    hg(0, 0) = 2; hg(0, 1) = -3; hg(0, 2) = Real{0.75};
    hg(1, 0) = -1; hg(1, 1) = 4; hg(1, 2) = 2;
    Kokkos::deep_copy(exec, gradient, hg);
    const auto gx = Kokkos::subview(gradient, Kokkos::ALL(), 0);
    const auto gy = Kokkos::subview(gradient, Kokkos::ALL(), 1);
    const auto gz = Kokkos::subview(gradient, Kokkos::ALL(), 2);
    const auto law = transport::IsotropicDiffusion{transport::OwnerCellCoefficient{coefficient}};
    manager.evaluate_async(ReconstructBoundaryGradient{q, gx, gy, gz}, law,
                           bc::GradientCorrection{gx, gy, gz});
    Kokkos::deep_copy(exec, hg, gradient);
    auto after = Kokkos::create_mirror(q); Kokkos::deep_copy(exec, after, q); exec.fence();
    constexpr Real expected[7][3]{{2, -3, Real{0.75}}, {-1, 4, 2}, {-777, -777, -777},
        {Real{-2.25}, -3, Real{0.75}}, {-11, 4, 2},
        {Real{77}/18, Real{-13}/18, Real{17}/9}, {-2, 4, 2}};
    for (std::size_t i = 0; i < 7; ++i) {
        equal(after(i), hq(i), "Evaluate does not fill or modify scalar values");
        for (std::size_t axis = 0; axis < 3; ++axis)
            near(hg(i, axis), expected[i][axis], tolerance,
                 "Boundary gradient enforces the normal derivative and preserves owner tangents/cells");
    }
}

// Matches data/bc.toml: two owned cells, one MPI ghost and three labelled patches.
krona::mesh::MeshPart file_fixture(bool skew = false, bool unlabelled = false, bool empty = false)
{
    using G = krona::GeometryReal;
    krona::mesh::MeshPart m;
    m.faces_grouped = true;
    m.patches = {{"fixed", "inlet", "BCInflow"}, {"gradient", "wall", "BCWall"},
                 {"flux", "wall", "BCWall"}};
    m.n_bfaces_g = unlabelled ? 5 : 4;
    m.n_labeled_bfaces_g = 4;
    if (empty) return m;
    m.n_own = 2; m.n_cells = 3;
    m.n_faces = unlabelled ? 6 : 5;
    m.n_internal_faces = 1; m.n_inner_faces = 1;
    m.cell_centroid_x = {0, 2, 4};
    m.cell_centroid_y = {0, 0, 0}; m.cell_centroid_z = {0, 0, 0};
    m.cell_volume = {1, 1, 1};
    m.face_owner = {0, 0, 1, 0, 1}; m.face_neigh = {1, -1, -1, -1, -1};
    m.face_patch = {-1, 0, 0, 1, 2};
    m.face_centroid_x = {1, 1, 3, -1, 4};
    m.face_centroid_y = {0, skew ? static_cast<G>(0.2) : G{0},
        skew ? static_cast<G>(-0.3) : G{0}, skew ? G{0.25} : G{0},
        skew ? static_cast<G>(-0.4) : G{0}};
    m.face_centroid_z = {0, skew ? static_cast<G>(0.1) : G{0},
        skew ? static_cast<G>(-0.15) : G{0}, skew ? static_cast<G>(0.2) : G{0},
        skew ? G{-0.5} : G{0}};
    m.face_normal_x = {1, 1, 1, -1, 1};
    m.face_normal_y = {0, 0, 0, 0, 0}; m.face_normal_z = {0, 0, 0, 0, 0};
    m.face_area = {1, 2, 1, 1, 1};
    if (unlabelled) {
        m.face_owner.push_back(0); m.face_neigh.push_back(-1); m.face_patch.push_back(-1);
        m.face_centroid_x.push_back(0); m.face_centroid_y.push_back(1); m.face_centroid_z.push_back(0);
        m.face_normal_x.push_back(0); m.face_normal_y.push_back(1); m.face_normal_z.push_back(0);
        m.face_area.push_back(1);
    }
    return m;
}

constexpr auto selection = krona::mesh::compute::MeshData::FaceCells |
                           krona::mesh::compute::MeshData::Geometry;

scalar::Configuration file_configuration()
{ return scalar::read_config(KRONA_BOUNDARY_TEST_DATA, "temperature"); }

template<class View>
auto snapshot(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view);
    exec.fence();
    return host;
}

template<class View>
void initialize_component(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(host, Real{-99});
    if (host.extent(0) >= 3) { host(0) = 7; host(1) = 11; host(2) = 42; }
    Kokkos::deep_copy(exec, view, host);
    // The host staging allocation is local to this helper.
    exec.fence();
}

void config_case(const Exec& exec)
{
    std::ifstream file(KRONA_BOUNDARY_TEST_DATA);
    krona::mpi::require(file.is_open(), "Cannot open scalar boundary test data");
    const std::string text{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    const auto parsed = scalar::parse_config(text, "temperature");
    const auto loaded = file_configuration();
    const auto tracer = scalar::read_config(KRONA_BOUNDARY_TEST_DATA, "tracer");
    equal(loaded.patches.size(), std::size_t{3}, "Read all labelled patches from a nonempty file");
    equal(parsed.patches.size(), loaded.patches.size(), "Text and file readers agree");
    for (std::size_t p = 0; p < loaded.patches.size(); ++p) {
        equal(parsed.patches[p].patch_id, loaded.patches[p].patch_id, "Parsed patch ID");
        equal(parsed.patches[p].name == loaded.patches[p].name, true, "Parsed patch name");
        check(parsed.patches[p].condition.kind == loaded.patches[p].condition.kind, "Parsed condition kind");
        equal(parsed.patches[p].condition.parameter, loaded.patches[p].condition.parameter, "Parsed parameter");
    }
    equal(loaded.patches[0].condition.parameter, 5.0, "Temperature-specific prescribed value");
    equal(tracer.patches[0].condition.parameter, 0.25, "Independent scalar-specific value");
    check(tracer.patches[1].condition.kind == scalar::Kind::NormalGradient, "ZERO_GRADIENT alias");
    equal(tracer.patches[1].condition.parameter, 0.0, "ZERO_GRADIENT parameter");
    krona::mpi::require(loaded.unlabelled.has_value() && tracer.unlabelled.has_value(),
                        "Both fields have explicit unlabelled conditions");
    check(loaded.unlabelled->kind == scalar::Kind::NormalGradient, "Read unlabelled condition");
    auto m = file_fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    auto shuffled = loaded;
    std::swap(shuffled.patches.front(), shuffled.patches.back());
    scalar::Manager<Exec, Real> manager(m, mesh.view(), shuffled, exec);
    equal(manager.catalog().patches()[0].range.begin, krona::LocalIndex{1}, "Bind by ID, not file order");
    equal(manager.catalog().patches()[0].range.end, krona::LocalIndex{3}, "Absolute patch end");
    equal(manager.catalog().unlabelled().empty(), true, "Internal face_patch=-1 is not a boundary patch");
    const auto empty = scalar::parse_config("boundary_condition = []", "temperature");
    equal(empty.patches.size(), std::size_t{0}, "Explicit empty patch array is accepted");
}

void file_fill_case(const Exec& exec)
{
    const auto m = file_fixture(false, true);
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), file_configuration(), exec);
    krona::fields::FieldStorage<Real, 1, Exec> field(
        "temperature", manager.required_field_size(), krona::fields::FieldLocation::CellExtnd, exec);
    initialize_component(exec, field.view().component(0));
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{3}};
    manager.apply_async(field.view(), law);
    auto host = snapshot(exec, field.view().component(0));
    constexpr Real expected[]{7, 11, 42, 5, 5, 5, 7, 7};
    for (std::size_t i = 0; i < 8; ++i)
        near(host(i), expected[i], tolerance, "File -> configuration -> manager -> boundary values");
    manager.set(0, scalar::FixedValue<Real>{8});
    manager.apply_async(field.view(), law);
    host = snapshot(exec, field.view().component(0));
    near(host(3), Real{8}, tolerance, "Updated condition overrides the file binding");
    near(host(4), Real{8}, tolerance, "Updated condition covers its entire patch");

    scalar::Manager<Exec, Real> tracer_manager(m, mesh.view(),
        scalar::read_config(KRONA_BOUNDARY_TEST_DATA, "tracer"), exec);
    krona::fields::FieldStorage<Real, 1, Exec> tracer(
        "tracer", tracer_manager.required_field_size(), krona::fields::FieldLocation::CellExtnd, exec);
    initialize_component(exec, tracer.view().component(0));
    tracer_manager.apply_async(tracer.view());
    const auto ht = snapshot(exec, tracer.view().component(0));
    constexpr Real expected_tracer[]{7, 11, 42, Real{0.25}, Real{0.25}, 7, 11, 7};
    for (std::size_t i = 0; i < 8; ++i)
        near(ht(i), expected_tracer[i], tolerance, "Other scalar conditions are applied independently");
}

template<class Layout>
void layout_case(const Exec& exec)
{
    const auto m = file_fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), file_configuration(), exec);
    krona::fields::FieldStorage<Real, 3, Exec, Layout> field(
        "vector", manager.required_field_size() + 2, krona::fields::FieldLocation::Custom, exec);
    field.fill(exec, Real{-17});
    initialize_component(exec, field.view().component(1));
    manager.apply_component_async(field.view().component(1),
        transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{3}});
    const auto host = snapshot(exec, field.view().kokkos_view());
    constexpr Real expected[]{7, 11, 42, 5, 5, 5, 7, -99, -99};
    for (std::size_t i = 0; i < 9; ++i) {
        near(host(i, 1), expected[i], tolerance, "Selected component, cell prefix and trailing storage");
        equal(host(i, 0), Real{-17}, "First unselected component is preserved");
        equal(host(i, 2), Real{-17}, "Last unselected component is preserved");
    }
}

void layouts_case(const Exec& exec)
{ layout_case<Kokkos::LayoutLeft>(exec); layout_case<Kokkos::LayoutRight>(exec); }

struct AffineValue {
    KOKKOS_INLINE_FUNCTION scalar::Relation<Real> operator()(const bc::FaceContext<Real>& c, Real increment) const
    {
        return scalar::FixedValue<Real>{Real{7} + Real{2}*c.x + Real{3}*c.y - Real{4}*c.z + Real{5}*c.time}
            (c, increment);
    }
};

template<class Values, class Law>
struct CapturePhysicalFlux {
    Values owner_values;
    Array fluxes;
    Law law;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c, const scalar::Relation<Real>& r) const
    {
        const Real owner = r.gradient_owner == Real{0} ? Real{0}
            : static_cast<Real>(owner_values(static_cast<std::size_t>(c.owner)));
        fluxes(c.boundary_index) = static_cast<Real>(law.outward_flux(c, r.normal_gradient(owner)));
    }
};

void affine_case(const Exec& exec)
{
    const auto m = file_fixture(true);
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real, AffineValue> manager(m, mesh.view(), file_configuration(), exec);
    manager.set(0, AffineValue{});
    manager.set(2, scalar::FixedFlux<Real>{-6});
    krona::fields::FieldStorage<Real, 1, Exec> field(
        "affine", manager.required_field_size(), krona::fields::FieldLocation::CellExtnd, exec);
    initialize_component(exec, field.view().component(0));
    const auto values = field.view();
    Kokkos::parallel_for("test/affine_time", Kokkos::RangePolicy<Exec>(exec, 0, 2),
        KOKKOS_LAMBDA(int i) { values(static_cast<std::size_t>(i)) += Real{5}; });
    Array gx("gx", 2), gy("gy", 2), gz("gz", 2), fluxes("fluxes", 4);
    Kokkos::deep_copy(exec, gx, Real{2}); Kokkos::deep_copy(exec, gy, Real{3});
    Kokkos::deep_copy(exec, gz, Real{-4});
    const auto correction = bc::GradientCorrection{gx, gy, gz};
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{3}};
    manager.apply_async(values, law, correction, Real{1});
    manager.evaluate_async(CapturePhysicalFlux{values.component(0), fluxes, law}, law, correction, Real{1});
    const auto host = snapshot(exec, values.component(0));
    const auto hf = snapshot(exec, fluxes);
    for (krona::LocalIndex f = m.n_inner_faces; f < m.n_faces; ++f) {
        const auto face = static_cast<std::size_t>(f);
        const auto b = static_cast<std::size_t>(f - m.n_inner_faces);
        const Real exact = Real{12} + Real{2}*static_cast<Real>(m.face_centroid_x[face]) +
            Real{3}*static_cast<Real>(m.face_centroid_y[face]) - Real{4}*static_cast<Real>(m.face_centroid_z[face]);
        const Real exact_flux = Real{-3}*(Real{2}*static_cast<Real>(m.face_normal_x[face]) +
            Real{3}*static_cast<Real>(m.face_normal_y[face]) - Real{4}*static_cast<Real>(m.face_normal_z[face]));
        near(host(static_cast<std::size_t>(m.n_cells) + b), exact, Real{16}*tolerance,
             "Corrected face value reproduces a space/time-dependent affine field");
        near(hf(b), exact_flux, Real{16}*tolerance, "Normal derivative reproduces the exact physical flux");
    }
}

template<class Law>
struct AddDiffusionBoundary {
    Array diagonal, rhs;
    Law law;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c, const scalar::Relation<Real>& r) const
    {
        const auto p = static_cast<std::size_t>(c.owner);
        const Real a = static_cast<Real>(law.outward_flux(c, r.gradient_owner));
        const Real b = static_cast<Real>(law.outward_flux(c, r.gradient_constant));
        Kokkos::atomic_add(&diagonal(p), c.area*a);
        Kokkos::atomic_add(&rhs(p), -c.area*b);
    }
};

// A test-local provider demonstrates that face-indexed data remain usable through the generic law.
struct FaceCoefficient {
    Array values;
    KOKKOS_INLINE_FUNCTION Real operator()(const bc::FaceContext<Real>& c) const
    { return values(static_cast<std::size_t>(c.face)); }
};

void assembly_case(const Exec& exec)
{
    const auto m = file_fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), file_configuration(), exec);
    Array diagonal("diagonal", 2), rhs("rhs", 2), coefficient("coefficient", 5);
    const auto law = transport::IsotropicDiffusion{FaceCoefficient{coefficient}};
    const auto assemble = [&] {
        Kokkos::deep_copy(exec, diagonal, Real{0}); Kokkos::deep_copy(exec, rhs, Real{0});
        manager.evaluate_async(AddDiffusionBoundary{diagonal, rhs, law}, law);
    };
    Kokkos::deep_copy(exec, coefficient, Real{3});
    assemble();
    auto d = snapshot(exec, diagonal), b = snapshot(exec, rhs);
    near(d(0), Real{6}, tolerance, "Dirichlet diagonal includes area exactly once");
    near(d(1), Real{3}, tolerance, "Second owner diagonal");
    near(b(0), Real{24}, tolerance, "RHS includes the outward normal-gradient flux with its sign");
    near(b(1), Real{9}, tolerance, "RHS includes prescribed outward flux");
    auto hc = Kokkos::create_mirror(coefficient);
    hc(0) = 13; hc(1) = 2; hc(2) = 4; hc(3) = 5; hc(4) = 6;
    Kokkos::deep_copy(exec, coefficient, hc);
    assemble();
    d = snapshot(exec, diagonal); b = snapshot(exec, rhs);
    near(d(0), Real{4}, tolerance, "Variable face coefficient in the first diagonal");
    near(d(1), Real{4}, tolerance, "Variable face coefficient in the second diagonal");
    near(b(0), Real{10}, tolerance, "Variable coefficient converts prescribed normal gradient to flux");
    near(b(1), Real{14}, tolerance, "Fixed flux is independent of the material coefficient");
    krona::fields::FieldStorage<Real, 1, Exec> field(
        "phi", manager.required_field_size(), krona::fields::FieldLocation::CellExtnd, exec);
    initialize_component(exec, field.view().component(0));
    manager.apply_async(field.view(), law);
    const auto host = snapshot(exec, field.view().component(0));
    near(host(6), Real{9}, tolerance, "Flux-to-gradient conversion uses the face-indexed provider");
}

void empty_labels_case(const Exec& exec)
{
    const auto m = file_fixture(false, true, true);
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), file_configuration(), exec);
    krona::fields::FieldStorage<Real, 1, Exec> field(
        "empty", 0, krona::fields::FieldLocation::CellExtnd, exec);
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{3}};
    manager.apply_async(field.view(), law);
    manager.for_each_face_async(PrescribedFluxSink{Array("empty_flux", 0)});
    int calls = 0;
    manager.for_each_patch([&](bc::PatchRange, const auto&) { ++calls; });
    equal(calls, 0, "Globally labelled patches with no local faces do not dispatch");
    equal(manager.catalog().needs_unlabelled(), true, "Global counts require an unlabelled binding on an empty rank");
    equal(manager.required_field_size(), std::size_t{0}, "Empty rank needs no boundary allocation");
    exec.fence();
}

void distributed_case(const Exec& exec)
{
    int rank = 0, ranks = 0;
    krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    equal(ranks, 2, "Distributed scalar boundary test requires two ranks");
    if (rank == 0) file_fill_case(exec); else empty_labels_case(exec);
    int local = 1, total = 0;
    krona::mpi::check(MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce");
    equal(total, 2, "Rank with no local boundary faces finishes independently");
}

template<class Mutation>
void reject_with_manager(const Exec& exec, Mutation mutation)
{
    auto m = file_fixture();
    auto config = file_configuration();
    mutation(m, config);
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), config, exec);
}

void wrong_name_case(const Exec& exec)
{ reject_with_manager(exec, [](auto&, auto& c) { c.patches[0].name = "wrong"; }); }

void scattered_case(const Exec& exec)
{ reject_with_manager(exec, [](auto& m, auto&) { m.face_patch = {-1, 0, 1, 0, 2}; }); }

void invalid_geometry_case(const Exec& exec)
{ reject_with_manager(exec, [](auto& m, auto&) { m.face_normal_x[1] = -1; }); }

void unlabelled_case(const Exec& exec)
{
    auto m = file_fixture(false, true);
    auto config = file_configuration(); config.unlabelled.reset();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), config, exec);
}

void unknown_type_case(const Exec&)
{ (void)scalar::parse_config("[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.phi]\ntype='INLET_TOTAL_TP'\n", "phi"); }

void typo_case(const Exec&)
{ (void)scalar::parse_config("[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.phi]\ntype='FIXED_VALUE'\nvalue=3\nvlaue=4\n", "phi"); }

void duplicate_case(const Exec&)
{
    const std::string entry = "[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.phi]\ntype='ZERO_GRADIENT'\n";
    (void)scalar::parse_config(entry + entry, "phi");
}

void missing_scalar_case(const Exec&)
{ (void)scalar::parse_config("[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.other]\ntype='ZERO_GRADIENT'\n", "phi"); }

void short_field_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    configure(manager, false);
    manager.apply_component_async(Array("short_field", manager.required_field_size() - 1));
}

void short_correction_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    configure(manager, false);
    Array field("field", manager.required_field_size()), short_gradient("short_gradient", 1);
    manager.apply_component_async(field, scalar::NoFluxLaw{},
        bc::GradientCorrection{short_gradient, short_gradient, short_gradient});
}

void missing_law_empty_case(const Exec& exec)
{
    const auto m = file_fixture(false, true, true);
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {.data = selection}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), file_configuration(), exec);
    manager.apply_component_async(Array("empty", 0));
}

void zero_coefficient_case(const Exec& exec)
{
    // Probe the actual assertion macro rather than assuming a particular Kokkos build configuration.
    bool assertions_enabled = false;
    KOKKOS_ASSERT((assertions_enabled = true));
    if (!assertions_enabled) {
        std::cout << "KRONA_TEST_SKIP_NO_KOKKOS_ASSERT\n";
        return;
    }
    Array result("result", 1);
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{0}};
    Kokkos::parallel_for("test/reject_zero_coefficient", Kokkos::RangePolicy<Exec>(exec, 0, 1),
        KOKKOS_LAMBDA(int) {
            bc::FaceContext<Real> c;
            result(0) = static_cast<Real>(law.normal_gradient(c, Real{1}));
        });
    exec.fence();
}

void missing_law_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    configure(manager, true);
    Array field("field", 7);
    manager.apply_component_async(field);
}

void missing_binding_case(const Exec& exec)
{
    const auto m = fixture();
    krona::mesh::compute::MeshStorage<Exec> mesh(m, {}, exec);
    scalar::Manager<Exec, Real> manager(m, mesh.view(), exec);
    manager.set(0, scalar::FixedValue<Real>{7});
    manager.set(1, scalar::NormalGradient<Real>{0});
    manager.set_unlabelled(scalar::NormalGradient<Real>{0});
    // Label 2 is locally empty, but still needs an explicit binding.
    manager.for_each_patch([](bc::PatchRange, const auto&) {});
}

} // namespace

int main(int argc, char** argv)
{
    constexpr std::array cases{Case{"reconstruction", reconstruction_case}, Case{"visitation", visitation_case},
        Case{"custom", custom_case}, Case{"relations", relations_case}, Case{"empty", empty_case},
        Case{"independent_value", independent_value_case},
        Case{"boundary_gradients", boundary_gradients_case}, Case{"config", config_case},
        Case{"file_fill", file_fill_case}, Case{"layouts", layouts_case}, Case{"affine", affine_case},
        Case{"assembly", assembly_case}, Case{"empty_labels", empty_labels_case}, Case{"distributed", distributed_case},
        Case{"missing_law", missing_law_case}, Case{"missing_binding", missing_binding_case},
        Case{"wrong_name", wrong_name_case}, Case{"scattered", scattered_case}, Case{"unlabelled", unlabelled_case},
        Case{"unknown_type", unknown_type_case}, Case{"typo", typo_case}, Case{"duplicate", duplicate_case},
        Case{"invalid_geometry", invalid_geometry_case}, Case{"missing_scalar", missing_scalar_case},
        Case{"short_field", short_field_case}, Case{"short_correction", short_correction_case},
        Case{"missing_law_empty", missing_law_empty_case}, Case{"zero_coefficient", zero_coefficient_case}};
    return krona::test::run(argc, argv, "scalar_boundary", cases);
}
