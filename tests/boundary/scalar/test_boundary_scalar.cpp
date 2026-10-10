#include "common/test_support.hpp"

#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

#include <array>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

namespace {
using namespace krona::test;
namespace bc = krona::boundary::scalar;
using Calc = krona::ComputeReal;
using Space = typename Exec::memory_space;
using Field = krona::fields::FieldStorage<Real, 1, Exec>;
using MeshPart = krona::mesh::MeshPart;
using MeshStorage = krona::mesh::compute::MeshStorage<Exec>;
using DeviceVector = Kokkos::View<Real*, Space>;
constexpr auto selection = krona::mesh::compute::MeshData::FaceCells | krona::mesh::compute::MeshData::Geometry;

Real tolerance() { return Real{256} * std::numeric_limits<Real>::epsilon(); }

/// @brief Small indexing fixture: two owned cells, one MPI ghost, and three contiguous patches.
MeshPart fixture(bool skew = false, bool unlabelled = false, bool empty = false)
{
    MeshPart m;
    m.faces_grouped = true;
    m.patches = {{"fixed", "inlet", "BCInflow"}, {"gradient", "wall", "BCWall"}, {"flux", "wall", "BCWall"}};
    m.n_bfaces_g = unlabelled ? 5 : 4;
    m.n_labeled_bfaces_g = 4;
    if (empty) return m;
    m.n_own = 2; m.n_cells = 3;
    m.n_faces = unlabelled ? 6 : 5;
    m.n_internal_faces = 1; m.n_inner_faces = 1;
    m.cell_centroid_x = {0, 2, 4}; m.cell_centroid_y = {0, 0, 0}; m.cell_centroid_z = {0, 0, 0};
    m.cell_volume = {1, 1, 1};
    m.face_owner = {0, 0, 1, 0, 1}; m.face_neigh = {1, -1, -1, -1, -1};
    m.face_patch = {-1, 0, 0, 1, 2};
    m.face_centroid_x = {1, 1, 3, -1, 4};
    m.face_centroid_y = {0, skew ? 0.2 : 0, skew ? -0.3 : 0, skew ? 0.25 : 0, skew ? -0.4 : 0};
    m.face_centroid_z = {0, skew ? 0.1 : 0, skew ? -0.15 : 0, skew ? 0.2 : 0, skew ? -0.5 : 0};
    m.face_normal_x = {1, 1, 1, -1, 1}; m.face_normal_y = {0, 0, 0, 0, 0}; m.face_normal_z = {0, 0, 0, 0, 0};
    m.face_area = {1, 2, 1, 1, 1};
    if (unlabelled) {
        m.face_owner.push_back(0); m.face_neigh.push_back(-1); m.face_patch.push_back(-1);
        m.face_centroid_x.push_back(0); m.face_centroid_y.push_back(1); m.face_centroid_z.push_back(0);
        m.face_normal_x.push_back(0); m.face_normal_y.push_back(1); m.face_normal_z.push_back(0); m.face_area.push_back(1);
    }
    return m;
}

bc::Configuration configuration()
{ return bc::read_config(KRONA_BOUNDARY_TEST_DATA, "temperature"); }

template<class View>
auto snapshot(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    complete(exec, [&] { Kokkos::deep_copy(exec, host, view); });
    return host;
}

template<class View>
void initialize(const Exec& exec, View view)
{
    auto host = Kokkos::create_mirror(view);
    for (std::size_t i = 0; i < host.extent(0); ++i) host(i) = Real{-99};
    if (host.extent(0) >= 3) { host(0) = Real{7}; host(1) = Real{11}; host(2) = Real{42}; }
    complete(exec, [&] { Kokkos::deep_copy(exec, view, host); });
}

void relations(const Exec&)
{
    bc::Context<Calc> c;
    c.distance = 2; c.diffusivity = 3; c.tangential_increment = static_cast<Calc>(0.75);
    const auto value = bc::FixedValue<Calc>{5}(c);
    near(value.face_value(Calc{3}), Calc{5}, Calc{0}, "Dirichlet face value");
    near(value.outward_flux(Calc{3}), static_cast<Calc>(-1.875), Calc{0}, "Dirichlet corrected outward flux");
    const auto gradient = bc::NormalGradient<Calc>{-4}(c);
    near(gradient.face_value(Calc{3}), static_cast<Calc>(-4.25), Calc{0}, "Neumann face reconstruction");
    near(gradient.outward_flux(Calc{3}), Calc{12}, Calc{0}, "Outward derivative and flux have opposite signs");
    const auto flux = bc::FixedFlux<Calc>{6}(c);
    near(flux.face_value(Calc{3}), static_cast<Calc>(-0.25), Calc{0}, "Prescribed flux face reconstruction");
    near(flux.outward_flux(Calc{3}), Calc{6}, Calc{0}, "Flux is a density, before multiplication by area");
    c.diffusivity = 0;
    near(bc::NormalGradient<Calc>{2}(c).outward_flux(Calc{3}), Calc{0}, Calc{0}, "Zero diffusion is valid for Neumann");
}

void config(const Exec& exec)
{
    std::ifstream file{KRONA_BOUNDARY_TEST_DATA};
    const std::string text{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    const auto parsed = bc::parse_config(text, "temperature");
    const auto tracer = bc::parse_config(text, "tracer");
    equal(parsed.patches.size(), std::size_t{3}, "Shared template metadata is accepted");
    equal(parsed.patches[0].condition.parameter, 5.0, "Select field-specific value");
    equal(tracer.patches[0].condition.parameter, 0.25, "Other scalar sections remain independent");
    check(tracer.patches[1].condition.kind == bc::Kind::NormalGradient, "ZERO_GRADIENT maps to Neumann");
    auto m = fixture();
    MeshStorage mesh{m, {.data = selection}, exec};
    auto shuffled = parsed;
    std::swap(shuffled.patches.front(), shuffled.patches.back());
    bc::Manager<> manager{m, mesh.view(), shuffled, exec};
    equal(manager.catalog().patches()[0].range.begin, Index{1}, "Absolute patch begin");
    equal(manager.catalog().patches()[0].range.end, Index{3}, "Absolute patch end");
    check(manager.catalog().unlabelled().empty(), "Internal face_patch=-1 is not an exterior patch");
}

void fill(const Exec& exec)
{
    auto m = fixture(false, true);
    MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), configuration(), exec};
    Field field{"phi", manager.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    initialize(exec, field.view().component(0));
    manager.apply_async(field.view(), bc::UniformDiffusivity<Calc>{3});
    auto host = snapshot(exec, field.view().component(0));
    near(host(0), Real{7}, tolerance(), "Owned cell unchanged");
    near(host(1), Real{11}, tolerance(), "Second owned cell unchanged");
    near(host(2), Real{42}, tolerance(), "MPI ghost unchanged");
    near(host(3), Real{5}, tolerance(), "First patch face");
    near(host(4), Real{5}, tolerance(), "Second patch face");
    near(host(5), Real{5}, tolerance(), "Outward normal gradient extrapolation");
    near(host(6), Real{7}, tolerance(), "Prescribed outward flux extrapolation");
    near(host(7), Real{7}, tolerance(), "Explicit unlabelled zero-gradient condition");
    // Parameters are copied into submitted kernels, allowing host-side updates between submissions.
    manager.set(0, bc::FixedValue<Calc>{8});
    manager.apply_async(field.view(), bc::UniformDiffusivity<Calc>{3});
    host = snapshot(exec, field.view().component(0));
    near(host(3), Real{8}, tolerance(), "Updated condition is used on subsequent apply");
}

template<class Layout>
void layout(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), configuration(), exec};
    krona::fields::FieldStorage<Real, 3, Exec, Layout> field{
        "state", manager.required_field_size()+2, krona::fields::FieldLocation::Custom, exec};
    complete(exec, [&] { field.fill(exec, Real{-17}); });
    auto view = field.view();
    initialize(exec, view.component(1));
    manager.apply_component_async(view.component(1), bc::UniformDiffusivity<Calc>{3});
    const auto host = snapshot(exec, view.kokkos_view());
    near(host(3, 1), Real{5}, tolerance(), "Selected component boundary value");
    near(host(6, 1), Real{7}, tolerance(), "Selected component flux reconstruction");
    near(host(3, 0), Real{-17}, tolerance(), "Other component unchanged");
    near(host(3, 2), Real{-17}, tolerance(), "Third component unchanged");
    near(host(7, 1), Real{-99}, tolerance(), "Trailing storage unchanged");
}

void layouts(const Exec& exec) { layout<Kokkos::LayoutLeft>(exec); layout<Kokkos::LayoutRight>(exec); }

struct AffineValue {
    KOKKOS_INLINE_FUNCTION bc::Relation<Calc> operator()(const bc::Context<Calc>& c) const
    {
        return bc::FixedValue<Calc>{Calc{7}+Calc{2}*c.x+Calc{3}*c.y-Calc{4}*c.z+Calc{5}*c.time}(c);
    }
};

template<class Values>
struct CaptureFlux {
    Values owner_values;
    DeviceVector fluxes;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<Calc>& c, const bc::Relation<Calc>& r) const
    {
        fluxes(c.boundary_index) = static_cast<Real>(r.outward_flux(
            static_cast<Calc>(owner_values(static_cast<std::size_t>(c.owner)))));
    }
};

/// @brief An exact linear field verifies the nonorthogonal correction and spatial/time-dependent extension.
void affine(const Exec& exec)
{
    auto m = fixture(true);
    MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<Exec, Calc, AffineValue> manager{m, mesh.view(), configuration(), exec};
    manager.set(0, AffineValue{});
    Field field{"phi", manager.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    initialize(exec, field.view().component(0));
    DeviceVector gx{"gx", 2}, gy{"gy", 2}, gz{"gz", 2};
    Kokkos::deep_copy(exec, gx, Real{2}); Kokkos::deep_copy(exec, gy, Real{3}); Kokkos::deep_copy(exec, gz, Real{-4});
    const auto correction = bc::GradientCorrection{gx, gy, gz};
    // Keep owner values and prescribed face values on the same time level.
    const auto field_view = field.view();
    Kokkos::parallel_for("test/affine_time", Policy(exec, 0, 2), KOKKOS_LAMBDA(Index i) {
        field_view(static_cast<std::size_t>(i)) += Real{5};
    });
    manager.set(2, bc::FixedFlux<Calc>{-6});
    manager.apply_async(field.view(), bc::UniformDiffusivity<Calc>{3}, correction, Calc{1});
    DeviceVector fluxes{"boundary_flux", 4};
    manager.evaluate_async(CaptureFlux{field.view().component(0), fluxes},
                           bc::UniformDiffusivity<Calc>{3}, correction, Calc{1});
    const auto host = snapshot(exec, field.view().component(0));
    const auto flux_host = snapshot(exec, fluxes);
    for (Index f = m.n_inner_faces; f < m.n_faces; ++f) {
        const auto face = static_cast<std::size_t>(f);
        const Real exact = static_cast<Real>(12+2*m.face_centroid_x[face]+3*m.face_centroid_y[face]-4*m.face_centroid_z[face]);
        near(host(mesh.view().sizes.n_cells + face - static_cast<std::size_t>(m.n_inner_faces)),
             exact, Real{16}*tolerance(), "Corrected boundary reproduces a linear field at face centroid");
        const Real exact_flux = static_cast<Real>(-3*(2*m.face_normal_x[face]+3*m.face_normal_y[face]-4*m.face_normal_z[face]));
        near(flux_host(face-static_cast<std::size_t>(m.n_inner_faces)), exact_flux,
             Real{16}*tolerance(), "Corrected Dirichlet and Neumann fluxes reproduce the exact linear derivative");
    }
}

struct AddDiffusionBoundary {
    DeviceVector diagonal, rhs;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<Calc>& c, const bc::Relation<Calc>& relation) const
    {
        const auto owner = static_cast<std::size_t>(c.owner);
        Kokkos::atomic_add(&diagonal(owner), static_cast<Real>(c.area * relation.flux_owner));
        Kokkos::atomic_add(&rhs(owner), static_cast<Real>(-c.area * relation.flux_constant));
    }
};

/// @brief Uses the public device sink to assemble diagonal/RHS contributions without a second coefficient cache.
void assembly(const Exec& exec)
{
    auto m = fixture();
    MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), configuration(), exec};
    DeviceVector diagonal{"diagonal", 2}, rhs{"rhs", 2}, diffusivity{"diffusivity", 5};
    Kokkos::deep_copy(exec, diagonal, Real{0}); Kokkos::deep_copy(exec, rhs, Real{0});
    Kokkos::deep_copy(exec, diffusivity, Real{3});
    manager.evaluate_async(AddDiffusionBoundary{diagonal, rhs}, bc::FaceDiffusivity{diffusivity});
    const auto d = snapshot(exec, diagonal), b = snapshot(exec, rhs);
    near(d(0), Real{6}, tolerance(), "Dirichlet diagonal with area=2");
    near(d(1), Real{3}, tolerance(), "Dirichlet diagonal with area=1");
    near(b(0), Real{24}, tolerance(), "Dirichlet RHS minus prescribed outward Neumann flux");
    near(b(1), Real{9}, tolerance(), "Dirichlet RHS minus prescribed outward flux");
    // Exercise variable per-face coefficients in the fill path as well.
    Field field{"phi", manager.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    initialize(exec, field.view().component(0));
    manager.apply_async(field.view(), bc::FaceDiffusivity{diffusivity});
    const auto host = snapshot(exec, field.view().component(0));
    near(host(6), Real{7}, tolerance(), "Face-indexed diffusivity is used for prescribed flux");
}

void empty(const Exec& exec)
{
    auto m = fixture(false, false, true);
    MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), configuration(), exec};
    Field field{"phi", 0, krona::fields::FieldLocation::Custom, exec};
    manager.apply_async(field.view(), bc::UniformDiffusivity<Calc>{3});
    complete(exec, [] {});
    equal(manager.required_field_size(), std::size_t{0}, "Empty partition has no boundary storage");
    int calls = 0;
    manager.for_each_patch([&](krona::boundary::PatchRange, const auto&) { ++calls; });
    equal(calls, 0, "Empty local patches do not launch work");
    // Also accept a globally empty catalogue without labelled patches or configuration entries.
    auto no_patches = m;
    no_patches.patches.clear(); no_patches.n_bfaces_g = 0; no_patches.n_labeled_bfaces_g = 0;
    MeshStorage no_patch_mesh{no_patches, {.data = selection}, exec};
    bc::Manager<> no_patch_manager{no_patches, no_patch_mesh.view(),
                                  bc::parse_config("boundary_condition = []", "phi"), exec};
    no_patch_manager.apply_async(field.view());
    complete(exec, [] {});
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
    mutation(m, config);
    MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), config, exec};
    Field field{"phi", manager.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    manager.apply_async(field.view(), bc::UniformDiffusivity<Calc>{3});
    complete(exec, [] {});
}

void reject_missing(const Exec& exec)
{
    auto m = fixture(); MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), exec};
    manager.for_each_patch([](krona::boundary::PatchRange, const auto&) {});
}
void reject_wrong_name(const Exec& exec)
{ reject_with_manager(exec, [](auto&, auto& c) { c.patches[0].name = "wrong"; }); }
void reject_scattered(const Exec& exec)
{ reject_with_manager(exec, [](auto& m, auto&) { m.face_patch = {-1, 0, 1, 0, 2}; }); }
void reject_invalid_geometry(const Exec& exec)
{ reject_with_manager(exec, [](auto& m, auto&) { m.face_normal_x[1] = -1; }); }
void reject_unlabelled(const Exec& exec)
{
    auto m = fixture(false, true); MeshStorage mesh{m, {.data = selection}, exec};
    auto c = configuration(); c.unlabelled.reset();
    bc::Manager<> manager{m, mesh.view(), c, exec};
}
void reject_zero_diffusivity(const Exec& exec)
{
    auto m = fixture(); MeshStorage mesh{m, {.data = selection}, exec};
    bc::Manager<> manager{m, mesh.view(), configuration(), exec};
    Field field{"phi", manager.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    manager.apply_async(field.view(), bc::UniformDiffusivity<Calc>{0});
}
void reject_unknown_type(const Exec&)
{ (void)bc::parse_config("[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.phi]\ntype='INLET_TOTAL_TP'\n", "phi"); }
void reject_typo(const Exec&)
{ (void)bc::parse_config("[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.phi]\ntype='FIXED_VALUE'\nvalue=3\nvlaue=4\n", "phi"); }
void reject_duplicate(const Exec&)
{
    const std::string entry = "[[boundary_condition]]\npatch_id=0\nname='fixed'\n[boundary_condition.scalar.phi]\ntype='ZERO_GRADIENT'\n";
    (void)bc::parse_config(entry+entry, "phi");
}

constexpr std::array cases{
    Case{"relations", relations}, Case{"config", config}, Case{"fill", fill}, Case{"layouts", layouts},
    Case{"affine", affine}, Case{"assembly", assembly}, Case{"empty", empty}, Case{"distributed", distributed},
    Case{"reject_missing", reject_missing}, Case{"reject_wrong_name", reject_wrong_name}, Case{"reject_scattered", reject_scattered},
    Case{"reject_unlabelled", reject_unlabelled}, Case{"reject_zero_diffusivity", reject_zero_diffusivity},
    Case{"reject_unknown_type", reject_unknown_type}, Case{"reject_typo", reject_typo}, Case{"reject_duplicate", reject_duplicate},
    Case{"reject_invalid_geometry", reject_invalid_geometry}};
} // namespace

int main(int argc, char** argv)
{ return krona::test::run(argc, argv, "boundary.scalar", cases); }
