#include <common/test_support.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/fields/field_storage.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace { 
using namespace krona;
namespace td = transport::scalar::diffusion;
namespace ts = transport::scalar;
namespace bc = boundary::scalar;
namespace gr = numerics::gradient;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using Real = ComputeReal;
using test::check;

#include "fixture.hpp" 

template<class View>
auto snapshot(View view, const Exec& exec)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view);
    exec.fence("Diffusion core test snapshot");
    return host;
}

struct Attached {
    mesh::device::MeshStorage<Exec> mesh;
    mesh::device::AuxConnectivityStorage<Exec> connectivity;
    mesh::device::AuxGeometryStorage<Exec> geometry;
    td::DiffusionPlan<Exec> plan;
    Attached(const Fixture& f, mpi::CommView comm, const Exec& exec, bool orthogonal)
        : mesh(f.host, {}, exec), connectivity(f.connectivity, exec), geometry(f.geometry, exec),
          plan(mesh.view(), connectivity.view(), geometry.view(), comm, exec, {orthogonal, 1e-6}) {}
};

KOKKOS_INLINE_FUNCTION Real affine(Real x, Real y, Real z, Real time)
{ return Real{2}+Real(0.7)*x-Real(1.1)*y+Real(0.6)*z+time; }

struct ValueProfile {
    bool invalid = false;
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const
    { return invalid ? Kokkos::quiet_NaN_v<Real> : affine(c.x, c.y, c.z, c.time); }
};
struct GradientProfile {
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const
    { return Real(0.7)*c.nx-Real(1.1)*c.ny+Real(0.6)*c.nz; }
};
struct FluxProfile {
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const
    { return -Real{2}*GradientProfile{}(c); }
};
struct Robin {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c, Real increment) const
    {
        const Real prescribed = ValueProfile{}(c)+GradientProfile{}(c);
        const Real denominator = Real{1}+c.distance;
        return {Real{1}/denominator, (c.distance*prescribed+increment)/denominator,
                -Real{1}/denominator, (prescribed-increment)/denominator};
    }
};
using ProfileValue = bc::FixedValue<Real, ValueProfile>;
using ProfileGradient = bc::NormalGradient<Real, GradientProfile>;
using ProfileFlux = bc::FixedFlux<Real, FluxProfile>;
using Boundary = bc::Manager<Exec, Real, ProfileValue, ProfileGradient, ProfileFlux, Robin>;

void configure(Boundary& boundary, const Fixture& f, bool mixed)
{
    for (std::size_t p = 0; p < f.host.patches.size(); ++p)
        boundary.set(static_cast<mesh::PatchId>(p), ProfileValue{ValueProfile{}});
    if (mixed) {
        boundary.set(0, ProfileGradient{GradientProfile{}});
        boundary.set(1, ProfileFlux{FluxProfile{}});
        boundary.set(2, Robin{});
    }
    boundary.finalize();
}

// A real native-entry SystemView, independent of an algebra wrapper. Each row has
// one diagonal and one entry per unique local neighbour; duplicate faces share entries.
template<class R>
struct NativeSystem {
    struct Entries { Kokkos::View<std::size_t*, Space> diagonal, owner_entries, neighbour_entries; } entries;
    Kokkos::View<R***, Kokkos::LayoutRight, Space> blocks;
    Kokkos::View<R**, Kokkos::LayoutRight, Space> right;
    std::vector<std::size_t> offsets{0};
    std::vector<LocalIndex> columns;
    std::size_t owned;

    NativeSystem(const Fixture& f, const Exec& exec) : owned(static_cast<std::size_t>(f.host.n_own))
    {
        const auto& m = f.host;
        for (LocalIndex cell = 0; cell < m.n_own; ++cell) {
            std::vector<LocalIndex> row{cell};
            for (LocalIndex face = 0; face < m.n_inner_faces; ++face) {
                const auto i = static_cast<std::size_t>(face);
                if (m.face_owner[i] == cell) row.push_back(m.face_neigh[i]);
                else if (m.face_neigh[i] == cell) row.push_back(m.face_owner[i]);
            }
            std::sort(row.begin(), row.end()); row.erase(std::unique(row.begin(), row.end()), row.end());
            columns.insert(columns.end(), row.begin(), row.end()); offsets.push_back(columns.size());
        }
        entries = {Kokkos::View<std::size_t*, Space>("diagonal", owned),
                   Kokkos::View<std::size_t*, Space>("owner entries", m.n_inner_faces),
                   Kokkos::View<std::size_t*, Space>("neighbour entries", m.n_internal_faces)};
        const auto find = [&](LocalIndex row, LocalIndex col) {
            const auto i = static_cast<std::size_t>(row);
            auto first = columns.begin()+static_cast<std::ptrdiff_t>(offsets[i]);
            const auto last = columns.begin()+static_cast<std::ptrdiff_t>(offsets[i+1]);
            const auto p = std::find(first, last, col); check(p != last, "missing test graph entry");
            return static_cast<std::size_t>(p-columns.begin());
        };
        auto diagonal = Kokkos::create_mirror(entries.diagonal);
        auto owner = Kokkos::create_mirror(entries.owner_entries);
        auto neighbour = Kokkos::create_mirror(entries.neighbour_entries);
        for (LocalIndex i = 0; i < m.n_own; ++i) diagonal(i) = find(i, i);
        for (LocalIndex fidx = 0; fidx < m.n_inner_faces; ++fidx) {
            const auto k = static_cast<std::size_t>(fidx);
            owner(fidx) = find(m.face_owner[k], m.face_neigh[k]);
            if (fidx < m.n_internal_faces) neighbour(fidx) = find(m.face_neigh[k], m.face_owner[k]);
        }
        Kokkos::deep_copy(exec, entries.diagonal, diagonal); Kokkos::deep_copy(exec, entries.owner_entries, owner);
        Kokkos::deep_copy(exec, entries.neighbour_entries, neighbour);
        blocks = decltype(blocks)("native blocks", columns.size(), 2, 2);
        right = decltype(right)("strided RHS", owned+1, 3);
        reset(exec);
    }
    auto view() const
    {
        const auto values = Kokkos::subview(blocks, Kokkos::ALL(), 1, 1);
        const auto rhs = Kokkos::subview(right, std::pair<std::size_t, std::size_t>{0, owned}, 1);
        return ts::SystemView{values, rhs, entries};
    }
    void reset(const Exec& exec)
    {
        Kokkos::deep_copy(exec, blocks, R{7}); Kokkos::deep_copy(exec, right, R{7});
        Kokkos::deep_copy(exec, view().values, R{0}); Kokkos::deep_copy(exec, view().rhs, R{0});
    }
};

template<class Storage = double>
void run_case(const Exec& exec, bool skew, bool mixed = false, bool empty = false, bool duplicate = false)
{
    const auto comm = mpi::CommView::world();
    auto f = skew ? skew_grid(comm) : bar(comm, duplicate, false, empty);
    if (!skew) {
        f.geometry.active_mask = f.geometry.active_mask | mesh::AuxGeomType::FaceNonOrthogonalityVector;
        f.geometry.face_non_ortho_x.assign(static_cast<std::size_t>(f.host.n_faces), 0);
        f.geometry.face_non_ortho_y = f.geometry.face_non_ortho_x; f.geometry.face_non_ortho_z = f.geometry.face_non_ortho_x;
    }
    Attached a(f, comm, exec, !skew);
    boundary::BoundaryLayout<Exec> layout(f.host, a.mesh.view(), exec);
    Boundary boundary(layout); configure(boundary, f, mixed);
    const auto m = a.mesh.view();
    fields::FieldStorage<Storage, 3, Exec, Kokkos::LayoutRight> values("strided state", layout.required_field_size(), fields::FieldLocation::Custom, exec);
    fields::FieldStorage<Storage, 6, Exec, Kokkos::LayoutRight> gradients("strided gradients", static_cast<std::size_t>(m.sizes.n_cells), fields::FieldLocation::Custom, exec);
    const auto phi = values.view().component(1); const auto g = gradients.view();
    constexpr Real time = Real(0.25);
    Kokkos::parallel_for("test/diffusion_inputs", Kokkos::RangePolicy<Exec>(exec, 0, m.sizes.n_cells), KOKKOS_LAMBDA(int i) {
        phi(i) = static_cast<Storage>(affine(static_cast<Real>(m.geometry.cell_centroid_x(i)), static_cast<Real>(m.geometry.cell_centroid_y(i)), static_cast<Real>(m.geometry.cell_centroid_z(i)), time));
        g(i, 3) = Storage(0.7); g(i, 4) = Storage(-1.1); g(i, 5) = Storage(0.6);
    });
    const auto correction = gr::GradientCorrection{gr::cell_gradients(g, 1)};
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    const td::DiffusionOperator op(a.plan);
    NativeSystem<Storage> system(f, exec);
    ts::AssemblyStatus<Exec> status(exec);
    status.begin_async(); op.add_system_async(system.view(), boundary, status, law, correction, time); (void)status.check(comm);
    const auto coefficients = snapshot(system.view().values, exec), rhs = snapshot(system.view().rhs, exec);
    Kokkos::View<Storage*, Space> residual("residual", m.sizes.n_own);
    Kokkos::deep_copy(exec, residual, Storage{0});
    status.begin_async(); op.add_residual_async(phi, residual, boundary, status, law, correction, time); (void)status.check(comm);
    const auto result = snapshot(residual, exec);
    const auto field = snapshot(phi, exec);
    const double tolerance = std::is_same_v<Storage, float> ? 2e-4 : 2e-11;
    for (std::size_t row = 0; row < system.owned; ++row) {
        double action = static_cast<double>(rhs(row));
        for (auto k = system.offsets[row]; k < system.offsets[row+1]; ++k)
            action -= static_cast<double>(coefficients(k))*static_cast<double>(field(system.columns[k]));
        test::near(action, 0., tolerance, "affine assembled equation");
        test::near(static_cast<double>(result(row)), action, tolerance, "matrix-free versus assembled residual");
    }
    // The correction-only update must reproduce the complete RHS without changing the matrix.
    system.reset(exec);
    status.begin_async(); op.add_system_async(system.view(), boundary, status, law, time); (void)status.check(comm);
    const auto base_coefficients = snapshot(system.view().values, exec);
    status.begin_async(); op.add_correction_async(system.view().rhs, boundary, status, law, correction, time); (void)status.check(comm);
    const auto corrected_rhs = snapshot(system.view().rhs, exec);
    for (std::size_t k = 0; k < system.columns.size(); ++k)
        test::near(static_cast<double>(base_coefficients(k)), static_cast<double>(coefficients(k)), 0., "correction changed matrix coefficients");
    for (std::size_t row = 0; row < system.owned; ++row)
        test::near(static_cast<double>(corrected_rhs(row)), static_cast<double>(rhs(row)), tolerance, "correction-only RHS differs from full assembly");
    const auto blocks = snapshot(system.blocks, exec);
    const auto right = snapshot(system.right, exec);
    for (std::size_t k = 0; k < system.columns.size(); ++k)
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
            if (i != 1 || j != 1) test::near(static_cast<double>(blocks(k, i, j)), 7., 0., "unselected block coefficient changed");
    for (std::size_t row = 0; row <= system.owned; ++row) for (int c = 0; c < 3; ++c)
        if (row == system.owned || c != 1) test::near(static_cast<double>(right(row, c)), 7., 0., "unselected RHS component/tail changed");
    const auto balance = op.balance(phi, boundary, law, ts::UniformSource<Real>{0}, correction, time);
    test::near(balance.defect, 0., tolerance*static_cast<double>(comm.size()), "global affine balance");
}

void orthogonal(const Exec& e) { run_case(e, false); }
void nonorthogonal(const Exec& e) { run_case(e, true); }
void mixed_profiles(const Exec& e) { run_case(e, true, true); }
void float_storage(const Exec& e) { run_case<float>(e, true, true); }
void empty(const Exec& e) { run_case(e, false, false, true); }
void duplicate(const Exec& e) { run_case(e, false, false, false, true); }

struct ZeroProjection {
    std::size_t expected_cells;
    template<class E> void validate(std::size_t cells) const
    { mpi::require(cells == expected_cells, "Wrong cell count passed to custom correction validator"); }
    template<class C, class R> KOKKOS_INLINE_FUNCTION R interior(const C&, R, R, R) const { return R{0}; }
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return std::remove_cvref_t<decltype(c.tx)>{0}; }
};

void custom_correction(const Exec& exec)
{
    const auto comm = mpi::CommView::world(); auto f = bar(comm);
    f.geometry.active_mask = f.geometry.active_mask | mesh::AuxGeomType::FaceNonOrthogonalityVector;
    f.geometry.face_non_ortho_x.assign(static_cast<std::size_t>(f.host.n_faces), 0);
    f.geometry.face_non_ortho_y = f.geometry.face_non_ortho_x; f.geometry.face_non_ortho_z = f.geometry.face_non_ortho_x;
    Attached a(f, comm, exec, true); boundary::BoundaryLayout<Exec> layout(f.host, a.mesh.view(), exec);
    Boundary boundary(layout); configure(boundary, f, false);
    Kokkos::View<Real*, Space> field("custom field", f.host.n_cells), first("first", f.host.n_own), second("second", f.host.n_own);
    Kokkos::deep_copy(exec, field, Real{2}); Kokkos::deep_copy(exec, first, Real{0}); Kokkos::deep_copy(exec, second, Real{0});
    const td::DiffusionOperator op(a.plan); ts::AssemblyStatus<Exec> status(exec);
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    status.begin_async(); op.add_residual_async(field, first, boundary, status, law); (void)status.check(comm);
    status.begin_async(); op.add_residual_async(field, second, boundary, status, law, ZeroProjection{static_cast<std::size_t>(f.host.n_cells)}); (void)status.check(comm);
    const auto x = snapshot(first, exec), y = snapshot(second, exec);
    for (std::size_t i = 0; i < x.extent(0); ++i) test::near(x(i), y(i), Real{0}, "custom correction without a gradients member");
}

void rejection(const Exec& exec, std::string_view which)
{
    const auto comm = mpi::CommView::world(); auto f = skew_grid(comm);
    if (which == "stale_metric") f.geometry.face_normal_dist[0] *= GeometryReal{2};
    if (which == "missing_weights") f.geometry.active_mask = mesh::AuxGeomType::FaceNormalDistance |
        mesh::AuxGeomType::FaceAreaOverNormalDistance | mesh::AuxGeomType::FaceNonOrthogonalityVector;
    Attached a(f, which == "null_comm" ? mpi::CommView{} : comm, exec, false);
    boundary::BoundaryLayout<Exec> layout(f.host, a.mesh.view(), exec); Boundary boundary(layout); configure(boundary, f, false);
    if (which == "bad_profile") { boundary.set(0, ProfileValue{ValueProfile{true}}); boundary.finalize(); }
    const auto count = static_cast<std::size_t>(f.host.n_cells);
    Kokkos::View<Real*, Space> gx("gx", which == "short_gradient" ? count-1 : count), gy("gy", count), gz("gz", count), residual("residual", f.host.n_own), field("field", count);
    Kokkos::deep_copy(exec, gx, which == "nan_gradient" ? std::numeric_limits<Real>::quiet_NaN() : Real{1});
    Kokkos::deep_copy(exec, gy, Real{1}); Kokkos::deep_copy(exec, gz, Real{1}); Kokkos::deep_copy(exec, field, Real{1}); Kokkos::deep_copy(exec, residual, Real{0});
    const auto correction = gr::GradientCorrection{gr::cell_gradients(gx, gy, gz)};
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{which == "bad_coefficient" ? Real{-1} : Real{2}}};
    ts::AssemblyStatus<Exec> status(exec); status.begin_async();
    td::DiffusionOperator(a.plan).add_residual_async(field, residual, boundary, status, law, correction);
    (void)status.check(comm); check(false, "invalid diffusion input was accepted");
}
void stale_metric(const Exec& e) { rejection(e, "stale_metric"); }
void missing_weights(const Exec& e) { rejection(e, "missing_weights"); }
void null_comm(const Exec& e) { rejection(e, "null_comm"); }
void short_gradient(const Exec& e) { rejection(e, "short_gradient"); }
void nan_gradient(const Exec& e) { rejection(e, "nan_gradient"); }
void bad_coefficient(const Exec& e) { rejection(e, "bad_coefficient"); }
void bad_profile(const Exec& e) { rejection(e, "bad_profile"); }
} // namespace

int main(int argc, char** argv)
{
    const std::array cases{test::Case{"orthogonal", orthogonal}, test::Case{"nonorthogonal", nonorthogonal},
        test::Case{"mixed_profiles", mixed_profiles}, test::Case{"float_storage", float_storage}, test::Case{"empty", empty}, test::Case{"duplicate", duplicate},
        test::Case{"custom_correction", custom_correction}};
    const std::array rejected{test::Case{"stale_metric", stale_metric}, test::Case{"missing_weights", missing_weights}, test::Case{"null_comm", null_comm},
        test::Case{"short_gradient", short_gradient}, test::Case{"nan_gradient", nan_gradient}, test::Case{"bad_coefficient", bad_coefficient}, test::Case{"bad_profile", bad_profile}};
    if (argc == 2) for (const auto& c : rejected) if (c.name == argv[1]) return test::run(argc, argv, "transport.diffusion", rejected);
    return test::run(argc, argv, "transport.diffusion", cases);
}
