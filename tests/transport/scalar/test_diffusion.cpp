#include "common/test_support.hpp"

#include <krona/transport/scalar/steady_diffusion.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/linalg/bsr_matrix.hpp>

#include <array>
#include <limits>
#include <numeric>
#include <string_view>

namespace {
using namespace krona;
namespace ts = transport::scalar;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;
using krona::test::check;

struct Fixture {
    mesh::MeshPart host;
    mesh::MeshAuxConnectivity connectivity;
    mesh::MeshAuxGeometry geometry;
    int first = 0, total = 0;
};

Fixture bar(mpi::CommView comm, bool duplicate = false, bool empty_last = false, bool all_empty = false)
{
    Fixture f;
    auto& m = f.host;
    const int active = all_empty ? 0 : (empty_last && comm.size() > 1 ? comm.size()-1 : comm.size());
    const int count = comm.rank() < active ? 3 : 0;
    f.first = comm.rank()*3; f.total = active*3;
    m.rank = comm.rank(); m.nprocs = comm.size(); m.n_own = count;
    m.n_cells_g = static_cast<std::uint64_t>(f.total);
    m.faces_grouped = true;
    m.patches.resize(2); m.patches[0].name = "left"; m.patches[1].name = "right";
    m.n_bfaces_g = f.total == 0 ? 0 : 2; m.n_labeled_bfaces_g = m.n_bfaces_g;
    for (int i = 0; i < count; ++i) m.cell_centroid_x.push_back(static_cast<GeometryReal>(f.first+i)+GeometryReal{0.5});
    for (int peer : {comm.rank()-1, comm.rank()+1}) {
        if (count == 0 || peer < 0 || peer >= active) continue;
        m.nb_ranks.push_back(peer);
        m.send_owned_local.push_back(peer < comm.rank() ? 0 : count-1);
        m.recv_ghost_local.push_back(static_cast<LocalIndex>(m.cell_centroid_x.size()));
        m.cell_centroid_x.push_back(static_cast<GeometryReal>(peer < comm.rank() ? f.first-1 : f.first+count)+GeometryReal{0.5});
        m.send_offsets.push_back(m.send_owned_local.size()); m.recv_offsets.push_back(m.recv_ghost_local.size());
    }
    m.n_cells = static_cast<LocalIndex>(m.cell_centroid_x.size());
    m.cell_centroid_y.assign(m.cell_centroid_x.size(), 0); m.cell_centroid_z.assign(m.cell_centroid_x.size(), 0);
    m.cell_volume.assign(m.cell_centroid_x.size(), 1);
    const auto face = [&](LocalIndex owner, LocalIndex neighbour, GeometryReal x, GeometryReal normal, GeometryReal area, mesh::PatchId patch) {
        m.face_owner.push_back(owner); m.face_neigh.push_back(neighbour); m.face_patch.push_back(patch);
        m.face_centroid_x.push_back(x); m.face_centroid_y.push_back(0); m.face_centroid_z.push_back(0);
        m.face_normal_x.push_back(normal); m.face_normal_y.push_back(0); m.face_normal_z.push_back(0); m.face_area.push_back(area);
        const GeometryReal distance = neighbour < 0 ? GeometryReal{0.5} : GeometryReal{1};
        f.geometry.face_normal_dist.push_back(distance);
        f.geometry.face_area_over_normal_dist.push_back(area/distance);
        f.geometry.face_interp_weight.push_back(neighbour < 0 ? 0 : GeometryReal{0.5});
    };
    for (LocalIndex cell = 0; cell+1 < count; ++cell)
        for (int copy = 0; copy < (duplicate ? 2 : 1); ++copy)
            face(cell, cell+1, static_cast<GeometryReal>(f.first+cell+1), 1, duplicate ? GeometryReal{0.5} : GeometryReal{1}, -1);
    m.n_internal_faces = static_cast<LocalIndex>(m.face_owner.size());
    for (std::size_t p = 0; p < m.nb_ranks.size(); ++p) {
        const bool left = m.nb_ranks[p] < comm.rank();
        face(left ? 0 : count-1, m.recv_ghost_local[p], static_cast<GeometryReal>(left ? f.first : f.first+count), left ? -1 : 1, 1, -1);
    }
    m.n_inner_faces = static_cast<LocalIndex>(m.face_owner.size());
    if (count != 0 && comm.rank() == 0) face(0, -1, 0, -1, 1, 0);
    if (count != 0 && comm.rank() == active-1) face(count-1, -1, static_cast<GeometryReal>(f.total), 1, 1, 1);
    m.n_faces = static_cast<LocalIndex>(m.face_owner.size());
    auto& c = f.connectivity;
    c.active_mask = mesh::AuxConnType::CellFaces; c.n_faces = m.n_faces; c.n_cells_own = m.n_own;
    c.cell_faces_offsets = {0};
    for (LocalIndex cell = 0; cell < count; ++cell) {
        for (LocalIndex entry = 0; entry < m.n_faces; ++entry)
            if (m.face_owner[static_cast<std::size_t>(entry)] == cell || m.face_neigh[static_cast<std::size_t>(entry)] == cell)
                c.cell_faces.push_back(entry);
        c.cell_faces_offsets.push_back(c.cell_faces.size());
    }
    f.geometry.active_mask = mesh::AuxGeomType::FaceNormalDistance | mesh::AuxGeomType::FaceAreaOverNormalDistance |
                             mesh::AuxGeomType::FaceInterpolationWeights;
    f.geometry.n_faces = m.n_faces;
    return f;
}

struct Attached {
    mesh::compute::MeshStorage<Exec> mesh;
    mesh::compute::AuxConnectivityStorage<Exec> connectivity;
    mesh::compute::AuxGeometryStorage<Exec> geometry;
    ts::DiffusionPlan<Exec, Real> plan;
    Attached(const Fixture& f, mpi::CommView comm, Exec exec)
        : mesh(f.host, {}, exec), connectivity(f.host, f.connectivity, exec), geometry(f.host, f.geometry, exec),
          plan(f.host, mesh.view(), connectivity.view(), geometry.view(), comm, exec) {}
};

struct Robin {
    Real a = 1, b = 1, value = 0;
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c, Real increment = 0) const {
        const Real denominator = b+a*c.distance;
        return {b/denominator, (c.distance*value+b*increment)/denominator,
                -a/denominator, (value-a*increment)/denominator};
    }
};
struct NegativeRelation {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>&, Real = 0) const
    { return {1, 0, 1, 0}; }
};
using Boundary = bc::Manager<Exec, Real, Robin, NegativeRelation>;

ts::SteadyDiffusionOptions settings(bool no_iterations = false) {
    ts::SteadyDiffusionOptions o;
    o.linear.method = "FGMRES"; o.linear.relative_tolerance = 1e-10;
    o.linear.preconditioner.type = "RELAXATION";
    if (no_iterations) o.linear.maximum_iterations = 0;
    return o;
}

template<class Storage, class Layout>
auto snapshot(const fields::FieldStorage<Storage, 3, Exec, Layout>& field, Exec exec) {
    auto result = field.create_host_buffer(); Kokkos::deep_copy(exec, result, field.view().kokkos_view()); exec.fence("test snapshot"); return result;
}

template<class Storage = double, class Layout = Kokkos::LayoutLeft, class Prec = double, class Scalar = double>
void linear_case(const Exec& exec, int kind = 0, bool duplicate = false, bool empty_last = false, bool all_empty = false)
{
    const auto comm = mpi::CommView::world();
    const auto f = bar(comm, duplicate, empty_last, all_empty);
    Attached a(f, comm, exec);
    Boundary boundary(f.host, a.mesh.view(), exec);
    boundary.set(0, bc::FixedValue<Real>{1});
    const Real total = static_cast<Real>(f.total);
    if (kind == 1) boundary.set(1, bc::NormalGradient<Real>{2});
    else if (kind == 2) boundary.set(1, bc::FixedFlux<Real>{-4});
    else if (kind == 3) boundary.set(1, Robin{1, 1, Real{3}+Real{2}*total});
    else boundary.set(1, bc::FixedValue<Real>{Real{1}+Real{2}*total});
    fields::FieldStorage<Storage, 3, Exec, Layout> field("test/temperature", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
    field.fill(exec, Storage{-77});
    auto options = settings();
    if constexpr (std::is_same_v<Scalar, float>) options.linear.relative_tolerance = 1e-5;
    ts::SteadyDiffusion<Storage, Scalar, Exec, Real, Prec> solver(a.plan, field.view().component(1), options);
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    const auto result = solver.solve_owned(boundary, law);
    check(result.converged, "linear bar did not converge"); solver.solution().require_committed();
    auto h = snapshot(field, exec);
    const double tolerance = std::is_same_v<Scalar, float> ? 2e-4 : (std::is_same_v<Storage, float> ? 2e-5 : 1e-8);
    for (std::size_t i = 0; i < field.size(); ++i) {
        test::near(static_cast<double>(h(i, 0)), -77.0, 0.0, "another component was overwritten");
        test::near(static_cast<double>(h(i, 2)), -77.0, 0.0, "another component was overwritten");
        if (i < static_cast<std::size_t>(f.host.n_own))
            test::near(static_cast<double>(h(i, 1)), 1+2*static_cast<double>(f.host.cell_centroid_x[i]), tolerance, "owned linear solution is incorrect");
        else test::near(static_cast<double>(h(i, 1)), -77.0, 0.0, "solve_owned wrote a ghost/boundary slot");
    }
    halo::Exchange<Storage, Exec> exchange(a.mesh.view(), a.mesh.halo_plan(), comm, 17, exec);
    exchange.add_component(field.view().component(1)); exchange.prepare(); exchange.exchange();
    boundary.apply_component_async(field.view().component(1), law); boundary.execution_space().fence("test BC complete");
    h = snapshot(field, exec);
    for (std::size_t i = 0; i < static_cast<std::size_t>(f.host.n_cells); ++i)
        test::near(static_cast<double>(h(i, 1)), 1+2*static_cast<double>(f.host.cell_centroid_x[i]), tolerance, "halo publication is incorrect");
    for (LocalIndex face = f.host.n_inner_faces; face < f.host.n_faces; ++face) {
        const auto index = static_cast<std::size_t>(f.host.n_cells+face-f.host.n_inner_faces);
        test::near(static_cast<double>(h(index, 1)), 1+2*static_cast<double>(f.host.face_centroid_x[static_cast<std::size_t>(face)]), tolerance, "boundary publication is incorrect");
    }
    const auto balance = solver.balance(boundary, law);
    test::near(balance.defect, 0.0, tolerance*10, "linear bar violates global balance");
}

void source_case(const Exec& exec) {
    const auto comm = mpi::CommView::world(); auto f = bar(comm);
    // Cross-section two checks both volume integration and face-area scaling.
    for (auto& volume : f.host.cell_volume) volume *= 2;
    for (auto& area : f.host.face_area) area *= 2;
    for (auto& metric : f.geometry.face_area_over_normal_dist) metric *= 2;
    Attached a(f, comm, exec);
    Boundary boundary(f.host, a.mesh.view(), exec); boundary.set(0, bc::FixedValue<Real>{0}); boundary.set(1, bc::FixedValue<Real>{0});
    fields::FieldStorage<double, 3, Exec> field("source", boundary.required_field_size(), fields::FieldLocation::Custom, exec); field.fill(exec, 0);
    ts::SteadyDiffusion<double> solver(a.plan, field.view().component(0), settings());
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    Kokkos::View<Real*, Space> source("source density", static_cast<std::size_t>(f.host.n_own)); Kokkos::deep_copy(exec, source, Real{3});
    const ts::CellSource provider{source};
    check(solver.solve_owned(boundary, law, provider).converged, "source solve failed");
    const auto h = snapshot(field, exec);
    for (LocalIndex i = 0; i < f.host.n_own; ++i) {
        const double x = static_cast<double>(f.host.cell_centroid_x[static_cast<std::size_t>(i)]);
        // Exact discrete TPFA solution: the half-cell boundary stencil adds this 1/4 term.
        test::near(h(static_cast<std::size_t>(i), 0), 0.75*(x*(f.total-x)+0.25), 1e-8, "wrong integrated source or boundary stencil");
    }
    const auto balance = solver.balance(boundary, law, provider);
    test::near(balance.source_integral, 6.0*f.total, 1e-10, "source density was not volume integrated");
    test::near(balance.defect, 0.0, 1e-7, "source/boundary flux signs disagree");
}

void variable_case(const Exec& exec) {
    const auto comm = mpi::CommView::world(); auto f = bar(comm);
    // Unequal cell widths exercise distance-weighted interpolation, including MPI-cut orientation.
    const auto width = [](int gid) { return gid%2 == 0 ? 1.0 : 2.0; };
    const auto position = [](int gid) { return static_cast<double>(gid+gid/2); };
    std::vector<int> gids;
    for (const auto x : f.host.cell_centroid_x) gids.push_back(static_cast<int>(x-GeometryReal{0.5}));
    for (std::size_t i = 0; i < gids.size(); ++i) {
        f.host.cell_centroid_x[i] = static_cast<GeometryReal>(position(gids[i])+width(gids[i])/2);
        f.host.cell_volume[i] = static_cast<GeometryReal>(width(gids[i]));
    }
    for (std::size_t face = 0; face < f.host.face_owner.size(); ++face) {
        const auto ordinal = static_cast<int>(f.host.face_centroid_x[face]);
        f.host.face_centroid_x[face] = static_cast<GeometryReal>(position(ordinal));
        const auto owner = static_cast<std::size_t>(f.host.face_owner[face]);
        const auto neighbour = f.host.face_neigh[face];
        const double d = neighbour < 0 ? width(gids[owner])/2 : (width(gids[owner])+width(gids[static_cast<std::size_t>(neighbour)]))/2;
        f.geometry.face_normal_dist[face] = static_cast<GeometryReal>(d);
        f.geometry.face_area_over_normal_dist[face] = static_cast<GeometryReal>(1/d);
        f.geometry.face_interp_weight[face] = neighbour < 0 ? 0 : static_cast<GeometryReal>(width(gids[static_cast<std::size_t>(neighbour)])/(2*d));
    }
    Attached a(f, comm, exec);
    Boundary boundary(f.host, a.mesh.view(), exec); boundary.set(0, bc::FixedValue<Real>{0}); boundary.set(1, bc::FixedValue<Real>{1});
    fields::FieldStorage<double, 3, Exec> field("variable", boundary.required_field_size(), fields::FieldLocation::Custom, exec); field.fill(exec, 0);
    Kokkos::View<Real*, Space> coefficient("cell D", static_cast<std::size_t>(f.host.n_cells));
    auto hD = Kokkos::create_mirror(coefficient);
    for (std::size_t i = 0; i < hD.extent(0); ++i) {
        hD(i) = gids[i]%2 == 0 ? Real{1} : Real{4};
    }
    Kokkos::deep_copy(exec, coefficient, hD);
    const auto law = transport::IsotropicDiffusion{transport::InterpolatedCellCoefficient{coefficient}};
    ts::SteadyDiffusion<double> solver(a.plan, field.view().component(0), settings());
    check(solver.solve_owned(boundary, law).converged, "variable coefficient solve failed");
    double resistance = 0; for (int i = 0; i < f.total; ++i) resistance += width(i)*(i%2 == 0 ? 1.0 : 0.25);
    const auto h = snapshot(field, exec);
    for (LocalIndex i = 0; i < f.host.n_own; ++i) {
        const int gid = f.first+i; double prefix = 0; for (int j = 0; j < gid; ++j) prefix += width(j)*(j%2 == 0 ? 1.0 : 0.25);
        const double expected = (prefix+0.5*width(gid)*(gid%2 == 0 ? 1.0 : 0.25))/resistance;
        test::near(h(static_cast<std::size_t>(i), 0), expected, 1e-9, "harmonic cell coefficient violates series resistance");
    }
    test::near(solver.balance(boundary, law).defect, 0.0, 1e-8, "variable coefficient violates balance across MPI cuts");
    // Explicit arithmetic interpolation is a distinct, supported numerical policy.
    const auto arithmetic = transport::IsotropicDiffusion{transport::InterpolatedCellCoefficient<decltype(coefficient), transport::ArithmeticCoefficient>{coefficient}};
    check(solver.solve_owned(boundary, arithmetic).converged, "arithmetic coefficient solve failed");
    test::near(solver.balance(boundary, arithmetic).defect, 0.0, 1e-8, "arithmetic coefficient is not conservative");
}

void shared_case(const Exec& exec) {
    const auto comm = mpi::CommView::world(); const auto f = bar(comm); Attached a(f, comm, exec);
    Boundary left(f.host, a.mesh.view(), exec), right(f.host, a.mesh.view(), exec);
    left.set(0, bc::FixedValue<Real>{1}); left.set(1, bc::FixedValue<Real>{1+2*static_cast<Real>(f.total)});
    right.set(0, bc::FixedValue<Real>{3}); right.set(1, bc::FixedValue<Real>{3-static_cast<Real>(f.total)/2});
    fields::FieldStorage<double, 3, Exec, Kokkos::LayoutRight> field("two scalars", left.required_field_size(), fields::FieldLocation::Custom, exec); field.fill(exec, -77);
    ts::SteadyDiffusion<double> first(a.plan, field.view().component(0), settings()), second(a.plan, field.view().component(1), settings());
    check(first.operation().matrix().pattern().native().getRawPtr() == second.operation().matrix().pattern().native().getRawPtr(), "shared plan duplicated the graph");
    check(first.operation().plan().row_plan().owner_entries.data() == second.operation().plan().row_plan().owner_entries.data(), "shared plan duplicated entry maps");
    check(first.operation().plan().geometry_view().face_normal_dist.data() == a.geometry.view().face_normal_dist.data(), "shared geometry was copied");
    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    check(first.solve_owned(left, law).converged && second.solve_owned(right, law).converged, "shared scalar solve failed");
    halo::Exchange<double, Exec> exchange(a.mesh.view(), a.mesh.halo_plan(), comm, 19, exec);
    exchange.add_component(field.view().component(0)); exchange.add_component(field.view().component(1)); exchange.prepare(); exchange.exchange();
    left.apply_component_async(field.view().component(0), law); right.apply_component_async(field.view().component(1), law); exec.fence("two scalars published");
    const auto h = snapshot(field, exec);
    for (LocalIndex i = 0; i < f.host.n_cells; ++i) {
        const auto j = static_cast<std::size_t>(i); const double x = static_cast<double>(f.host.cell_centroid_x[j]);
        test::near(h(j, 0), 1+2*x, 1e-8, "first scalar publication failed");
        test::near(h(j, 1), 3-0.5*x, 1e-8, "second scalar publication failed");
        test::near(h(j, 2), -77.0, 0.0, "group exchange touched another component");
    }
    // A coefficient revision refreshes only the first equation, preserving the shared topology.
    const auto revision = first.operation().matrix().revision();
    check(first.solve_owned(left, transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{5}}).converged, "coefficient reassembly failed");
    check(first.operation().matrix().revision() > revision, "reassembly did not update matrix revision");
}

template<class Values>
KOKKOS_INLINE_FUNCTION void add_coefficient(Values values, std::size_t entry, int equation, int variable,
                                           typename Values::non_const_value_type value)
{
    if constexpr (Values::rank == 3) values(entry, equation, variable) += value;
    else values(entry) += value;
}

template<class Matrix>
void assembly_product(Matrix& matrix, const Fixture& f, mesh::compute::MeshView<Space> mesh,
    mesh::compute::AuxConnectivityView<Space> connectivity, linalg::FaceAssemblyMap<Space> entries, const Exec& exec)
{
    using Scalar = typename Matrix::scalar_type;
    const int width = matrix.block_size();
    const auto values = [&] {
        if constexpr (Matrix::block_storage) return matrix.overwrite_blocks(exec);
        else return matrix.overwrite_values(exec);
    }();
    Kokkos::deep_copy(exec, values, Scalar{0});
    const auto offsets = connectivity.cell_faces_offsets;
    const auto faces = connectivity.cell_faces;
    const auto owners = mesh.connectivity.face_owner;
    const auto inner = mesh.sizes.n_inner_faces;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    Kokkos::parallel_for("test/shared_face_assembly", Policy(exec, 0, mesh.sizes.n_own), KOKKOS_LAMBDA(LocalIndex cell) {
        const auto i = static_cast<std::size_t>(cell);
        for (int equation = 0; equation < width; ++equation) {
            add_coefficient(values, entries.diagonal(i), equation, equation, Scalar{2});
            for (auto k = offsets(i); k < offsets(i+1); ++k) {
                const auto face = faces(k);
                if (face >= inner) continue;
                const auto face_index = static_cast<std::size_t>(face);
                const auto entry = owners(face_index) == cell ? entries.owner_entries(face_index) : entries.neighbour_entries(face_index);
                for (int variable = 0; variable < width; ++variable) {
                    const auto value = static_cast<Scalar>((equation+1)*4+variable+1);
                    add_coefficient(values, entry, equation, variable, -value);
                    add_coefficient(values, entries.diagonal(i), equation, variable, value);
                }
            }
        }
    });
    matrix.commit();
    linalg::Vector<Scalar, Exec> x(matrix.distribution()), y(matrix.distribution());
    const auto local = x.overwrite(exec);
    const int first = f.first;
    Kokkos::parallel_for("test/shared_face_input", Policy(exec, 0, f.host.n_own), KOKKOS_LAMBDA(LocalIndex cell) {
        for (int c = 0; c < width; ++c)
            local(static_cast<std::size_t>(cell)*static_cast<std::size_t>(width)+static_cast<std::size_t>(c), 0) =
                static_cast<Scalar>(10*(first+cell+1)+c+1);
    });
    x.commit(); matrix.apply(x, y);
    const auto result = y.read(exec);
    auto host = Kokkos::create_mirror(result);
    Kokkos::deep_copy(exec, host, result); y.commit();
    for (LocalIndex cell = 0; cell < f.host.n_own; ++cell) {
        const auto i = static_cast<std::size_t>(cell);
        for (int equation = 0; equation < width; ++equation) {
            double expected = 2*(10*(f.first+cell+1)+equation+1);
            for (auto k = f.connectivity.cell_faces_offsets[i]; k < f.connectivity.cell_faces_offsets[i+1]; ++k) {
                const auto face = static_cast<std::size_t>(f.connectivity.cell_faces[k]);
                if (face >= static_cast<std::size_t>(f.host.n_inner_faces)) continue;
                const auto other = f.host.face_owner[face] == cell ? f.host.face_neigh[face] : f.host.face_owner[face];
                const double delta = 10*(static_cast<double>(f.host.cell_centroid_x[i])-
                                         static_cast<double>(f.host.cell_centroid_x[static_cast<std::size_t>(other)]));
                for (int variable = 0; variable < width; ++variable)
                    expected += ((equation+1)*4+variable+1)*delta;
            }
            const auto dof = i*static_cast<std::size_t>(width)+static_cast<std::size_t>(equation);
            test::near(static_cast<double>(host(dof, 0)), expected, 1e-5, "shared CSR/BSR map assembled the wrong entry or block");
        }
    }
}

void assembly_case(const Exec& exec) {
    const auto comm = mpi::CommView::world();
    for (int scenario = 0; scenario < 4; ++scenario) {
        const auto f = bar(comm, scenario == 1, scenario == 2, scenario == 3);
        // Returning a copy exercises allocation lifetime after the original plan is destroyed.
        auto assembly = [&] {
            const linalg::FaceAssemblyPlan<Exec> original(f.host, comm, exec);
            auto copy = original;
            check(copy.pattern().native().getRawPtr() == original.pattern().native().getRawPtr(), "assembly copy duplicated the graph");
            check(copy.entry_map().owner_entries.data() == original.entry_map().owner_entries.data() &&
                  copy.entry_map().neighbour_entries.data() == original.entry_map().neighbour_entries.data() &&
                  copy.entry_map().diagonal.data() == original.entry_map().diagonal.data(), "assembly copy duplicated the entry map");
            return copy;
        }();
        check(!assembly.pattern().has_input_mapping(), "assembly retained an unused input permutation");
        mesh::compute::MeshStorage<Exec> mesh(f.host, {}, exec);
        mesh::compute::AuxConnectivityStorage<Exec> connectivity(f.host, f.connectivity, exec);
        mesh::compute::AuxGeometryStorage<Exec> geometry(f.host, f.geometry, exec);
        ts::DiffusionPlan<Exec, Real> first(f.host, mesh.view(), connectivity.view(), geometry.view(), assembly, exec);
        ts::DiffusionPlan<Exec, float> second(f.host, mesh.view(), connectivity.view(), geometry.view(), assembly, exec);
        check(first.pattern().native().getRawPtr() == assembly.pattern().native().getRawPtr() &&
              second.pattern().native().getRawPtr() == assembly.pattern().native().getRawPtr(), "diffusion rebuilt the shared graph");
        check(first.row_plan().owner_entries.data() == assembly.entry_map().owner_entries.data() &&
              second.row_plan().diagonal.data() == assembly.entry_map().diagonal.data(), "diffusion rebuilt the shared entry map");
        linalg::CsrMatrix<double, Exec> scalar(assembly.pattern());
        linalg::CsrMatrix<float, Exec> scalar_float(assembly.pattern());
        linalg::BsrPattern<Exec> pairs(assembly.pattern(), 2), triples(assembly.pattern(), 3);
        linalg::BsrMatrix<double, Exec> block2(pairs);
        linalg::BsrMatrix<float, Exec> block3(triples);
        check(scalar.pattern().native().getRawPtr() == pairs.native().getRawPtr() &&
              pairs.native().getRawPtr() == triples.native().getRawPtr(), "CSR/BSR patterns duplicated the graph");
        assembly_product(scalar, f, mesh.view(), connectivity.view(), assembly.entry_map(), exec);
        assembly_product(scalar_float, f, mesh.view(), connectivity.view(), assembly.entry_map(), exec);
        assembly_product(block2, f, mesh.view(), connectivity.view(), assembly.entry_map(), exec);
        assembly_product(block3, f, mesh.view(), connectivity.view(), assembly.entry_map(), exec);
        Boundary boundary(f.host, mesh.view(), exec);
        boundary.set(0, bc::FixedValue<Real>{0}); boundary.set(1, bc::FixedValue<Real>{1});
        ts::DiffusionOperator<> diffusion(first);
        diffusion.assemble_async(boundary); diffusion.commit(); diffusion.require_ready();
        check(diffusion.matrix().pattern().native().getRawPtr() == assembly.pattern().native().getRawPtr(),
              "diffusion matrix duplicated the shared graph");
    }
}

void nonconvergence(const Exec& exec) {
    const auto comm = mpi::CommView::world(); const auto f = bar(comm); Attached a(f, comm, exec);
    Boundary boundary(f.host, a.mesh.view(), exec); boundary.set(0, bc::FixedValue<Real>{0}); boundary.set(1, bc::FixedValue<Real>{1});
    fields::FieldStorage<double, 3, Exec> field("unchanged", boundary.required_field_size(), fields::FieldLocation::Custom, exec); field.fill(exec, -77);
    ts::SteadyDiffusion<double> solver(a.plan, field.view().component(1), settings(true));
    const auto result = solver.solve_owned(boundary);
    check(!result.converged && !result.iterations_started, "zero iteration solve started Krylov iterations");
    const auto h = snapshot(field, exec);
    for (std::size_t i = 0; i < field.size(); ++i) for (int c = 0; c < 3; ++c)
        test::near(h(i, c), -77.0, 0.0, "nonconvergence modified application storage");
}

void linear(const Exec& e) { linear_case(e); }
void gradient(const Exec& e) { linear_case(e, 1); }
void flux(const Exec& e) { linear_case(e, 2); }
void robin(const Exec& e) { linear_case(e, 3); }
void duplicates(const Exec& e) { linear_case(e, 0, true); }
void strided_float(const Exec& e) { linear_case<float, Kokkos::LayoutRight, float>(e); }
void float_solver(const Exec& e) { linear_case<float, Kokkos::LayoutLeft, float, float>(e); }
void empty(const Exec& e) { linear_case(e, 0, false, true); linear_case(e, 0, false, false, true); }

template<int Which>
void rejected(const Exec& exec) {
    const auto comm = mpi::CommView::world(); auto f = bar(comm);
    if constexpr (Which == 4) f.geometry.face_area_over_normal_dist[0] *= 2;
    if constexpr (Which == 5) {
        f.geometry.active_mask = mesh::AuxGeomType::FaceNormalDistance | mesh::AuxGeomType::FaceAreaOverNormalDistance;
        f.geometry.face_interp_weight.clear();
    }
    if constexpr (Which == 6) f.host.face_centroid_y[static_cast<std::size_t>(f.host.n_inner_faces)] = GeometryReal{0.25};
    Attached a(f, comm, exec); Boundary boundary(f.host, a.mesh.view(), exec);
    boundary.set(0, bc::FixedValue<Real>{0}); boundary.set(1, bc::FixedValue<Real>{1});
    if constexpr (Which == 1) { boundary.set(0, bc::NormalGradient<Real>{0}); boundary.set(1, bc::NormalGradient<Real>{0}); }
    if constexpr (Which == 7) boundary.set(1, NegativeRelation{});
    ts::DiffusionOperator<> op(a.plan);
    if constexpr (Which == 2) op.assemble_async(boundary, transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{0}});
    else if constexpr (Which == 3) op.assemble_async(boundary, transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{1}}, ts::UniformSource<Real>{std::numeric_limits<Real>::quiet_NaN()});
    else if constexpr (Which == 5) {
        Kokkos::View<Real*, Space> d("D", static_cast<std::size_t>(f.host.n_cells)); Kokkos::deep_copy(exec, d, Real{1});
        op.assemble_async(boundary, transport::IsotropicDiffusion{transport::InterpolatedCellCoefficient{d}});
    } else op.assemble_async(boundary);
    if constexpr (Which == 0) op.assemble_async(boundary);
    op.commit();
    check(false, "invalid system unexpectedly accepted");
}

template<auto Function>
void with_tpetra(const Exec& exec) {
    mpi::guarded([&] {
        Tpetra::ScopeGuard tpetra(nullptr, nullptr, mpi::CommView::world().native());
        test::complete(exec, [&] { Function(exec); });
    });
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::array positive{
        test::Case{"linear", with_tpetra<linear>}, test::Case{"normal_gradient", with_tpetra<gradient>},
        test::Case{"fixed_flux", with_tpetra<flux>}, test::Case{"robin", with_tpetra<robin>},
        test::Case{"source", with_tpetra<source_case>}, test::Case{"variable", with_tpetra<variable_case>},
        test::Case{"shared", with_tpetra<shared_case>}, test::Case{"shared_assembly", with_tpetra<assembly_case>}, test::Case{"nonconvergence", with_tpetra<nonconvergence>},
        test::Case{"duplicate_faces", with_tpetra<duplicates>}, test::Case{"strided_float", with_tpetra<strided_float>},
        test::Case{"empty", with_tpetra<empty>}, test::Case{"float_solver", with_tpetra<float_solver>}};
    const std::array negative{
        test::Case{"pending", with_tpetra<rejected<0>>}, test::Case{"pure_neumann", with_tpetra<rejected<1>>},
        test::Case{"bad_coefficient", with_tpetra<rejected<2>>}, test::Case{"bad_source", with_tpetra<rejected<3>>},
        test::Case{"bad_metric", with_tpetra<rejected<4>>}, test::Case{"missing_weights", with_tpetra<rejected<5>>},
        test::Case{"nonorthogonal", with_tpetra<rejected<6>>}, test::Case{"bad_boundary", with_tpetra<rejected<7>>}};
    if (argc == 2) for (const auto& c : negative) if (std::string_view(argv[1]) == c.name)
        return test::run(argc, argv, "transport.scalar", negative);
    return test::run(argc, argv, "transport.scalar", positive);
}