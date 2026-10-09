#include "common/test_support.hpp"

#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace {
using namespace krona::test;
using namespace krona::mesh::compute;
namespace gradient = krona::numerics::gradient;
using krona::mesh::MeshPart;
using krona::mesh::MeshAuxConnectivity;
using krona::mesh::MeshAuxGeometry;
using Vec = std::array<double, 3>;
constexpr std::array schemes{gradient::Scheme::GreenGaussCell,
                            gradient::Scheme::GreenGaussFace,
                            gradient::Scheme::LeastSquaresFace};

Vec point(double x, double y, double z, bool skew, double scale)
{ return {scale * (x + (skew ? 0.3*y : 0)), scale * (y + (skew ? 0.2*z : 0)), scale*z}; }

struct Fixture {
    MeshPart mesh;
    MeshAuxConnectivity connectivity;
    MeshAuxGeometry geometry;
};

/// @brief Two unequal-width affine hexahedra; each MPI rank owns one in the distributed case.
Fixture fixture(int rank = 0, int ranks = 1, bool unequal = false, bool skew = false, double scale = 1)
{
    Fixture result;
    auto& m = result.mesh;
    m.rank = rank; m.nprocs = ranks; m.faces_grouped = true;
    const std::array<double, 3> edge{0, 1, unequal ? 3.0 : 2.0};
    std::vector<int> global;
    if (ranks == 1) global = {0, 1};
    else global = {rank, 1-rank};
    m.n_own = ranks == 1 ? 2 : 1; m.n_cells = 2;
    std::map<int, Index> local;
    for (std::size_t i = 0; i < global.size(); ++i) {
        const auto g = global[i]; local[g] = static_cast<Index>(i);
        m.cell_gid.push_back(g);
        const auto c = point(0.5*(edge[static_cast<std::size_t>(g)]+edge[static_cast<std::size_t>(g+1)]), 0.5, 0.5, skew, scale);
        m.cell_centroid_x.push_back(c[0]); m.cell_centroid_y.push_back(c[1]); m.cell_centroid_z.push_back(c[2]);
        m.cell_volume.push_back(scale*scale*scale*(edge[static_cast<std::size_t>(g+1)]-edge[static_cast<std::size_t>(g)]));
    }
    struct Face { Index owner, neigh; Vec centre, normal; double area, weight; };
    std::array<std::vector<Face>, 3> groups;
    for (Index i = 0; i < m.n_own; ++i) {
        const int g = global[static_cast<std::size_t>(i)];
        const double left = edge[static_cast<std::size_t>(g)], right = edge[static_cast<std::size_t>(g+1)];
        for (int axis = 0; axis < 3; ++axis) for (int sign : {-1, 1}) {
            const int ng = g + sign;
            const bool interior = axis == 0 && ng >= 0 && ng < 2;
            const auto neigh = interior ? local.at(ng) : Index{-1};
            if (interior && neigh < m.n_own && neigh < i) continue;
            const auto centre = point(axis == 0 ? (sign < 0 ? left : right) : 0.5*(left+right),
                                      axis == 1 ? (sign < 0 ? 0 : 1) : 0.5,
                                      axis == 2 ? (sign < 0 ? 0 : 1) : 0.5, skew, scale);
            Vec surface = axis == 0 ? Vec{1, skew ? -0.3 : 0, skew ? 0.06 : 0}
                          : axis == 1 ? Vec{0, 1, skew ? -0.2 : 0} : Vec{0, 0, 1};
            const double length = std::sqrt(surface[0]*surface[0]+surface[1]*surface[1]+surface[2]*surface[2]);
            for (auto& value : surface) value *= static_cast<double>(sign)/length;
            const double area = length * scale*scale * (axis == 0 ? 1 : right-left);
            const double weight = interior ?
                (edge[static_cast<std::size_t>(ng+1)]-edge[static_cast<std::size_t>(ng)]) /
                (right-left+edge[static_cast<std::size_t>(ng+1)]-edge[static_cast<std::size_t>(ng)]) : 0.5;
            const int group = !interior ? 2 : neigh < m.n_own ? 0 : 1;
            groups[static_cast<std::size_t>(group)].push_back({i, neigh, centre, surface, area, weight});
        }
    }
    m.n_internal_faces = static_cast<Index>(groups[0].size());
    m.n_inner_faces = m.n_internal_faces + static_cast<Index>(groups[1].size());
    for (const auto& group : groups) for (const auto& f : group) {
        m.face_owner.push_back(f.owner); m.face_neigh.push_back(f.neigh);
        m.face_centroid_x.push_back(f.centre[0]); m.face_centroid_y.push_back(f.centre[1]); m.face_centroid_z.push_back(f.centre[2]);
        m.face_normal_x.push_back(f.normal[0]); m.face_normal_y.push_back(f.normal[1]); m.face_normal_z.push_back(f.normal[2]);
        m.face_area.push_back(f.area); result.geometry.face_interp_weight.push_back(f.weight);
        const auto o = static_cast<std::size_t>(f.owner), n = static_cast<std::size_t>(f.neigh >= 0 ? f.neigh : f.owner);
        result.geometry.face_cell_dist_x.push_back(f.neigh >= 0 ? m.cell_centroid_x[n]-m.cell_centroid_x[o] : 2*(f.centre[0]-m.cell_centroid_x[o]));
        result.geometry.face_cell_dist_y.push_back(f.neigh >= 0 ? m.cell_centroid_y[n]-m.cell_centroid_y[o] : 2*(f.centre[1]-m.cell_centroid_y[o]));
        result.geometry.face_cell_dist_z.push_back(f.neigh >= 0 ? m.cell_centroid_z[n]-m.cell_centroid_z[o] : 2*(f.centre[2]-m.cell_centroid_z[o]));
    }
    m.n_faces = static_cast<Index>(m.face_owner.size());
    auto& conn = result.connectivity;
    conn.active_mask = krona::mesh::AuxConnType::CellFaces;
    conn.n_cells_own = m.n_own; conn.n_faces = m.n_faces;
    conn.cell_faces_offsets = {0};
    for (Index i = 0; i < m.n_own; ++i) {
        for (Index f = 0; f < m.n_faces; ++f)
            if (m.face_owner[static_cast<std::size_t>(f)] == i || m.face_neigh[static_cast<std::size_t>(f)] == i) conn.cell_faces.push_back(f);
        conn.cell_faces_offsets.push_back(conn.cell_faces.size());
    }
    result.geometry.active_mask = krona::mesh::AuxGeomType::FaceInterpolationWeights | krona::mesh::AuxGeomType::FaceCellDistanceVector;
    result.geometry.n_faces = m.n_faces;
    if (ranks == 2) {
        m.nb_ranks = {1-rank}; m.send_owned_local = {0}; m.recv_ghost_local = {1};
        m.send_offsets = {0, 1}; m.recv_offsets = {0, 1};
    }
    return result;
}

template<class T, int N, class Layout, class Coefficient = krona::GeometryReal,
         class Accumulator = krona::AccumulationReal>
void numerical(const Exec& exec, Fixture data, gradient::Options options, bool constant = false,
               bool ranges = false, bool distributed = false, bool use_geometry = true)
{
    const auto& m = data.mesh;
    MeshStorage<> mesh(m, {.data = MeshData::FiniteVolume}, exec);
    AuxConnectivityStorage<> connectivity(m, data.connectivity, exec);
    AuxGeometryStorage<> geometry(m, data.geometry, exec);
    gradient::GradientOperator<Exec, Coefficient, Accumulator> op(mesh.view(), connectivity.view(),
        use_geometry ? geometry.view() : AuxGeometryView<>{}, options, exec);
    const auto n = op.required_input_size(), owned = static_cast<std::size_t>(m.n_own);
    krona::fields::FieldStorage<T, N, Exec, Layout> input("q", n, krona::fields::FieldLocation::Custom, exec);
    krona::fields::FieldStorage<T, 3*N, Exec, Layout> output("grad_q", n, krona::fields::FieldLocation::Custom, exec);
    const auto q = input.view();
    const auto g = output.view();
    auto hq = input.create_host_buffer();
    for (std::size_t i = 0; i < n; ++i) {
        Vec p{};
        if (i < static_cast<std::size_t>(m.n_cells)) p = {m.cell_centroid_x[i], m.cell_centroid_y[i], m.cell_centroid_z[i]};
        else {
            const auto f = static_cast<std::size_t>(m.n_inner_faces) + i - static_cast<std::size_t>(m.n_cells);
            p = {m.face_centroid_x[f], m.face_centroid_y[f], m.face_centroid_z[f]};
            if (options.boundary == gradient::BoundaryValues::MirrorCell) {
                const auto owner = static_cast<std::size_t>(m.face_owner[f]);
                p = {2*p[0]-m.cell_centroid_x[owner], 2*p[1]-m.cell_centroid_y[owner], 2*p[2]-m.cell_centroid_z[owner]};
            }
        }
        for (int c = 0; c < N; ++c) {
            T value = static_cast<T>(constant ? 9.0 : static_cast<double>(c+1)*(2*p[0]-3*p[1]+0.75*p[2]));
            if (distributed && i >= owned && i < static_cast<std::size_t>(m.n_cells)) value = std::numeric_limits<T>::quiet_NaN();
            if constexpr (N == 1) hq(i) = value; else hq(i, c) = value;
        }
    }
    Kokkos::deep_copy(exec, q.kokkos_view(), hq);
    output.fill(exec, T{-999});
    auto hg = output.create_host_buffer();
    if (distributed) {
        krona::halo::Exchange<T> exchange(mesh.view(), mesh.halo_plan(), MPI_COMM_WORLD, 931, exec);
        for (int c = 0; c < N; ++c)
            exchange.add_component(Kokkos::subview(q.component(c), std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(m.n_cells)}));
        exchange.prepare(); exchange.start(); exchange.finish_async();
        op.apply(q.as_const(), g);
        exchange.wait();
        // Gradient halo fills MPI ghosts; physical boundary entries retain the sentinel.
        krona::halo::Exchange<T> gradient_exchange(mesh.view(), mesh.halo_plan(), MPI_COMM_WORLD, 932, exec);
        for (int c = 0; c < 3*N; ++c)
            gradient_exchange.add_component(Kokkos::subview(g.component(c), std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(m.n_cells)}));
        gradient_exchange.prepare(); gradient_exchange.exchange();
        Kokkos::deep_copy(exec, hg, g.kokkos_view()); exec.fence();
    } else {
        if (ranges && owned > 0) {
            op.apply(q, g, {0, 1});
            Kokkos::deep_copy(exec, hg, g.kokkos_view()); exec.fence();
            for (std::size_t i = 1; i < n; ++i) for (int c = 0; c < 3*N; ++c)
                equal(hg(i, c), T{-999}, "Partial range preserves remaining cells");
            op.apply(q, g, {1, m.n_own});
        } else op.apply(q, g);
        Kokkos::deep_copy(exec, hg, g.kokkos_view()); exec.fence();
    }
    const std::size_t ready = distributed ? static_cast<std::size_t>(m.n_cells) : owned;
    const std::array<double, 3> slope{2, -3, 0.75};
    // Well-conditioned manufactured stencils reproduce affine fields within rounding error.
    constexpr double epsilon = std::max({static_cast<double>(std::numeric_limits<T>::epsilon()),
        static_cast<double>(std::numeric_limits<Coefficient>::epsilon()),
        static_cast<double>(std::numeric_limits<Accumulator>::epsilon()),
        static_cast<double>(std::numeric_limits<krona::GeometryReal>::epsilon())});
    for (std::size_t i = 0; i < n; ++i) for (int c = 0; c < N; ++c) for (int axis = 0; axis < 3; ++axis) {
        if (i >= ready) equal(hg(i, 3*c+axis), T{-999}, "Ghost/boundary tail is not written");
        else {
            const T expected = static_cast<T>(constant ? 0 : (c+1)*slope[static_cast<std::size_t>(axis)]);
            const T tolerance = static_cast<T>(64*epsilon*std::max(1.0, std::abs(static_cast<double>(expected))));
            near(hg(i, 3*c+axis), expected, tolerance, "Manufactured affine/constant gradient");
        }
    }
    equal(op.coefficient_bytes() == 0, options.scheme != gradient::Scheme::LeastSquaresFace || owned == 0,
          "Only nonempty LS plans own coefficient storage");
}

void constant_case(const Exec& exec)
{
    for (const auto scheme : schemes) for (const auto boundary : {gradient::BoundaryValues::Face, gradient::BoundaryValues::MirrorCell})
        numerical<krona::StorageReal, 1, Kokkos::LayoutLeft>(exec, fixture(), {.scheme=scheme, .boundary=boundary}, true);
}
void affine_case(const Exec& exec)
{
    for (const auto scheme : schemes) for (const auto boundary : {gradient::BoundaryValues::Face, gradient::BoundaryValues::MirrorCell})
        numerical<krona::StorageReal, 3, Kokkos::LayoutLeft>(exec, fixture(0, 1, false, true), {.scheme=scheme, .boundary=boundary});
}
void weighted_case(const Exec& exec)
{
    for (const auto scheme : schemes) for (const auto boundary : {gradient::BoundaryValues::Face, gradient::BoundaryValues::MirrorCell}) {
        numerical<double, 1, Kokkos::LayoutLeft>(exec, fixture(0, 1, true, true),
            {.scheme=scheme, .boundary=boundary, .interpolation=gradient::Interpolation::AuxiliaryWeights});
        if (scheme == gradient::Scheme::LeastSquaresFace)
            numerical<double, 1, Kokkos::LayoutLeft>(exec, fixture(0, 1, true, true), {.scheme=scheme, .boundary=boundary}, false, false, false, false);
    }
}
void layouts_case(const Exec& exec)
{
    for (const auto scheme : schemes)
        numerical<krona::StorageReal, 3, Kokkos::LayoutRight>(exec, fixture(), {.scheme=scheme});
    auto data = fixture();
    MeshStorage<> mesh(data.mesh, {}, exec);
    AuxConnectivityStorage<> connectivity(data.mesh, data.connectivity, exec);
    gradient::GradientOperator<> op(mesh.view(), connectivity.view(), {}, {}, exec);
    using Memory = typename Exec::memory_space;
    Kokkos::View<double*[2], Kokkos::LayoutRight, Memory> q("batch_q", op.required_input_size());
    Kokkos::View<double*[6], Kokkos::LayoutRight, Memory> g("batch_g", op.required_input_size());
    Kokkos::deep_copy(exec, q, 8.0); Kokkos::deep_copy(exec, g, -999.0);
    op.apply_set(std::array{Kokkos::subview(q, Kokkos::ALL(), 0), Kokkos::subview(q, Kokkos::ALL(), 1)},
                 std::array{Kokkos::subview(g, Kokkos::ALL(), 0), Kokkos::subview(g, Kokkos::ALL(), 3)},
                 std::array{Kokkos::subview(g, Kokkos::ALL(), 1), Kokkos::subview(g, Kokkos::ALL(), 4)},
                 std::array{Kokkos::subview(g, Kokkos::ALL(), 2), Kokkos::subview(g, Kokkos::ALL(), 5)});
    op.apply_component(Kokkos::subview(q, Kokkos::ALL(), 0), Kokkos::subview(g, Kokkos::ALL(), 0),
                        Kokkos::subview(g, Kokkos::ALL(), 1), Kokkos::subview(g, Kokkos::ALL(), 2));
    auto host = Kokkos::create_mirror(g); Kokkos::deep_copy(exec, host, g); exec.fence();
    for (std::size_t i = 0; i < op.required_input_size(); ++i) for (int c = 0; c < 6; ++c)
        equal(host(i, c), i < 2 ? 0.0 : -999.0, "Strided component batch and scalar API");
}
void range_case(const Exec& exec)
{ for (const auto scheme : schemes) numerical<double, 1, Kokkos::LayoutLeft>(exec, fixture(), {.scheme=scheme}, false, true); }
void scale_case(const Exec& exec)
{
    for (const double scale : {1.0e-9, 1.0e9})
        numerical<double, 1, Kokkos::LayoutLeft>(exec, fixture(0, 1, true, true, scale), {}, false, false, false, false);
}
void precision_case(const Exec& exec)
{
    for (const auto scheme : schemes) {
        numerical<float, 3, Kokkos::LayoutLeft, float, float>(exec, fixture(0, 1, true, true),
            {.scheme=scheme, .interpolation=gradient::Interpolation::AuxiliaryWeights});
        numerical<double, 1, Kokkos::LayoutLeft, float, double>(exec, fixture(0, 1, true, true),
            {.scheme=scheme, .interpolation=gradient::Interpolation::AuxiliaryWeights});
    }
}
void degree_case(const Exec& exec)
{
    auto data = fixture();
    auto& m = data.mesh;
    m.n_own = 1; m.n_cells = 1; m.n_internal_faces = 0; m.n_inner_faces = 0; m.n_faces = 90;
    m.cell_centroid_x = {0}; m.cell_centroid_y = {0}; m.cell_centroid_z = {0}; m.cell_volume = {1};
    m.face_owner.assign(90, 0); m.face_neigh.assign(90, -1);
    m.face_area.assign(90, 1); m.face_centroid_x.assign(90, 0); m.face_centroid_y.assign(90, 0); m.face_centroid_z.assign(90, 0);
    m.face_normal_x.assign(90, 0); m.face_normal_y.assign(90, 0); m.face_normal_z.assign(90, 0);
    data.connectivity.n_cells_own = 1; data.connectivity.n_faces = 90;
    data.connectivity.cell_faces_offsets = {0, 90}; data.connectivity.cell_faces.clear();
    for (Index f = 0; f < 90; ++f) {
        data.connectivity.cell_faces.push_back(f);
        auto* coordinate = f%3 == 0 ? &m.face_centroid_x : f%3 == 1 ? &m.face_centroid_y : &m.face_centroid_z;
        (*coordinate)[static_cast<std::size_t>(f)] = f%2 == 0 ? 1 : -1;
    }
    data.geometry = {}; data.geometry.n_faces = 90;
    numerical<double, 1, Kokkos::LayoutLeft>(exec, data, {});
}
void empty_case(const Exec& exec)
{
    Fixture data; data.mesh.faces_grouped = true;
    data.connectivity.active_mask = krona::mesh::AuxConnType::CellFaces;
    data.connectivity.cell_faces_offsets = {0};
    for (const auto scheme : schemes) numerical<double, 1, Kokkos::LayoutLeft>(exec, data, {.scheme=scheme});
}
void distributed_case(const Exec& exec)
{
    int rank = 0, ranks = 0;
    krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    equal(ranks, 2, "Distributed case requires two ranks");
    for (const auto scheme : schemes)
        numerical<krona::StorageReal, 3, Kokkos::LayoutLeft>(exec, fixture(rank, ranks, true, true),
            {.scheme=scheme, .interpolation=gradient::Interpolation::AuxiliaryWeights}, false, false, true);
}
void singular_case(const Exec& exec)
{
    auto data = fixture();
    std::fill(data.mesh.cell_centroid_z.begin(), data.mesh.cell_centroid_z.end(), 0);
    std::fill(data.mesh.face_centroid_z.begin(), data.mesh.face_centroid_z.end(), 0);
    data.geometry = {}; data.geometry.n_faces = data.mesh.n_faces;
    numerical<double, 1, Kokkos::LayoutLeft>(exec, data, {});
}
} // namespace

int main(int argc, char** argv)
{
    constexpr std::array cases{Case{"constant", constant_case}, Case{"affine", affine_case},
        Case{"weighted", weighted_case}, Case{"layouts", layouts_case}, Case{"range", range_case},
        Case{"scale", scale_case}, Case{"precision", precision_case}, Case{"degree", degree_case}, Case{"empty", empty_case},
        Case{"distributed", distributed_case}, Case{"singular", singular_case}};
    return krona::test::run(argc, argv, "gradient", cases);
}
