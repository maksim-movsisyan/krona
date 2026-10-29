#include "../common/test_support.hpp"
#include "mesh_fixture.hpp"

#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/validation.hpp>

#include <array>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using namespace krona::test;
using krona::GeometryReal;
using krona::mesh::host::MeshPart;
using namespace krona::mesh::device;
using Memory = typename Exec::memory_space;
namespace validation = krona::mesh::device::validation;

/// @brief Compares every transferred element, including enums and CSR offsets.
template<class View, class T>
void expect_transferred(const Exec& exec, View view, const std::vector<T>& expected, const char* message)
{
    equal(view.extent(0), expected.size(), message);
    if (expected.empty()) return;
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    complete(exec, [&] { Kokkos::deep_copy(exec, host, view); });
    for (std::size_t i = 0; i < expected.size(); ++i)
        check(host(i) == expected[i], message);
}
 
/// @brief Independent masks expose exactly their selected groups.
void selection(const Exec& exec)
{
    const auto source = make_mesh();
    MeshStorage<> metadata(source, {.data = MeshData::None}, exec);
    const auto counts = metadata.view();
    equal(counts.sizes.n_own, Index{2}, "Owned count survives an empty mask");
    equal(counts.sizes.n_cells, Index{3}, "Total count includes one ghost");
    check(counts.active_mask == MeshData::None, "No array groups were selected");
    check(counts.connectivity.face_owner.extent(0) == 0, "No implicit face-cell data");
    check(counts.geometry.cell_volume.extent(0) == 0, "No implicit geometry");
    check(metadata.halo_plan().send_offsets.empty(), "No implicit host halo plan");

    MeshStorage<> nodal(source, {.data = MeshData::CellNodes | MeshData::NodeCoordinates}, exec);
    const auto nodes = nodal.view();
    check(nodes.has(MeshData::CellNodes | MeshData::NodeCoordinates), "Nodal groups are ready");
    check(!nodes.has(MeshData::FaceCells), "Nodal selection does not require face-cell links");
    equal(nodes.connectivity.cell_nodes.extent(0), std::size_t{12}, "Three tetrahedral rows");
    equal(nodes.geometry.node_x.extent(0), std::size_t{6}, "All local node coordinates");

    MeshStorage<> defaults(source, {}, exec);
    const auto fvm = defaults.view();
    check(fvm.active_mask == MeshData::FiniteVolume, "Default preset is FiniteVolume");
    check(!fvm.has(MeshData::NodeCoordinates), "Node coordinates remain optional");

    // Availability is independent of whether the caller uses grouped face intervals.
    auto ungrouped = source;
    ungrouped.faces_grouped = false;
    MeshStorage<> links(ungrouped, {.data = MeshData::FaceCells}, exec);
    check(links.view().has(MeshData::FaceCells) && !links.view().sizes.faces_grouped,
          "Face-cell links are available without grouping");
}

/// @brief Kernels read geometry and distinguish owned cells from ghost copies.
void geometry(const Exec& exec)
{
    const auto source = make_mesh();
    MeshStorage<> storage(source, {.data = MeshData::All}, exec);
    const auto mesh = storage.view();
    check(mesh.sizes.faces_grouped, "Face intervals require grouping");
    static_assert(std::is_const_v<typename decltype(mesh.geometry.cell_volume)::value_type>);

    Kokkos::View<Index*, Memory> face_codes(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/face_codes"), 7);
    auto host_codes = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, face_codes);
    GeometryReal owned_volume = 0, all_volume = 0;
    complete(exec, [&] {
        Kokkos::parallel_reduce("test/owned_volume", Policy(exec, 0, mesh.sizes.n_own),
            KOKKOS_LAMBDA(Index cell, GeometryReal& sum) {
                sum += mesh.geometry.cell_volume(cell);
            }, owned_volume);
        Kokkos::parallel_reduce("test/all_volume", Policy(exec, 0, mesh.sizes.n_cells),
            KOKKOS_LAMBDA(Index cell, GeometryReal& sum) {
                sum += mesh.geometry.cell_volume(cell);
            }, all_volume);
        Kokkos::parallel_for("test/face_links", Policy(exec, 0, mesh.sizes.n_faces),
            KOKKOS_LAMBDA(Index face) {
                face_codes(face) = 10 * mesh.connectivity.face_owner(face) +
                                   mesh.connectivity.face_neigh(face) + 1;
            });
        Kokkos::deep_copy(exec, host_codes, face_codes);
    });
    constexpr auto tolerance = GeometryReal{16} * std::numeric_limits<GeometryReal>::epsilon();
    near(owned_volume, GeometryReal{1} / 3, tolerance, "Owned volume excludes the ghost");
    near(all_volume, GeometryReal{2} / 3, tolerance, "All-cell sum includes the ghost copy");
    constexpr std::array<Index, 7> expected{2, 3, 0, 0, 10, 10, 10};
    for (std::size_t face = 0; face < expected.size(); ++face)
        equal(host_codes(face), expected[face], "Face owner/neighbour values survive transfer");

    // Check the uploaded coordinates and oriented normal through independent host mirrors.
    auto host_z = Kokkos::create_mirror(Kokkos::WithoutInitializing, mesh.geometry.node_z);
    auto host_nz = Kokkos::create_mirror(Kokkos::WithoutInitializing, mesh.geometry.face_normal_z);
    complete(exec, [&] {
        Kokkos::deep_copy(exec, host_z, mesh.geometry.node_z);
        Kokkos::deep_copy(exec, host_nz, mesh.geometry.face_normal_z);
    });
    equal(host_z(4), GeometryReal{-1}, "Negative node coordinate is preserved");
    equal(host_nz(0), GeometryReal{-1}, "Internal-face normal points out of its owner");
}

/// @brief CPU views borrow arrays; non-HostSpace storage survives source destruction.
void ownership(const Exec& exec)
{
    auto source = std::make_unique<MeshPart>(make_mesh());
    MeshStorage<> storage(*source, {.data = MeshData::All}, exec);
    const auto mesh = storage.view();
    GeometryReal expected = GeometryReal{1} / 6;
    if constexpr (MeshStorage<>::borrows_host) {
        check(mesh.geometry.cell_volume.data() == source->cell_volume.data(), "CPU volume array is borrowed");
        check(mesh.connectivity.cell_nodes.data() == source->cell_nodes.data(), "CPU CSR payload is borrowed");
        check(mesh.halo.send_owned_local.data() == source->send_owned_local.data(), "CPU halo indices are borrowed");
        check(storage.halo_plan().nb_ranks.data() == source->nb_ranks.data(), "CPU peer table is borrowed");
        source->cell_volume[0] = GeometryReal{2};
        expected = GeometryReal{2};
    } else {
        // Device arrays and the small host halo plan must be independent.
        source.reset();
        equal(storage.halo_plan().nb_ranks[0], 1, "Owned host plan survives source destruction");
    }

    GeometryReal observed = 0;
    complete(exec, [&] {
        Kokkos::parallel_reduce("test/ownership", Policy(exec, 0, 1),
            KOKKOS_LAMBDA(Index cell, GeometryReal& sum) {
                sum += mesh.geometry.cell_volume(cell);
            }, observed);
    });
    equal(observed, expected, "Kernel reads the correct backing allocation");
}

/// @brief Host peer segments align with execution-space gather/scatter maps.
void halo_plan(const Exec& exec)
{
    const auto source = make_mesh();
    MeshStorage<> storage(source, {.data = MeshData::Halo}, exec);
    const auto mesh = storage.view();
    const auto plan = storage.halo_plan();
    equal(plan.n_neighbors(), std::size_t{1}, "One peer in the synthetic snapshot");
    equal(plan.nb_ranks[0], 1, "Peer rank is host-accessible");
    equal(plan.send_offsets[0], std::size_t{0}, "Send segment starts at zero");
    equal(plan.recv_offsets[1], std::size_t{1}, "Receive segment has one record");
    equal(plan.send_count(0), std::size_t{1}, "Send offsets count cells");
    equal(plan.recv_count(0), std::size_t{1}, "Receive offsets count cells");

    Kokkos::View<Index[2], Memory> indices(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/halo_indices"));
    auto host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, indices);
    complete(exec, [&] {
        Kokkos::parallel_for("test/halo_maps", Policy(exec, 0, 1), KOKKOS_LAMBDA(Index) {
            indices(0) = mesh.halo.send_owned_local(0);
            indices(1) = mesh.halo.recv_ghost_local(0);
        });
        Kokkos::deep_copy(exec, host, indices);
    });
    equal(host(0), Index{0}, "Send map addresses an owned cell");
    equal(host(1), Index{2}, "Receive map addresses the ghost slot");
}

/// @brief Optional CSR rows and metrics remain independent of the base view.
void auxiliary(const Exec& exec)
{
    const auto source = make_mesh();
    krona::mesh::host::MeshAuxConnectivity cache;
    cache.active_mask = krona::mesh::AuxConnType::CellFaces;
    cache.n_faces = source.n_faces;
    cache.n_cells_own = source.n_own;
    cache.n_nodes_own = source.n_nodes_own;
    cache.cell_faces_offsets = {0, 4, 8};
    cache.cell_faces = {0, 1, 2, 3, 0, 4, 5, 6};
    AuxConnectivityStorage<> storage(cache, exec);
    const auto rows = storage.view();
    check(rows.has_cell_faces() && !rows.has_node_cells(), "Only requested CSR is ready");

    Kokkos::View<Index[2], Memory> sums(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/row_sums"));
    auto host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, sums);
    complete(exec, [&] {
        Kokkos::parallel_for("test/aux_rows", Policy(exec, 0, 2), KOKKOS_LAMBDA(Index cell) {
            Index sum = 0;
            std::size_t ptr1 = rows.cell_faces_offsets[cell];
            std::size_t ptr2 = rows.cell_faces_offsets[cell+1];
            for (std::size_t i = ptr1; i < ptr2; ++i) sum += rows.cell_faces[i];
            sums(cell) = sum;
        });
        Kokkos::deep_copy(exec, host, sums);
    });
    equal(host(0), Index{6}, "First CSR row is correct in a kernel");
    equal(host(1), Index{15}, "Second CSR row uses its own offsets");

    krona::mesh::host::MeshAuxGeometry metrics;
    metrics.active_mask = krona::mesh::AuxGeomType::WallDistance;
    metrics.n_faces = source.n_faces;
    metrics.n_cells = source.n_cells;
    metrics.cell_wall_dist = {GeometryReal{0.125}, GeometryReal{0.25}, GeometryReal{0.5}};
    AuxGeometryStorage<> metric_storage(metrics, exec);
    const auto distances = metric_storage.view();
    auto host_distances = Kokkos::create_mirror(Kokkos::WithoutInitializing, distances.cell_wall_dist);
    complete(exec, [&] { Kokkos::deep_copy(exec, host_distances, distances.cell_wall_dist); });
    check(distances.has_wall_dist(), "Wall-distance cache is selected explicitly");
    equal(host_distances(2), GeometryReal{0.5}, "Auxiliary geometry includes the ghost cell");
}

/// @brief Selected groups remain ready on a rank with no local entities.
void empty(const Exec& exec)
{
    const MeshPart source;
    MeshStorage<> storage(source, {.data = MeshData::All}, exec);
    const auto mesh = storage.view();
    check(mesh.has(MeshData::All), "Ready groups may have empty payloads");
    equal(mesh.sizes.n_cells, Index{0}, "No local cells");
    equal(mesh.connectivity.cell_nodes_offsets.extent(0), std::size_t{1}, "Empty CSR keeps its terminal zero");
    equal(storage.halo_plan().n_neighbors(), std::size_t{0}, "No halo peers");
    equal(storage.halo_plan().send_offsets.size(), std::size_t{1}, "Selected empty halo keeps one offset");
    equal(storage.halo_plan().send_offsets[0], std::size_t{0}, "Empty halo offset is zero");
    GeometryReal volume = GeometryReal{-1};
    complete(exec, [&] {
        Kokkos::parallel_reduce("test/empty_volume", Policy(exec, 0, mesh.sizes.n_own),
            KOKKOS_LAMBDA(Index cell, GeometryReal& sum) { sum += mesh.geometry.cell_volume(cell); }, volume);
    });
    equal(volume, GeometryReal{0}, "Empty reduction produces its identity");
}

/// @brief Every selected base array preserves its complete payload.
void transfer(const Exec& exec)
{
    auto source = make_mesh();
    // Translation preserves the mesh while making coordinate components distinct.
    const auto shift = [](auto& values, GeometryReal delta) {
        for (auto& value : values) value += delta;
    };
    shift(source.node_x, GeometryReal{2});
    shift(source.node_y, GeometryReal{4});
    shift(source.node_z, GeometryReal{8});
    shift(source.cell_centroid_x, GeometryReal{2});
    shift(source.cell_centroid_y, GeometryReal{4});
    shift(source.cell_centroid_z, GeometryReal{8});
    shift(source.face_centroid_x, GeometryReal{2});
    shift(source.face_centroid_y, GeometryReal{4});
    shift(source.face_centroid_z, GeometryReal{8});
    const std::array<GeometryReal, 3> delta{2, 4, 8};
    for (std::size_t axis = 0; axis < delta.size(); ++axis) {
        source.bbox_lo[axis] += delta[axis];
        source.bbox_hi[axis] += delta[axis];
    }

    MeshStorage<> storage(source, {.data = MeshData::All}, exec);
    const auto view = storage.view();
    using Volume = decltype(view.geometry.cell_volume);
    static_assert(Volume::memory_traits::is_unmanaged);
    equal(view.sizes.n_nodes_own, source.n_nodes_own, "Owned-used node count is preserved");
    equal(view.sizes.n_nodes, source.n_nodes, "Ghost-only nodes are preserved");
    equal(view.sizes.n_internal_faces, source.n_internal_faces, "Internal face block is preserved");
    equal(view.sizes.n_inner_faces, source.n_inner_faces, "Interface face block is preserved");

    expect_transferred(exec, view.connectivity.face_owner, source.face_owner, "face_owner payload");
    expect_transferred(exec, view.connectivity.face_neigh, source.face_neigh, "face_neigh payload");
    expect_transferred(exec, view.connectivity.cell_type, source.cell_type, "cell_type payload");
    expect_transferred(exec, view.connectivity.cell_nodes_offsets, source.cell_nodes_offsets, "cell_nodes_offsets payload");
    expect_transferred(exec, view.connectivity.cell_nodes, source.cell_nodes, "cell_nodes payload");
    expect_transferred(exec, view.connectivity.face_type, source.face_type, "face_type payload");
    expect_transferred(exec, view.connectivity.face_local_index, source.face_local_index, "face_local_index payload");
    expect_transferred(exec, view.connectivity.face_nodes_offsets, source.face_nodes_offsets, "face_nodes_offsets payload");
    expect_transferred(exec, view.connectivity.face_nodes, source.face_nodes, "face_nodes payload");
    expect_transferred(exec, view.geometry.node_x, source.node_x, "node_x payload");
    expect_transferred(exec, view.geometry.node_y, source.node_y, "node_y payload");
    expect_transferred(exec, view.geometry.node_z, source.node_z, "node_z payload");
    expect_transferred(exec, view.geometry.cell_centroid_x, source.cell_centroid_x, "cell_centroid_x payload");
    expect_transferred(exec, view.geometry.cell_centroid_y, source.cell_centroid_y, "cell_centroid_y payload");
    expect_transferred(exec, view.geometry.cell_centroid_z, source.cell_centroid_z, "cell_centroid_z payload");
    expect_transferred(exec, view.geometry.cell_volume, source.cell_volume, "cell_volume payload");
    expect_transferred(exec, view.geometry.face_centroid_x, source.face_centroid_x, "face_centroid_x payload");
    expect_transferred(exec, view.geometry.face_centroid_y, source.face_centroid_y, "face_centroid_y payload");
    expect_transferred(exec, view.geometry.face_centroid_z, source.face_centroid_z, "face_centroid_z payload");
    expect_transferred(exec, view.geometry.face_normal_x, source.face_normal_x, "face_normal_x payload");
    expect_transferred(exec, view.geometry.face_normal_y, source.face_normal_y, "face_normal_y payload");
    expect_transferred(exec, view.geometry.face_normal_z, source.face_normal_z, "face_normal_z payload");
    expect_transferred(exec, view.geometry.face_area, source.face_area, "face_area payload");
    expect_transferred(exec, view.halo.send_owned_local, source.send_owned_local, "send_owned_local payload");
    expect_transferred(exec, view.halo.recv_ghost_local, source.recv_ghost_local, "recv_ghost_local payload");
}

/// @brief Explicit aux subsets preserve data and the CPU/GPU ownership contract.
void auxiliary_contracts(const Exec& exec)
{
    using krona::mesh::AuxConnType;
    using krona::mesh::AuxGeomType;
    const auto source = make_mesh();
    auto cache = std::make_unique<krona::mesh::host::MeshAuxConnectivity>();
    cache->active_mask = AuxConnType::CellFaces | AuxConnType::CellCellsByFace | AuxConnType::NodeCells;
    cache->n_faces = source.n_faces;
    cache->n_cells_own = source.n_own;
    cache->n_nodes_own = source.n_nodes_own;
    cache->cell_faces_offsets = {0, 4, 8};
    cache->cell_faces = {0, 1, 2, 3, 0, 4, 5, 6};
    cache->cell_cells_face_offsets = {0, 2, 3};
    cache->cell_cells_face = {1, 2, 0};
    cache->node_cells_offsets = {0, 2, 5, 8, 10, 11};
    cache->node_cells = {0, 1, 0, 1, 2, 0, 1, 2, 0, 2, 1};
    auto expected_cache = *cache;

    auto metrics = std::make_unique<krona::mesh::host::MeshAuxGeometry>();
    metrics->active_mask = AuxGeomType::FaceCellDistanceVector |
                          AuxGeomType::FaceInterpolationWeights | AuxGeomType::WallDistance;
    metrics->n_faces = source.n_faces;
    metrics->n_cells = source.n_cells;
    // Distances are computed independently from the fixture's saved centres.
    for (std::size_t face = 0; face < static_cast<std::size_t>(source.n_faces); ++face) {
        const auto owner = static_cast<std::size_t>(source.face_owner[face]);
        const auto neighbour = source.face_neigh[face];
        const auto other = neighbour < 0 ? face : static_cast<std::size_t>(neighbour);
        metrics->face_cell_dist_x.push_back(
            (neighbour < 0 ? source.face_centroid_x[other] : source.cell_centroid_x[other]) - source.cell_centroid_x[owner]);
        metrics->face_cell_dist_y.push_back(
            (neighbour < 0 ? source.face_centroid_y[other] : source.cell_centroid_y[other]) - source.cell_centroid_y[owner]);
        metrics->face_cell_dist_z.push_back(
            (neighbour < 0 ? source.face_centroid_z[other] : source.cell_centroid_z[other]) - source.cell_centroid_z[owner]);
    }
    metrics->face_interp_weight = {GeometryReal{0.5}, GeometryReal{2} / 3, 0, 0, 0, 0, 0};
    metrics->cell_wall_dist = {GeometryReal{0.125}, GeometryReal{0.25},
                             std::numeric_limits<GeometryReal>::infinity()};
    auto expected_metrics = *metrics;
    const auto selected_conn = AuxConnType::CellCellsByFace | AuxConnType::NodeCells;
    const auto selected_geom = AuxGeomType::FaceCellDistanceVector | AuxGeomType::WallDistance;
    AuxConnectivityStorage<> conn_storage(*cache, selected_conn, exec);
    AuxGeometryStorage<> geom_storage(*metrics, selected_geom, exec);
    const auto conn = conn_storage.view();
    const auto geom = geom_storage.view();
    check(conn.active_mask == selected_conn && geom.active_mask == selected_geom, "Explicit auxiliary masks are preserved");
    check(!conn.has_cell_faces() && conn.cell_faces_offsets.extent(0) == 0 && conn.cell_faces.extent(0) == 0,
          "Ready but unselected connectivity remains absent");
    check(!geom.has_face_interp_weights() && geom.face_interp_weight.extent(0) == 0,
          "Ready but unselected geometry remains absent");
    equal(geom.n_cells, source.n_cells, "Auxiliary cell count includes ghosts");

    if constexpr (AuxConnectivityStorage<>::borrows_host) {
        check(conn.node_cells_offsets.data() == cache->node_cells_offsets.data(), "CPU aux offsets are borrowed");
        check(conn.node_cells.data() == cache->node_cells.data(), "CPU aux indices are borrowed");
        check(geom.face_cell_dist_z.data() == metrics->face_cell_dist_z.data(), "CPU face metrics are borrowed");
        check(geom.cell_wall_dist.data() == metrics->cell_wall_dist.data(), "CPU wall distances are borrowed");
        metrics->cell_wall_dist[0] = GeometryReal{0.375};
        expected_metrics.cell_wall_dist[0] = GeometryReal{0.375};
    } else {
        check(conn.node_cells.data() != cache->node_cells.data(), "Copied aux indices have independent storage");
        check(geom.cell_wall_dist.data() != metrics->cell_wall_dist.data(), "Copied aux metrics have independent storage");
        cache->node_cells[0] = Index{2};
        metrics->cell_wall_dist[0] = GeometryReal{9};
        cache.reset();
        metrics.reset();
    }
    expect_transferred(exec, conn.cell_cells_face_offsets, expected_cache.cell_cells_face_offsets, "Aux face-neighbour offsets");
    expect_transferred(exec, conn.cell_cells_face, expected_cache.cell_cells_face, "Aux face-neighbour indices");
    expect_transferred(exec, conn.node_cells_offsets, expected_cache.node_cells_offsets, "Aux node-cell offsets");
    expect_transferred(exec, conn.node_cells, expected_cache.node_cells, "Aux node-cell indices");
    expect_transferred(exec, geom.face_cell_dist_x, expected_metrics.face_cell_dist_x, "Aux distance x component");
    expect_transferred(exec, geom.face_cell_dist_y, expected_metrics.face_cell_dist_y, "Aux distance y component");
    expect_transferred(exec, geom.face_cell_dist_z, expected_metrics.face_cell_dist_z, "Aux distance z component");
    expect_transferred(exec, geom.cell_wall_dist, expected_metrics.cell_wall_dist, "Aux distances and infinity sentinel");
}

/// @brief Validation rejects malformed data before unsafe incidence gathers.
void validation_cases(const Exec& exec)
{
    const auto source = make_mesh();
    MeshStorage<> defaults(source, {}, exec);
    check(validation::validate<MeshData::FiniteVolume>(exec, defaults.view()).ok(), "Valid FVM fixture passes validation");
    check(validation::validate(exec, defaults.view()).shapes != 0, "All-data validation rejects missing optional groups");

    MeshStorage<> storage(source, {.data = MeshData::All}, exec);
    const auto mesh = storage.view();
    check(validation::validate<MeshData::All, krona::mesh::AuxConnType::None,
                              krona::mesh::AuxGeomType::None, true>(exec, mesh).ok(),
          "Complete fixture passes numerical and incidence checks");

    auto missing = mesh;
    missing.geometry.cell_volume = {};
    const auto shape_error = validation::validate(exec, missing);
    check(shape_error.shapes != 0, "Missing selected array fails the shape gate");
    equal(shape_error.cells + shape_error.faces + shape_error.nodes, std::size_t{0}, "Shape failure prevents numerical checks");

    auto invalid_volume = source;
    invalid_volume.cell_volume[2] = GeometryReal{-1};
    MeshStorage<> volume_storage(invalid_volume, {.data = MeshData::CellGeometry}, exec);
    equal(validation::validate<MeshData::CellGeometry>(exec, volume_storage.view()).cells,
          std::size_t{1}, "Validation checks ghost-cell volume too");

    auto invalid_links = source;
    invalid_links.face_neigh[0] = Index{2};
    MeshStorage<> link_storage(invalid_links, {.data = MeshData::FaceCells}, exec);
    equal(validation::validate<MeshData::FaceCells>(exec, link_storage.view()).faces,
          std::size_t{1}, "Valid ghost index in the internal block violates grouping");

    auto invalid_nodes = source;
    invalid_nodes.cell_nodes[0] = source.n_nodes;
    MeshStorage<> node_storage(invalid_nodes, {.data = MeshData::Connectivity}, exec);
    const auto index_error = validation::validate<MeshData::Connectivity,
        krona::mesh::AuxConnType::None, krona::mesh::AuxGeomType::None, true>(exec, node_storage.view());
    equal(index_error.cell_nodes, std::size_t{1}, "Out-of-range node index fails its CSR row");
    equal(index_error.cell_incidence + index_error.face_incidence + index_error.node_incidence,
          std::size_t{0}, "Invalid indices never enter incidence gathers");

    krona::mesh::host::MeshAuxConnectivity invalid_incidence;
    invalid_incidence.active_mask = krona::mesh::AuxConnType::CellFaces;
    invalid_incidence.n_faces = source.n_faces;
    invalid_incidence.n_cells_own = source.n_own;
    invalid_incidence.n_nodes_own = source.n_nodes_own;
    invalid_incidence.cell_faces_offsets = {0, 4, 8};
    invalid_incidence.cell_faces = {0, 1, 2, source.n_faces, 0, 4, 5, 6};
    AuxConnectivityStorage<> incidence_storage(invalid_incidence, exec);
    const auto incidence_error = validation::validate<MeshData::FaceCells,
        krona::mesh::AuxConnType::CellFaces, krona::mesh::AuxGeomType::None, true>(
            exec, mesh, incidence_storage.view());
    equal(incidence_error.cell_faces, std::size_t{1}, "Out-of-range face index fails its CSR row");
    equal(incidence_error.cell_incidence + incidence_error.face_incidence,
          std::size_t{0}, "Invalid face indices cannot reach face-array gathers");

    // Inject bad CSR into a view, bypassing storage's constructor shape checks.
    const std::vector<std::size_t> bad_offsets{0, 4, 3, 12};
    Kokkos::View<std::size_t*, Memory> offsets(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/bad_offsets"), bad_offsets.size());
    const Kokkos::View<const std::size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_offsets(bad_offsets.data(), bad_offsets.size());
    complete(exec, [&] { Kokkos::deep_copy(exec, offsets, host_offsets); });
    auto malformed = mesh;
    malformed.connectivity.cell_nodes_offsets = offsets;
    const auto csr_error = validation::validate<MeshData::Connectivity,
        krona::mesh::AuxConnType::None, krona::mesh::AuxGeomType::None, true>(exec, malformed);
    check(csr_error.cell_nodes != 0, "Nonmonotone offsets are rejected");
    equal(csr_error.cell_incidence + csr_error.face_incidence + csr_error.node_incidence,
          std::size_t{0}, "Malformed CSR never enters incidence gathers");
}

/// @brief Halo metadata, index ranges, and duplicate destinations are separate checks.
void halo_validation(const Exec& exec)
{
    const auto source = make_mesh();
    MeshStorage<> storage(source, {.data = MeshData::Halo}, exec);
    const auto mesh = storage.view();
    const auto plan = storage.halo_plan();
    check(validation::valid_halo_plan(mesh, plan, source.rank, source.nprocs), "Valid peer plan passes");
    check(validation::validate<MeshData::Halo>(exec, mesh).ok(), "Valid halo indices pass");
    equal(validation::duplicate_halo_receives(exec, mesh), std::size_t{0}, "Unique receive destination passes");

    const std::array<std::size_t, 2> short_offsets{0, 0};
    auto invalid_plan = plan;
    invalid_plan.send_offsets = short_offsets;
    check(!validation::valid_halo_plan(mesh, invalid_plan, source.rank, source.nprocs),
          "Peer segments must cover the full send map");
    const std::array<int, 1> self_peer{source.rank};
    invalid_plan = plan;
    invalid_plan.nb_ranks = self_peer;
    check(!validation::valid_halo_plan(mesh, invalid_plan, source.rank, source.nprocs), "Self peer is rejected");

    auto duplicated = source;
    duplicated.recv_offsets = {0, 2};
    duplicated.recv_ghost_local = {2, 2};
    MeshStorage<> duplicate_storage(duplicated, {.data = MeshData::Halo}, exec);
    const auto duplicate_view = duplicate_storage.view();
    check(validation::validate<MeshData::Halo>(exec, duplicate_view).ok(), "Repeated ghost IDs are individually in range");
    check(validation::valid_halo_plan(duplicate_view, duplicate_storage.halo_plan(), duplicated.rank, duplicated.nprocs),
          "Duplicate receive destinations still have consistent segment lengths");
    equal(validation::duplicate_halo_receives(exec, duplicate_view), std::size_t{1}, "Duplicate destination is detected separately");

    auto wrong_ranges = source;
    wrong_ranges.send_owned_local = {2};
    wrong_ranges.recv_ghost_local = {1};
    MeshStorage<> range_storage(wrong_ranges, {.data = MeshData::Halo}, exec);
    const auto range_error = validation::validate<MeshData::Halo>(exec, range_storage.view());
    equal(range_error.halo_send, std::size_t{1}, "Send map cannot address a ghost");
    equal(range_error.halo_recv, std::size_t{1}, "Receive map cannot address an owned cell");
}

} // namespace

int main(int argc, char** argv)
{
    const std::array cases{
        Case{"selection", selection}, Case{"geometry", geometry},
        Case{"ownership", ownership}, Case{"halo_plan", halo_plan},
        Case{"auxiliary", auxiliary}, Case{"empty", empty},
        Case{"transfer", transfer}, Case{"auxiliary_contracts", auxiliary_contracts},
        Case{"validation", validation_cases}, Case{"halo_validation", halo_validation}};
    return krona::test::run(argc, argv, "mesh", cases);
}