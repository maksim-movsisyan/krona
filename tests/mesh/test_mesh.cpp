#include "../common/test_support.hpp"
#include "mesh_fixture.hpp"

#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

#include <array>
#include <limits>
#include <memory>
#include <type_traits>

namespace {

using namespace krona::test;
using krona::GeometryReal;
using krona::mesh::MeshPart;
using namespace krona::mesh::compute;
using Memory = typename Exec::memory_space;
 
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
    krona::mesh::MeshAuxConnectivity cache;
    cache.active_mask = krona::mesh::AuxConnType::CellFaces;
    cache.n_faces = source.n_faces;
    cache.n_cells_own = source.n_own;
    cache.n_nodes_own = source.n_nodes_own;
    cache.cell_faces_offsets = {0, 4, 8};
    cache.cell_faces = {0, 1, 2, 3, 0, 4, 5, 6};
    AuxConnectivityStorage<> storage(source, cache, exec);
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

    krona::mesh::MeshAuxGeometry metrics;
    metrics.active_mask = krona::mesh::AuxGeomType::WallDistance;
    metrics.n_faces = source.n_faces;
    metrics.cell_wall_dist = {GeometryReal{0.125}, GeometryReal{0.25}, GeometryReal{0.5}};
    AuxGeometryStorage<> metric_storage(source, metrics, exec);
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

} // namespace

int main(int argc, char** argv)
{
    const std::array cases{
        Case{"selection", selection}, Case{"geometry", geometry},
        Case{"ownership", ownership}, Case{"halo_plan", halo_plan},
        Case{"auxiliary", auxiliary}, Case{"empty", empty}};
    return krona::test::run(argc, argv, "mesh", cases);
}
