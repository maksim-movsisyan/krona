#include "fixtures.hpp"
#include "common/test_common_host_only.hpp"

#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/mesh/host/aux_geometry.hpp>
#include <krona/mesh/host/extrusion.hpp>
#include <krona/mesh/host/validate.hpp>
#include <krona/mpi/collectives.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

namespace host = krona::mesh::host;
namespace fixture = krona::test::mesh_host;
namespace mpi = krona::mpi;
using krona::GeometryReal;
using krona::GlobalIndex;
using krona::LocalIndex;
using krona::mesh::AuxConnType;
using krona::mesh::AuxGeomType;
using krona::test::host::Case;
using krona::test::host::check;
using krona::test::host::equal;
using krona::test::host::near;
constexpr double tolerance = 256 * static_cast<double>(std::numeric_limits<GeometryReal>::epsilon());

/// @brief Compares a small analytic reference using the configured geometry precision.
void close(double actual, double expected, std::string_view message) {
    near(actual, expected, tolerance * std::max(1.0, std::abs(expected)), message);
}

/// @brief Extracts one CSR row for order-sensitive reference comparisons.
std::vector<LocalIndex> row(const std::vector<std::size_t>& offsets, const std::vector<LocalIndex>& values,
                            std::size_t index) {
    return {values.begin() + static_cast<std::ptrdiff_t>(offsets[index]),
            values.begin() + static_cast<std::ptrdiff_t>(offsets[index + 1])};
}

/// @brief Sends cell identities through the actual halo maps and checks remote slot matching.
void check_halo(const host::MeshPart& mesh, mpi::CommView comm) {
    std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
    for (std::size_t p = 0; p < mesh.nb_ranks.size(); ++p)
        counts[static_cast<std::size_t>(mesh.nb_ranks[p])] =
            static_cast<int>(mesh.send_offsets[p + 1] - mesh.send_offsets[p]);
    std::vector<GlobalIndex> send;
    for (const auto local : mesh.send_owned_local)
        send.push_back(mesh.cell_gid[static_cast<std::size_t>(local)]);
    std::vector<GlobalIndex> receive;
    mpi::AlltoallvWorkspace exchange;
    mpi::alltoallv_packed(comm, std::span<const int>{counts}, std::span<const GlobalIndex>{send}, receive,
                          exchange);
    equal(receive.size(), mesh.recv_ghost_local.size(), "Every ghost has exactly one received value");
    for (std::size_t p = 0; p < mesh.nb_ranks.size(); ++p)
        equal(exchange.recv_counts[static_cast<std::size_t>(mesh.nb_ranks[p])],
              static_cast<int>(mesh.recv_offsets[p + 1] - mesh.recv_offsets[p]),
              "Per-peer halo counts agree");
    for (std::size_t i = 0; i < receive.size(); ++i)
        equal(receive[i], mesh.cell_gid[static_cast<std::size_t>(mesh.recv_ghost_local[i])],
              "Halo slot receives its ghost's global cell ID");
}

/// @brief Checks a migrated three-cube chain against geometry and ownership formulas.
void check_chain(const host::MeshPart& mesh, mpi::CommView comm, bool rotate = true) {
    equal(mesh.n_cells_g, std::uint64_t{3}, "Migration preserves global cell count");
    equal(mesh.n_nodes_g, std::uint64_t{16}, "Migration preserves global source nodes");
    equal(mesh.n_faces_g, std::uint64_t{16}, "Three cubes have 16 unique faces");
    equal(mesh.n_bfaces_g, std::uint64_t{14}, "Three cubes have 14 exterior faces");
    check(mesh.rank == comm.rank() && mesh.nprocs == comm.size(), "Mesh uses the supplied communicator");
    const auto owner = [&](GlobalIndex gid) {
        return (static_cast<int>(gid % comm.size()) + static_cast<int>(rotate)) % comm.size();
    };
    std::set<GlobalIndex> expected_owned, expected_ghosts, actual_owned, actual_ghosts;
    for (GlobalIndex gid = 0; gid < 3; ++gid)
        if (owner(gid) == comm.rank())
            expected_owned.insert(gid);
    for (const auto gid : expected_owned)
        for (const auto other : {gid - 1, gid + 1})
            if (other >= 0 && other < 3 && owner(other) != comm.rank())
                expected_ghosts.insert(other);
    std::set<GlobalIndex> owned_nodes;
    for (std::size_t c = 0; c < mesh.cell_gid.size(); ++c) {
        const auto gid = mesh.cell_gid[c];
        const bool owned = c < static_cast<std::size_t>(mesh.n_own);
        (owned ? actual_owned : actual_ghosts).insert(gid);
        equal(mesh.cell_donor[c], owned ? -1 : owner(gid), "Ghost donor names its destination owner");
        const GlobalIndex left = 4 * gid, right = left + 4;
        const std::array<GlobalIndex, 8> reference{left,     right,     right + 1, left + 1,
                                                   left + 3, right + 3, right + 2, left + 2};
        equal(mesh.cell_nodes_offsets[c + 1] - mesh.cell_nodes_offsets[c], std::size_t{8},
              "Migrated hexahedron retains eight nodes");
        for (std::size_t j = 0; j < reference.size(); ++j) {
            const auto node = mesh.cell_nodes[mesh.cell_nodes_offsets[c] + j];
            equal(mesh.node_gid[static_cast<std::size_t>(node)], reference[j],
                  "Local remapping preserves oriented global connectivity");
            if (owned)
                owned_nodes.insert(reference[j]);
        }
        if (!mesh.cell_volume.empty()) {
            close(mesh.cell_volume[c], 1, "Unit-cube volume survives migration and reordering");
            close(mesh.cell_centroid_x[c], static_cast<double>(gid) + 0.5, "Cube centroid x");
            close(mesh.cell_centroid_y[c], 0.5, "Cube centroid y");
            close(mesh.cell_centroid_z[c], 0.5, "Cube centroid z");
        }
    }
    check(actual_owned == expected_owned && actual_owned.size() == static_cast<std::size_t>(mesh.n_own),
          "Exactly the requested cells are owned once");
    check(actual_ghosts == expected_ghosts &&
              actual_ghosts.size() == static_cast<std::size_t>(mesh.n_cells - mesh.n_own),
          "Only distinct face-neighbour ghosts are retained");
    equal(static_cast<std::size_t>(mesh.n_nodes_own), owned_nodes.size(), "Owned-used node prefix is exact");
    for (std::size_t n = 0; n < mesh.node_gid.size(); ++n) {
        const auto gid = mesh.node_gid[n];
        const auto point = fixture::chain_point(gid);
        check(owned_nodes.contains(gid) == (n < static_cast<std::size_t>(mesh.n_nodes_own)),
              "Ghost-only nodes follow all owned-used nodes");
        close(mesh.node_x[n], point[0], "Remote node x was fetched correctly");
        close(mesh.node_y[n], point[1], "Remote node y was fetched correctly");
        close(mesh.node_z[n], point[2], "Remote node z was fetched correctly");
    }
    check_halo(mesh, comm);
}

/// @brief Checks exact distributed face counts, patch labels and the symmetric chain graph.
void topology(mpi::CommView comm) {
    const auto input = fixture::chain(comm);
    const auto result = host::build_faces(input, comm);
    equal(result.stats.faces, std::uint64_t{16}, "Unique faces: 5*N+1");
    equal(result.stats.interior_faces, std::uint64_t{2}, "Interior faces: N-1");
    equal(result.stats.boundary_faces, std::uint64_t{14}, "Boundary faces: 4*N+2");
    equal(result.stats.graph_arcs, std::uint64_t{4}, "Each shared face contributes two graph arcs");
    std::set<host::FaceKey> keys;
    for (const auto& face : result.faces) {
        equal(input.distribution.cells.owner(face.cell_a), comm.rank(), "Face belongs to cell_a's rank");
        check(keys.insert(face.key).second, "Each local face key occurs once");
        if (face.cell_b == host::invalid_cell_id) {
            const auto patch = face.cell_a == 0 && face.local_face_a == 4 ? 0 : 1;
            equal(face.patch, patch, "Boundary label survives distributed matching");
        } else {
            equal(face.cell_b, face.cell_a + 1, "Only neighbouring chain cells share a face");
            equal(face.patch, krona::mesh::invalid_patch_id, "Interior faces have no boundary patch");
            check(face.local_face_a == 2 && face.local_face_b == 4, "Shared local facets are right/left");
        }
    }
    for (std::size_t c = 0; c < input.cells.size(); ++c) {
        const auto gid = input.global_cell_id(static_cast<LocalIndex>(c));
        std::vector<GlobalIndex> expected;
        if (gid > 0)
            expected.push_back(gid - 1);
        if (gid < 2)
            expected.push_back(gid + 1);
        const auto begin = result.graph.offsets[c], end = result.graph.offsets[c + 1];
        check(std::vector<GlobalIndex>(result.graph.adjacency.begin() + static_cast<std::ptrdiff_t>(begin),
                                       result.graph.adjacency.begin() + static_cast<std::ptrdiff_t>(end)) ==
                  expected,
              "CSR row contains exactly the sorted adjacent global cell IDs");
    }
}

/// @brief Forces many matching rounds and compares them with the default batch layout.
void batching(mpi::CommView comm) {
    const auto input = fixture::chain(comm);
    const auto ordinary = host::build_faces(input, comm);
    const auto tiny = host::build_faces(input, comm, {.batch_bytes = 128});
    check(ordinary.graph.offsets == tiny.graph.offsets && ordinary.graph.adjacency == tiny.graph.adjacency,
          "Small communication rounds do not alter the dual graph");
    using Face = std::tuple<host::FaceKey, GlobalIndex, GlobalIndex, krona::mesh::PatchId>;
    std::set<Face> a, b;
    for (const auto& f : ordinary.faces)
        a.emplace(f.key, f.cell_a, f.cell_b, f.patch);
    for (const auto& f : tiny.faces)
        b.emplace(f.key, f.cell_a, f.cell_b, f.patch);
    check(a == b, "Batching preserves unique face records independently of vector order");
}

/// @brief Confirms that missing labels preserve exterior faces without creating a patch.
void unlabelled(mpi::CommView comm) {
    auto part = fixture::migrate(fixture::chain(comm, 3, false), comm);
    (void)host::compute_mesh_geometry(part, comm);
    host::reorder_local_mesh(part);
    host::validate_meshpart(part);
    const auto stats = host::collect_mesh_statistics(part, comm);
    equal(stats.boundary_faces, std::uint64_t{14}, "Unlabelled exterior faces remain boundaries");
    equal(stats.labelled_faces, std::uint64_t{0}, "No implicit physical patch is assigned");
    check(part.patch_faces.empty(), "Unlabelled faces do not enter patch CSR");
}

/// @brief Moves cells between ranks and checks ghosts, coordinates and actual halo exchanges.
void migration(mpi::CommView comm) {
    auto part = fixture::migrate(fixture::chain(comm), comm);
    check(!part.faces_grouped && part.cell_volume.empty(), "Migration leaves grouping and metrics explicit");
    check_chain(part, comm);
    (void)host::compute_mesh_geometry(part, comm);
    host::reorder_local_mesh(part);
    host::validate_meshpart(part);
    const auto stats = host::collect_mesh_statistics(part, comm);
    equal(stats.owned_cells, std::uint64_t{3}, "Global ownership is conserved");
    equal(stats.patch_face_counts[0], std::uint64_t{1}, "The inlet wall has one face");
    equal(stats.patch_face_counts[1], std::uint64_t{13}, "Other labelled faces are preserved");
    close(stats.total_volume, 3, "Ghost volumes are excluded from the owned-cell total");
    check_chain(part, comm);
}

/// @brief Migrates explicit unequal ownership with all input payloads initially on rank zero.
void unequal_ownership(mpi::CommView comm) {
    host::MigrationSource input;
    std::vector<host::FaceRecord> faces;
    if (comm.rank() == 0) {
        // A self communicator builds the full reference input only on this process.
        auto raw = fixture::chain(mpi::CommView::self());
        auto topology = host::build_faces(raw, mpi::CommView::self());
        faces = std::move(topology.faces);
        input.cells = std::move(raw.cells);
        input.coordinates = std::move(raw.coordinates);
    }
    input.patches = {{"wall", "solid", "BCWall"}, {"outer", "", ""}};
    input.cell_distribution.assign(static_cast<std::size_t>(comm.size()) + 1, 3);
    input.node_distribution.assign(static_cast<std::size_t>(comm.size()) + 1, 16);
    input.cell_distribution[0] = input.node_distribution[0] = 0;
    std::vector<int> targets(input.cells.size());
    for (std::size_t c = 0; c < targets.size(); ++c)
        targets[c] = static_cast<int>(c % comm.size());
    auto part = host::migrate_local_mesh(std::move(input), std::move(faces), targets, comm);
    (void)host::compute_mesh_geometry(part, comm);
    host::reorder_local_mesh(part);
    host::validate_meshpart(part);
    check_chain(part, comm, false);
}

/// @brief Checks analytic volume and centroids for every supported volume-cell type.
void geometry(mpi::CommView comm) {
    auto part = fixture::migrate(fixture::mixed(comm), comm);
    const auto stats = host::compute_mesh_geometry(part, comm);
    constexpr std::array<double, 4> volumes{1.0 / 6, 1.0 / 3, 0.5, 1};
    constexpr std::array<std::array<double, 3>, 4> centres{
        {{0.25, 0.25, 0.25}, {0.5, 0.5, 0.25}, {1.0 / 3, 1.0 / 3, 0.5}, {0.5, 0.5, 0.5}}};
    for (std::size_t c = 0; c < part.cell_gid.size(); ++c) {
        const auto gid = static_cast<std::size_t>(part.cell_gid[c]);
        close(part.cell_volume[c], volumes[gid], "Supported element has its analytic volume");
        close(part.cell_centroid_x[c], 3.0 * static_cast<double>(gid) + centres[gid][0],
              "Analytic centroid x");
        close(part.cell_centroid_y[c], centres[gid][1], "Analytic centroid y");
        close(part.cell_centroid_z[c], centres[gid][2], "Analytic centroid z");
    }
    close(stats.total_volume, 2, "Mixed reference volumes sum to two");
    close(stats.bbox_lo[0], 0, "Global lower x bound");
    close(stats.bbox_hi[0], 10, "Global upper x bound");
    for (std::size_t f = 0; f < part.face_area.size(); ++f) {
        check(part.face_area[f] > 0, "Every supported facet has positive area");
        close(std::hypot(part.face_normal_x[f], part.face_normal_y[f], part.face_normal_z[f]), 1,
              "Stored face normals are unit vectors");
    }
    host::reorder_local_mesh(part);
    host::validate_meshpart(part);
    const auto global = host::collect_mesh_statistics(part, comm);
    equal(global.boundary_faces, std::uint64_t{20}, "Isolated TET/PYRA/PRISM/HEXA have 20 facets");
}

/// @brief Separates tiny physical measures from translation-sensitive centroid arithmetic.
void geometry_scales(mpi::CommView comm) {
    for (const double scale : {1.0e-6, 0.5}) {
        auto raw = fixture::chain(comm);
        const double shift = scale == 0.5 ? (sizeof(GeometryReal) == 8 ? 1099511627776.0 : 256.0) : 0;
        for (auto* field : {&raw.coordinates.x, &raw.coordinates.y, &raw.coordinates.z})
            for (auto& value : *field)
                value = static_cast<GeometryReal>(shift + scale * value);
        auto part = fixture::migrate(std::move(raw), comm);
        const auto stats = host::compute_mesh_geometry(part, comm);
        const double volume = scale * scale * scale;
        near(stats.total_volume, 3 * volume, 32 * tolerance * volume,
             "Volume is stable across scale and translation");
        for (const auto stored : part.cell_volume)
            near(static_cast<double>(stored), volume, 32 * tolerance * volume,
                 "Tiny positive volume is retained");
        host::reorder_local_mesh(part);
        host::validate_meshpart(part);
    }
}

/// @brief Verifies NONE, Hilbert, RCM and explicit permutations preserve all entity references.
void reordering(mpi::CommView comm) {
    const auto original = fixture::prepared_chain(comm);
    constexpr std::array<host::ReorderMethod, 3> methods{
        host::ReorderMethod::NONE, host::ReorderMethod::HILBERT_SFC, host::ReorderMethod::RCM};
    for (const auto method : methods) {
        auto part = original;
        host::reorder_local_mesh(part, method);
        host::validate_meshpart(part);
        check_chain(part, comm);
        check(std::equal(original.cell_gid.begin() + original.n_own, original.cell_gid.end(),
                         part.cell_gid.begin() + part.n_own),
              "Automatic ordering preserves ghost order");
    }
    auto part = original;
    std::vector<LocalIndex> permutation(static_cast<std::size_t>(part.n_own));
    std::iota(permutation.rbegin(), permutation.rend(), LocalIndex{0});
    host::reorder_local_mesh(part, std::span<const LocalIndex>{permutation});
    for (std::size_t c = 0; c < permutation.size(); ++c)
        equal(part.cell_gid[c], original.cell_gid[static_cast<std::size_t>(permutation[c])],
              "Caller permutation is interpreted as new-to-old");
    host::validate_meshpart(part);
    check_chain(part, comm);
}

/// @brief Checks optional CSR contents using independent set-based incidence queries.
void auxiliary_connectivity(mpi::CommView comm) {
    const auto part = fixture::prepared_chain(comm);
    auto only = host::build_aux_connectivity(part, AuxConnType::CellCellsByNode);
    check(!only.has_node_cells() && only.node_cells.empty(),
          "Temporary dependencies do not become requested outputs");
    auto aux = host::build_aux_connectivity(part, AuxConnType::All);
    for (std::size_t c = 0; c < static_cast<std::size_t>(part.n_own); ++c) {
        equal(row(aux.cell_faces_offsets, aux.cell_faces, c).size(), std::size_t{6},
              "Owned hexahedron has six incident faces");
        std::vector<LocalIndex> neighbours;
        for (std::size_t other = 0; other < part.cell_gid.size(); ++other)
            if (std::abs(part.cell_gid[c] - part.cell_gid[other]) == 1)
                neighbours.push_back(static_cast<LocalIndex>(other));
        check(row(aux.cell_cells_face_offsets, aux.cell_cells_face, c) == neighbours,
              "Face-neighbour CSR contains owned and available ghost neighbours");
        check(row(aux.cell_cells_node_offsets, aux.cell_cells_node, c) == neighbours,
              "Node-neighbour CSR is unique, sorted and excludes self");
        check(row(only.cell_cells_node_offsets, only.cell_cells_node, c) == neighbours,
              "Temporary node transpose produces the same vertex stencil");
    }
    for (std::size_t node = 0; node < static_cast<std::size_t>(part.n_nodes_own); ++node) {
        std::vector<LocalIndex> expected_cells, expected_faces;
        // Brute-force references avoid reproducing the production CSR count/fill algorithm.
        for (std::size_t c = 0; c < part.cell_gid.size(); ++c) {
            const auto nodes = std::span<const LocalIndex>{part.cell_nodes}.subspan(
                part.cell_nodes_offsets[c], part.cell_nodes_offsets[c + 1] - part.cell_nodes_offsets[c]);
            if (std::find(nodes.begin(), nodes.end(), static_cast<LocalIndex>(node)) != nodes.end())
                expected_cells.push_back(static_cast<LocalIndex>(c));
        }
        for (std::size_t f = 0; f < part.face_owner.size(); ++f) {
            const auto nodes = std::span<const LocalIndex>{part.face_nodes}.subspan(
                part.face_nodes_offsets[f], part.face_nodes_offsets[f + 1] - part.face_nodes_offsets[f]);
            if (std::find(nodes.begin(), nodes.end(), static_cast<LocalIndex>(node)) != nodes.end())
                expected_faces.push_back(static_cast<LocalIndex>(f));
        }
        check(row(aux.node_cells_offsets, aux.node_cells, node) == expected_cells,
              "Owned-used node rows include every available cell, including ghosts");
        check(row(aux.node_faces_offsets, aux.node_faces, node) == expected_faces,
              "Owned-used node rows contain exactly their local faces");
    }
    const auto* retained = aux.cell_faces.data();
    aux.add_connectivity(part, AuxConnType::CellFaces);
    check(aux.cell_faces.data() == retained, "Already cached connectivity is retained");
    aux.clear();
    check(aux.active_mask == AuxConnType::None && aux.cell_faces.empty(),
          "Clearing invalidates all connectivity groups");
}

/// @brief Checks orthogonal distance, interpolation and nonorthogonality formulas independently.
void auxiliary_geometry(mpi::CommView comm) {
    const auto part = fixture::prepared_chain(comm);
    auto aux = host::build_aux_geometry(part, AuxGeomType::All);
    check(!aux.has_wall_dist(), "Local face construction does not advertise collective wall distance");
    for (std::size_t f = 0; f < part.face_owner.size(); ++f) {
        const bool interior = f < static_cast<std::size_t>(part.n_inner_faces);
        const double distance = interior ? 1 : 0.5;
        close(aux.face_cell_dist[f], distance, "Analytic centre separation");
        close(aux.face_normal_dist[f], distance, "Orthogonal projection equals separation");
        close(aux.face_cell_dist_inv[f], 1 / distance, "Reciprocal centre separation");
        close(aux.face_normal_dist_inv[f], 1 / distance, "Reciprocal normal separation");
        close(aux.face_area_over_normal_dist[f], 1 / distance, "Unit-face area divided by normal separation");
        close(aux.face_interp_weight[f], interior ? 0.5 : 0,
              "Symmetric interpolation and boundary endpoint weight");
        close(std::hypot(aux.face_cell_dist_x[f], aux.face_cell_dist_y[f], aux.face_cell_dist_z[f]), distance,
              "Stored displacement components match their norm");
        close(aux.face_non_ortho_x[f], 0, "Orthogonal correction x");
        close(aux.face_non_ortho_y[f], 0, "Orthogonal correction y");
        close(aux.face_non_ortho_z[f], 0, "Orthogonal correction z");
    }
    aux.clear();
    aux.add_geometry(part, AuxGeomType::FaceNormalDistanceInv);
    check(aux.has_face_normal_dist_inv() && aux.face_cell_dist.empty() && aux.face_normal_dist.empty(),
          "Requesting a derived metric allocates only its explicit output");
}

/// @brief Propagates one wall seed across rank interfaces and checks unreachable reporting.
void wall_distance(mpi::CommView comm) {
    const auto part = fixture::prepared_chain(comm);
    host::MeshAuxConnectivity connectivity;
    host::MeshAuxGeometry geometry;
    const std::array<std::uint8_t, 2> wall{1, 0};
    constexpr std::array<host::WallDistStencil, 2> stencils{host::WallDistStencil::FaceNeighbors,
                                                            host::WallDistStencil::NodeNeighbors};
    for (const auto stencil : stencils) {
        const auto stats = geometry.add_wall_distance(part, connectivity, comm, wall, {.stencil = stencil});
        check(stats.converged && stats.unreachable_owned == 0, "Wall paths converge on every rank");
        for (std::size_t c = 0; c < part.cell_gid.size(); ++c)
            close(geometry.cell_wall_dist[c], static_cast<double>(part.cell_gid[c]) + 0.5,
                  "Owned and final ghost distances match the analytic path from x=0");
    }
    const std::array<std::uint8_t, 2> none{0, 0};
    const auto missing = geometry.add_wall_distance(part, connectivity, comm, none);
    equal(missing.unreachable_owned, std::uint64_t{3},
          "No wall reports all globally owned cells as unreachable");
    for (const auto distance : geometry.cell_wall_dist)
        check(std::isinf(distance), "Unreachable distance stays infinite");
}

/// @brief Extrudes connected QUAD/TRI inputs in horizontal and rotated physical planes.
void extrusion(mpi::CommView comm) {
    for (const bool tilted : {false, true}) {
        auto input = fixture::planar(comm, tilted);
        const auto info =
            host::extrude_one_layer(input, comm, {.thickness_fraction = 0.25, .batch_elements = 1});
        close(info.thickness, 0.5, "Extrusion thickness follows the longest span");
        close(std::hypot(info.normal[0], info.normal[1], info.normal[2]), 1,
              "Automatic extrusion direction is unit length");
        equal(input.cell_dimension, 3, "Extruded cells are three-dimensional");
        equal(input.distribution.nodes.global_size(), GlobalIndex{10},
              "Top-layer IDs follow five original nodes");
        equal(input.distribution.boundary_ordinals.global_size(), GlobalIndex{9},
              "Five sides and four end caps");
        equal(input.patches.size(), std::size_t{3}, "End caps add two named patches");
        check(input.patches[1].name == "Z_MIN" && input.patches[2].name == "Z_MAX",
              "Default end-cap labels survive rotation");
        for (std::size_t c = 0; c < input.cells.size(); ++c) {
            const auto gid = input.global_cell_id(static_cast<LocalIndex>(c));
            check(input.cells.types[c] ==
                      (gid == 0 ? krona::mesh::CellType::HEXA : krona::mesh::CellType::PRISM),
                  "QUAD and TRI become HEXA and PRISM respectively");
        }
        auto part = fixture::migrate(std::move(input), comm);
        const auto stats = host::compute_mesh_geometry(part, comm);
        close(stats.total_volume, 0.75, "Extruded volume is planar area times thickness");
        host::reorder_local_mesh(part);
        host::validate_meshpart(part);
        const auto global = host::collect_mesh_statistics(part, comm);
        equal(global.boundary_faces, std::uint64_t{9}, "All extruded boundary elements match volume facets");
    }
}

/// @brief Runs the full host pipeline on an empty global mesh and empty auxiliary tables.
void empty(mpi::CommView comm) {
    auto input = fixture::distribute(comm, {}, {}, {});
    auto part = fixture::migrate(std::move(input), comm);
    const auto geometry = host::compute_mesh_geometry(part, comm);
    close(geometry.total_volume, 0, "Empty global volume is zero");
    close(geometry.min_cell_volume, 0, "Empty global extrema are normalized to zero");
    host::reorder_local_mesh(part);
    host::validate_meshpart(part);
    const auto stats = host::collect_mesh_statistics(part, comm);
    equal(stats.owned_cells, std::uint64_t{0}, "Empty global ownership is valid");
    auto connectivity = host::build_aux_connectivity(part, AuxConnType::All);
    check(connectivity.cell_faces_offsets == std::vector<std::size_t>{0} &&
              connectivity.node_cells_offsets == std::vector<std::size_t>{0},
          "Empty CSR retains its zero endpoint");
    auto aux = host::build_aux_geometry(part, AuxGeomType::All);
    const auto wall = aux.add_wall_distance(part, connectivity, comm, {});
    check(wall.converged && wall.unreachable_owned == 0, "Empty ranks participate in wall convergence");
}

/// @brief Rejects three volume cells matched to the same geometric facets.
void reject_nonmanifold(mpi::CommView comm) {
    auto input = fixture::chain(comm, 3, false);
    const std::array<GlobalIndex, 8> nodes{0, 4, 5, 1, 3, 7, 6, 2};
    for (std::size_t c = 0; c < input.cells.size(); ++c)
        std::copy(nodes.begin(), nodes.end(),
                  input.cells.node_ids.begin() + static_cast<std::ptrdiff_t>(8 * c));
    (void)host::build_faces(input, comm);
    check(false, "A nonmanifold face was accepted");
}

/// @brief Rejects a boundary label attached to a shared interior face.
void reject_internal_boundary(mpi::CommView comm) {
    auto input = fixture::chain(comm);
    if (comm.rank() == 0) {
        const std::array<GlobalIndex, 4> nodes{4, 5, 6, 7};
        std::copy(nodes.begin(), nodes.end(), input.boundary.elements.node_ids.begin());
    }
    (void)host::build_faces(input, comm);
    check(false, "An interior boundary label was accepted");
}

/// @brief Rejects an inverted tetrahedron during actual host metric integration.
void reject_inverted(mpi::CommView comm) {
    auto input = fixture::mixed(comm);
    if (comm.rank() == 0)
        std::swap(input.cells.node_ids[1], input.cells.node_ids[2]);
    auto part = fixture::migrate(std::move(input), comm);
    (void)host::compute_mesh_geometry(part, comm);
    check(false, "An inverted cell was accepted");
}

/// @brief Rejects a repeated owned-cell entry in a caller permutation.
void reject_permutation(mpi::CommView) {
    auto part = fixture::prepared_chain(mpi::CommView::self());
    const std::array<LocalIndex, 3> invalid{0, 0, 2};
    host::reorder_local_mesh(part, std::span<const LocalIndex>{invalid});
    check(false, "A nonbijective permutation was accepted");
}

/// @brief Rejects a halo receive that points into the owned-cell prefix.
void reject_halo(mpi::CommView comm) {
    auto part = fixture::prepared_chain(comm);
    part.recv_ghost_local[0] = 0;
    host::validate_meshpart(part);
    check(false, "An owned-cell halo receive was accepted");
}

} // namespace

/// @brief Runs healthy cases by default and explicit fatal-contract cases by name.
int main(int argc, char** argv) {
    constexpr std::array cases{Case{"topology", topology},
                               Case{"batching", batching},
                               Case{"unlabelled", unlabelled},
                               Case{"migration", migration},
                               Case{"unequal_ownership", unequal_ownership},
                               Case{"geometry", geometry},
                               Case{"geometry_scales", geometry_scales},
                               Case{"reordering", reordering},
                               Case{"auxiliary_connectivity", auxiliary_connectivity},
                               Case{"auxiliary_geometry", auxiliary_geometry},
                               Case{"wall_distance", wall_distance},
                               Case{"extrusion", extrusion},
                               Case{"empty", empty}};
    constexpr std::array rejections{
        Case{"reject_nonmanifold", reject_nonmanifold},
        Case{"reject_internal_boundary", reject_internal_boundary}, Case{"reject_inverted", reject_inverted},
        Case{"reject_permutation", reject_permutation}, Case{"reject_halo", reject_halo}};
    const bool reject = argc == 2 && std::string_view(argv[1]).starts_with("reject_");
    return krona::test::host::run(argc, argv, "mesh.host",
                                  reject ? std::span<const Case>{rejections} : std::span<const Case>{cases});
}
