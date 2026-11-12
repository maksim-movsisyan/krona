#include "driver.hpp"
#include "cli.hpp"

#include <krona/toml_config/file.hpp>
#include <krona/io/boundary_config/writer.hpp>
#include <krona/io/boundary_config/profile_template_writer.hpp>
#include <krona/io/cgns/reader.hpp>
#include <krona/io/solver_mesh/hdf5_writer.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/mesh/host/faces.hpp>
#include <krona/mesh/host/geometry.hpp>
#include <krona/mesh/host/migrate.hpp>
#include <krona/mesh/host/validate.hpp>
#include <krona/mesh/partition/partition.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

namespace krona::apps::mesh_partitioner {
namespace {

using Clock = std::chrono::steady_clock;

/// @brief Returns elapsed local wall time without synchronizing MPI ranks.
double seconds_since(Clock::time_point started) noexcept {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

/// @brief Replicates rank-zero settings with the common chunked string broadcast.
void broadcast_options(Options& options, mpi::CommView comm) {
    std::array<int, 2> settings{static_cast<int>(options.reorder), static_cast<int>(options.verbosity)};
    mpi::broadcast(comm, std::span{settings});
    toml_config::broadcast(options.input, comm);
    toml_config::broadcast(options.output, comm);
    toml_config::broadcast(options.boundary_output, comm);
    toml_config::broadcast(options.vtu_directory, comm);
    toml_config::broadcast(options.patches_template_directory, comm);
    options.reorder = static_cast<mesh::host::ReorderMethod>(settings[0]);
    options.verbosity = static_cast<Verbosity>(settings[1]);
}

/// @brief Prints source metadata, optional extrusion and imported sizes.
void print_import(const io::cgns::ReadResult& imported, mpi::CommView comm, bool detailed, double seconds) {
    const auto& mesh = imported.mesh;
    std::fprintf(stderr, "  Mesh: cells=%lld nodes=%lld boundary_elements=%lld patches=%zu ranks=%d\n",
                 static_cast<long long>(mesh.distribution.cells.global_size()),
                 static_cast<long long>(mesh.distribution.nodes.global_size()),
                 static_cast<long long>(mesh.distribution.boundary_ordinals.global_size()),
                 mesh.patches.size(), comm.size());

    if (imported.extrusion) {
        const auto& extrusion = *imported.extrusion;
        std::fprintf(stderr, "  Extrusion: 2D -> 3D, one layer, thickness=%.6g normal=(%.6g, %.6g, %.6g)\n",
                     static_cast<double>(extrusion.thickness), extrusion.normal[0], extrusion.normal[1],
                     extrusion.normal[2]);
    }

    if (detailed) {
        std::fprintf(stderr, "  Source: CGNS=%.2f base=\"%s\" zone=\"%s\" cell_dim=%d physical_dim=%d\n",
                     static_cast<double>(imported.source.version), imported.source.base_name.c_str(),
                     imported.source.zone_name.c_str(), imported.source.cell_dimension,
                     imported.source.physical_dimension);

        if (!mesh.patches.empty())
            std::fputs("  Patches:\n", stderr);

        for (std::size_t i = 0; i < mesh.patches.size(); ++i) {
            const auto& patch = mesh.patches[i];
            std::fprintf(stderr, "    [%zu] name=\"%s\" family=\"%s\" source=\"%s\"\n", i, patch.name.c_str(),
                         patch.family.empty() ? "-" : patch.family.c_str(),
                         patch.source_type.empty() ? "-" : patch.source_type.c_str());
        }

        for (const auto& name : imported.source.ignored_boundary_conditions)
            std::fprintf(stderr, "  Ignored BC: \"%s\"\n", name.c_str());
    }

    std::fprintf(stderr, "KRONA mesh_partitioner: import completed in %.3f s.\n", seconds);
}

/// @brief Prints existing global face counts and local dual-graph sizes.
void print_topology(const mesh::host::RawMesh& mesh, const mesh::host::BuildFacesResult& topology,
                    bool detailed, double seconds) {
    const auto& faces = topology.faces;
    const auto& graph = topology.graph;
    const auto& stats = topology.stats;

    std::fprintf(stderr, "  Faces: total=%llu interior=%llu boundary=%llu\n",
                 static_cast<unsigned long long>(stats.faces),
                 static_cast<unsigned long long>(stats.interior_faces),
                 static_cast<unsigned long long>(stats.boundary_faces));

    std::fprintf(stderr, "  Dual graph: vertices=%lld directed_arcs=%llu\n",
                 static_cast<long long>(mesh.distribution.cells.global_size()),
                 static_cast<unsigned long long>(stats.graph_arcs));

    if (detailed) {
        std::fprintf(stderr, "  Rank 0 owns: faces=%zu graph_rows=%zu graph_arcs=%zu\n", faces.size(),
                     mesh.cells.size(), graph.adjacency.size());
        std::fputs("  Face ownership: rank owning cell_a (smaller cell ID).\n", stderr);
        std::fputs("  CSR neighbours: sorted, unique global cell IDs.\n", stderr);
    }

    std::fprintf(stderr, "KRONA mesh_partitioner: topology completed in %.3f s.\n", seconds);
}

/// @brief Prints planned rank loads and the partition-to-rank placement.
void print_partition(const mesh::host::RawMesh& mesh, const mesh::partition::PartitionResult& pr,
                     const mesh::partition::DKaMinParOptions& partition_options, mpi::CommView comm,
                     bool detailed, double seconds) {
    const auto [minimum, maximum] =
        std::minmax_element(pr.vertices_per_rank.begin(), pr.vertices_per_rank.end());
    const double average = static_cast<double>(mesh.distribution.cells.global_size()) / comm.size();
    const double imbalance = average > 0 ? 100.0 * (static_cast<double>(*maximum) / average - 1.0) : 0.0;

    std::fprintf(stderr, "  Partition: blocks=%d edge_cut=%llu\n", comm.size(),
                 static_cast<unsigned long long>(pr.global_edge_cut));
    std::fprintf(stderr, "  Planned cells/rank: min=%llu avg=%.1f max=%llu imbalance=%.2f%%\n",
                 static_cast<unsigned long long>(*minimum), average,
                 static_cast<unsigned long long>(*maximum), imbalance);
    if (pr.shared_memory_groups > 0)
        std::fprintf(stderr, "  Placement: shared-memory grouping, groups=%d\n", pr.shared_memory_groups);
    else
        std::fputs("  Placement: MPI rank order.\n", stderr);

    if (detailed) {
        const auto shown = std::min<std::size_t>(pr.vertices_per_rank.size(), 16);
        std::fprintf(stderr, "  Settings: epsilon=%.4f threads/rank=%d seed=%d\n",
                     partition_options.imbalance_tolerance, partition_options.threads_per_rank,
                     partition_options.seed);
        std::fprintf(stderr, "  Rank placement and planned load (first %zu ranks):\n", shown);
        for (std::size_t rank = 0; rank < shown; ++rank)
            std::fprintf(stderr, "    rank=%zu block=%u cells=%llu\n", rank,
                         static_cast<unsigned int>(pr.rank2part[rank]),
                         static_cast<unsigned long long>(pr.vertices_per_rank[rank]));
        if (shown < pr.vertices_per_rank.size())
            std::fprintf(stderr, "    ... %zu further ranks omitted.\n", pr.vertices_per_rank.size() - shown);
        std::fputs("  Destinations computed; mesh migration has not started.\n", stderr);
    }
    std::fprintf(stderr, "KRONA mesh_partitioner: partitioning completed in %.3f s.\n", seconds);
}

/// @brief Prints migrated counts and the face-neighbour halo footprint.
void print_migration(const mesh::host::MeshPart& mp, bool detailed, double seconds) {
    const double halo_percent = mp.n_cells_g > 0 ? 100.0 * static_cast<double>(mp.n_ghost_copies_g) /
                                                       static_cast<double>(mp.n_cells_g)
                                                 : 0.0;
    std::fprintf(stderr, "  Migrated: cells=%llu source_nodes=%llu unique_faces=%llu\n",
                 static_cast<unsigned long long>(mp.n_cells_g), static_cast<unsigned long long>(mp.n_nodes_g),
                 static_cast<unsigned long long>(mp.n_faces_g));
    std::fprintf(stderr, "  Faces: boundary=%llu labeled=%llu partition_cut=%llu\n",
                 static_cast<unsigned long long>(mp.n_bfaces_g),
                 static_cast<unsigned long long>(mp.n_labeled_bfaces_g),
                 static_cast<unsigned long long>(mp.n_cut_faces_g));
    std::fprintf(stderr, "  Halo: ghost_copies=%llu overhead=%.2f%%\n",
                 static_cast<unsigned long long>(mp.n_ghost_copies_g), halo_percent);
    std::fprintf(stderr, "  Bounds: x=[%.6g, %.6g] y=[%.6g, %.6g] z=[%.6g, %.6g]\n",
                 static_cast<double>(mp.bbox_lo[0]), static_cast<double>(mp.bbox_hi[0]),
                 static_cast<double>(mp.bbox_lo[1]), static_cast<double>(mp.bbox_hi[1]),
                 static_cast<double>(mp.bbox_lo[2]), static_cast<double>(mp.bbox_hi[2]));
    if (detailed) {
        std::fprintf(stderr, "  Rank 0: owned=%lld ghosts=%lld nodes=%lld faces=%lld peers=%d\n",
                     static_cast<long long>(mp.n_own), static_cast<long long>(mp.n_cells - mp.n_own),
                     static_cast<long long>(mp.n_nodes), static_cast<long long>(mp.n_faces),
                     mp.n_neighbors());
        std::fputs("  Topology and halo maps ready; geometry and face grouping are pending.\n", stderr);
    }
    std::fprintf(stderr, "KRONA mesh_partitioner: migration completed in %.3f s.\n", seconds);
}

/// @brief Prints global measures and coordinate bounds.
void print_geometry(const mesh::host::MeshPart& mp, const mesh::host::GeometryStats& geometry_stats,
                    bool detailed, double seconds) {
    std::fprintf(stderr, "  Geometry: total_volume=%.12g cell_volume=[%.6g, %.6g]\n",
                 geometry_stats.total_volume, geometry_stats.min_cell_volume, geometry_stats.max_cell_volume);
    std::fprintf(stderr, "  Face flux area: min=%.6g max=%.6g\n", geometry_stats.min_face_area,
                 geometry_stats.max_face_area);
    if (detailed) {
        std::fprintf(stderr, "  Bounds: x=[%.12g, %.12g] y=[%.12g, %.12g] z=[%.12g, %.12g]\n",
                     geometry_stats.bbox_lo[0], geometry_stats.bbox_hi[0], geometry_stats.bbox_lo[1],
                     geometry_stats.bbox_hi[1], geometry_stats.bbox_lo[2], geometry_stats.bbox_hi[2]);
        std::fprintf(stderr, "  Rank 0: metric_cells=%lld metric_faces=%lld storage=%zu-bit\n",
                     static_cast<long long>(mp.n_cells), static_cast<long long>(mp.n_faces),
                     sizeof(krona::GeometryReal) * 8);
        if (mp.n_faces_g > 0)
            std::fputs("  Positive metrics and owner/neighbour face-side checks passed.\n", stderr);
    }
    std::fprintf(stderr, "KRONA mesh_partitioner: geometry completed in %.3f s.\n", seconds);
}

/// @brief Prints grouped face ranges and local patch sizes.
void print_reorder(const mesh::host::MeshPart& mp, bool detailed, double seconds) {
    std::fprintf(stderr, "  Rank 0 face regions: owned-owned=%lld owned-ghost=%lld boundary=%lld\n",
                 static_cast<long long>(mp.n_internal_faces),
                 static_cast<long long>(mp.n_inner_faces - mp.n_internal_faces),
                 static_cast<long long>(mp.n_faces - mp.n_inner_faces));
    if (detailed) {
        std::fprintf(stderr, "  Rank 0 cells: owned=%lld ghosts=%lld\n", static_cast<long long>(mp.n_own),
                     static_cast<long long>(mp.n_cells - mp.n_own));
        std::fprintf(stderr, "  Rank 0 boundary faces: labelled=%zu unlabelled=%zu\n", mp.patch_faces.size(),
                     static_cast<std::size_t>(mp.n_faces - mp.n_inner_faces) - mp.patch_faces.size());
        for (std::size_t p = 0; p < mp.patches.size(); ++p)
            std::fprintf(stderr, "  Patch %zu: name=%s local_faces=%zu\n", p, mp.patches[p].name.c_str(),
                         mp.patch_face_offsets[p + 1] - mp.patch_face_offsets[p]);
        std::fputs("  Three face regions, halo indices and patch-face tables ready.\n", stderr);
    }
    std::fprintf(stderr, "KRONA mesh_partitioner: reordering completed in %.3f s.\n", seconds);
}

/// @brief Prints validated counts, owned-cell loads and labelled boundary areas.
void print_validation(const mesh::host::MeshPart& mp, const mesh::host::MeshStatistics& mesh_stats,
                      mpi::CommView comm, bool detailed, double seconds) {
    const double average = static_cast<double>(mesh_stats.owned_cells) / static_cast<double>(comm.size());
    const double imbalance =
        average > 0 ? (static_cast<double>(mesh_stats.max_owned) / average - 1) * 100 : 0;
    std::fprintf(stderr, "  Verified: cells=%llu faces=%llu boundary=%llu labelled=%llu\n",
                 static_cast<unsigned long long>(mesh_stats.owned_cells),
                 static_cast<unsigned long long>(mp.n_faces_g),
                 static_cast<unsigned long long>(mesh_stats.boundary_faces),
                 static_cast<unsigned long long>(mesh_stats.labelled_faces));
    std::fprintf(stderr, "  Owned cells/rank: min=%llu avg=%.1f max=%llu imbalance=%.2f%%\n",
                 static_cast<unsigned long long>(mesh_stats.min_owned), average,
                 static_cast<unsigned long long>(mesh_stats.max_owned), imbalance);
    if (detailed) {
        std::fprintf(stderr, "  Faces: owned-owned=%llu partition-cut=%llu unlabelled-boundary=%llu\n",
                     static_cast<unsigned long long>(mesh_stats.interior_faces),
                     static_cast<unsigned long long>(mesh_stats.cut_faces),
                     static_cast<unsigned long long>(mesh_stats.boundary_faces - mesh_stats.labelled_faces));
        std::fprintf(stderr, "  Ghost copies: total=%llu min/rank=%llu max/rank=%llu\n",
                     static_cast<unsigned long long>(mesh_stats.ghost_copies),
                     static_cast<unsigned long long>(mesh_stats.min_ghosts),
                     static_cast<unsigned long long>(mesh_stats.max_ghosts));
        std::fprintf(stderr, "  Total owned-cell volume: %.12g\n", mesh_stats.total_volume);
        for (std::size_t p = 0; p < mp.patches.size(); ++p)
            std::fprintf(stderr, "  Patch %zu: name=%s source=%s faces=%llu flux_area=%.8g\n", p,
                         mp.patches[p].name.c_str(), mp.patches[p].source_type.c_str(),
                         static_cast<unsigned long long>(mesh_stats.patch_face_counts[p]),
                         mesh_stats.patch_flux_areas[p]);
    }
    std::fprintf(stderr, "KRONA mesh_partitioner: validation completed in %.3f s.\n", seconds);
}

/// @brief Prints the written solver-mesh format and global sizes.
void print_export(const Options& options, const mesh::host::MeshPart& mp, mpi::CommView comm, bool detailed,
                  double seconds) {
    std::fprintf(stderr, "  Solver mesh: %s\n", options.output.c_str());
    std::fprintf(stderr, "  Snapshot: cells=%llu nodes=%llu faces=%llu partitions=%d\n",
                 static_cast<unsigned long long>(mp.n_cells_g), static_cast<unsigned long long>(mp.n_nodes_g),
                 static_cast<unsigned long long>(mp.n_faces_g), comm.size());
    if (detailed)
        std::fprintf(stderr, "  Format: KRONA HDF5 v2, indices=%zu/%zu bits, geometry=%zu bits\n",
                     sizeof(krona::LocalIndex) * 8, sizeof(krona::GlobalIndex) * 8,
                     sizeof(krona::GeometryReal) * 8);
    std::fprintf(stderr, "KRONA mesh_partitioner: solver mesh written in %.3f s (rank 0).\n", seconds);
}

/// @brief Imports, partitions, migrates and exports one distributed solver mesh.
int run_partitioner(const Options& options, mpi::CommView comm) {
    const auto global_started = Clock::now();
    const bool report = comm.rank() == 0 && options.verbosity != Verbosity::quiet;
    const bool detailed = options.verbosity == Verbosity::detailed;
    if (report)
        std::fprintf(stderr, "KRONA mesh_partitioner: input=%s output=%s ranks=%d reorder=%s\n",
                     options.input.c_str(), options.output.c_str(), comm.size(),
                     mesh::host::reorder_method_to_string(options.reorder));

    // Import also converts planar 2D input into one 3D extrusion layer.
    auto started = Clock::now();
    auto imported = io::cgns::read_parallel(options.input, comm);
    auto& source = imported.mesh;
    if (report)
        print_import(imported, comm, detailed, seconds_since(started));

    // Topology uses global connectivity; graph rows follow current source ownership.
    started = Clock::now();
    auto topology = mesh::host::build_faces(source, comm);
    if (report)
        print_topology(source, topology, detailed, seconds_since(started));

    started = Clock::now();
    const mesh::partition::DKaMinParOptions partition_options{};
    auto partition = mesh::partition::partition_graph(topology.graph.view(), source.distribution.cells, comm,
                                                      partition_options);
    if (report)
        print_partition(source, partition, partition_options, comm, detailed, seconds_since(started));

    // Release the dual graph before consuming source arrays and creating the halo.
    started = Clock::now();
    topology.graph = {};
    auto part = mesh::host::migrate_local_mesh(std::move(source), std::move(topology.faces),
                                                         std::span<const int>{partition.vertex_target_rank}, comm);
    std::vector<int>{}.swap(partition.vertex_target_rank);
    if (report)
        print_migration(part, detailed, seconds_since(started));

    // Metrics precede centroid-based ordering; ordering also groups boundary faces.
    started = Clock::now();
    const auto geometry = mesh::host::compute_mesh_geometry(part, comm);
    if (report)
        print_geometry(part, geometry, detailed, seconds_since(started));

    started = Clock::now();
    mesh::host::reorder_local_mesh(part, options.reorder);
    if (report)
        print_reorder(part, detailed, seconds_since(started));

    started = Clock::now();
    mesh::host::validate_meshpart(part);
    const auto statistics = mesh::host::collect_mesh_statistics(part, comm);
    if (report)
        print_validation(part, statistics, comm, detailed, seconds_since(started));

    // Only the small BC template is rank-local; HDF5 and VTU calls are collective.
    if (!options.boundary_output.empty() && comm.rank() == 0)
        io::boundary_config::generate_bc_template_config(part, options.boundary_output,
                                                         statistics.patch_face_counts);
    if (!options.patches_template_directory.empty())
        io::boundary_config::generate_patch_profile_templates(part, options.patches_template_directory, comm);
                                                         
    started = Clock::now();
    io::solver_mesh::export_mesh_hdf5(part, options.output, comm, partition.part2rank);
    if (report)
        print_export(options, part, comm, detailed, seconds_since(started));
    if (!options.vtu_directory.empty()) {
        io::vtk::write_vtu(part, options.vtu_directory, "part", comm);
        if (report)
            std::fprintf(stderr, "  Visualization: %s/part.pvtu and part_bnd.pvtu\n",
                         options.vtu_directory.c_str());
    }
    if (report)
        std::fprintf(stderr, "KRONA mesh_partitioner: completed in %.3f s (rank 0).\n",
                     seconds_since(global_started));
    return EXIT_SUCCESS;
}

} // namespace

/// @brief Keeps CLI handling and all collective stages inside the MPI error guard.
int run(int argc, char* const* argv, mpi::CommView comm) noexcept {
    return mpi::guarded([&] {
        mpi::require(!comm.is_null(), "Mesh partitioner app requires a live communicator");
        ParseResult parsed;
        if (comm.rank() == 0) parsed = parse_arguments(argc, argv);

        // All ranks take the same help, error or execution branch.
        int action = static_cast<int>(parsed.action);
        mpi::broadcast(comm, action);
        if (action == static_cast<int>(CliAction::help)) {
            if (comm.rank() == 0) usage(stdout);
            return EXIT_SUCCESS;
        }
        if (action == static_cast<int>(CliAction::error)) {
            if (comm.rank() == 0) print_error(parsed, stderr);
            return EXIT_FAILURE;
        }
        broadcast_options(parsed.options, comm);
        return run_partitioner(parsed.options, comm);
    });
}

} // namespace krona::apps::mesh_partitioner
