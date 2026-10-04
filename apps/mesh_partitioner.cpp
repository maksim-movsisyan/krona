// mpi runtime
#include "krona/mesh/localmesh.hpp"
#include "krona/mesh/raw_mesh.hpp"
#include "krona/mpi/collective_status.hpp"
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>

// mesh partition utils
#include <krona/mesh/extrusion.hpp>
#include <krona/mesh/faces.hpp>
#include <krona/mesh_partition/partition.hpp>
#include <krona/mesh/migrate.hpp>
#include <krona/mesh/geometry.hpp>
#include <krona/mesh/reorder.hpp>
#include <krona/mesh/validate.hpp>
#include <krona/mesh/bc_config.hpp>

// input/output
#include <krona/io/solver_mesh/hdf5_writer.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/io/cgns/reader.hpp>

#include <utility>
#include <vector>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <string>
#include <string_view>
#include <chrono>

namespace {

enum class Verbosity : int { QUIET, SUMMARY, DETAILED };
enum class CliAction : int { RUN, HELP, ERROR };

// parsing options
struct Options {
    std::string input;
    std::string output;
    std::string boundary_output;
    std::string vtu_directory;
    krona::mesh::ReorderMethod reorder = krona::mesh::ReorderMethod::NONE;
    Verbosity verbosity = Verbosity::QUIET;
};

// parsing result
struct ParseResult {
    Options options;
    CliAction action = CliAction::RUN;
    std::string_view error;
    std::string_view argument;
};

// Print on rank 0; no arguments and --help are successful help requests.
void usage(std::FILE* stream) noexcept
{
    std::fputs(
        "KRONA distributed mesh partitioner\n"
        "\n"
        "Usage:\n"
        "  mpirun -np N krona_mesh_partition <in.cgns> <out.h5> [options]\n"
        "  krona_mesh_partition --help\n"
        "\n"
        "Options:\n"
        "  --bc <bc.txt>       Write a boundary-condition template (optional).\n"
        "  --vtu <dir>         Write VTU visualization files (optional).\n"
        "  --reorder <TYPE>    Local reordering: NONE, SFC, RCM (default: NONE).\n"
        "  -v, --verbose      Summary diagnostics.\n"
        "  -vv                Detailed diagnostics.\n"
        "  -h, --help         Show this help.\n"
        "  --                 End options; remaining arguments are paths.\n"
        "\n"
        "Without --bc, no boundary-condition template is requested.\n"
        "Named options also accept --option=value.\n"
        "\n"
        "Examples:\n"
        "  mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5\n"
        "  mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5 --bc mesh.toml\n"
        "  mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5 --vtu vtk --reorder RCM -v\n"
        "\n",
        stream);
}

// Parse only on rank 0; expected CLI errors are returned, not thrown.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv)
{
    // initialize reuslt
    ParseResult result;

    // aux helper (is token is optional)
    auto is_optional = [](std::string_view token) {
        return token.size() > 1 && token.front() == '-';
    };

    // aux helper for parsion errors
    auto fail = [&](std::string_view error, std::string_view argument = {}) {
        result.action = CliAction::ERROR;
        result.error = error;
        result.argument = argument;
    };

    // if mesh partition called with no arguments => show usage
    if (argc <= 1) { result.action = CliAction::HELP; return result; }

    // 
    int positional_count = 0;
    bool positional_only = false;
    bool reorder_seen = false;
    // loop over all additional arguments
    for (int i = 1; i < argc; ++i) {
        // get argument token
        const std::string_view token{argv[i]};

        // token '--' makrs end of default arugments
        if (!positional_only && token == "--") {
            positional_only = true;
            continue;
        }

        // -h token => show usage only
        if (!positional_only && (token == "-h" || token == "--help")) {
            result.action = CliAction::HELP;
            return result;
        }

        // -v => set vebosity level to SUMMARY
        if (!positional_only && (token == "-v" || token == "--verbose")) {
            result.options.verbosity = Verbosity::SUMMARY;
            continue;
        }

        // -vv => set verbosity level to DETAILED
        if (!positional_only && token == "-vv") {
            result.options.verbosity = Verbosity::DETAILED;
            continue;
        }

        // check optional tokens
        if (!positional_only && is_optional(token)) {
            // check if optional token has form: -token=key
            const auto equal = token.find('=');
            const auto name = token.substr(0, equal);

            // check if token name if avaliable
            if (name != "--bc" && name != "--vtu" && name != "--reorder") {
                fail("Unknown option", token);
                return result;
            }

            // check for optional argument duplication
            if ((name == "--bc" && !result.options.boundary_output.empty()) ||
                (name == "--vtu" && !result.options.vtu_directory.empty()) ||
                (name == "--reorder" && reorder_seen)) {
                fail("Duplicate option", name);
                return result;
            }

            // parse optional value
            std::string_view value;
            if (equal != std::string_view::npos) {
                value = token.substr(equal + 1);
            } else {
                if (i + 1 >= argc || is_optional(argv[i + 1])) {
                    fail("Missing value for option", name);
                    return result;
                }
                value = argv[++i];
            }

            // corectness check
            if (value.empty()) {
                fail("Empty value for option", name);
                return result;
            }

            if (name == "--bc") {
                result.options.boundary_output = value;
            } else if (name == "--vtu") {
                result.options.vtu_directory = value;
            } else {
                reorder_seen = true;
                if (value == "NONE") result.options.reorder = krona::mesh::ReorderMethod::NONE;
                else if (value == "SFC") result.options.reorder = krona::mesh::ReorderMethod::HILBERT_SFC;
                else if (value == "RCM") result.options.reorder = krona::mesh::ReorderMethod::RCM;
                else {
                    fail("Invalid reorder method; expected NONE, SFC or RCM", value);
                    return result;
                }
            }
            continue;
        }

        if (token.empty()) {
            fail("Input and output paths must not be empty");
            return result;
        }
        if (positional_count == 0) result.options.input = token;
        else if (positional_count == 1) result.options.output = token;
        else {
            fail("Unexpected positional argument", token);
            return result;
        }
        ++positional_count;
    }
    if (positional_count != 2) {
        fail("Expected an input CGNS path and an output HDF5 path");
    }
    return result;
}

// Replicate root's settings; absence of optional paths is encoded by length zero.
void broadcast_options(Options& options, krona::mpi::CommView comm)
{
    // echange sizes
    std::array<int, 6> header{};
    if (comm.rank() == 0) {
        header[0] = static_cast<int>(options.input.size());
        header[1] = static_cast<int>(options.output.size());
        header[2] = !options.boundary_output.empty() ? static_cast<int>(options.boundary_output.size()) : 0;
        header[3] = !options.vtu_directory.empty() ? static_cast<int>(options.vtu_directory.size()) : 0;
        header[4] = static_cast<int>(options.reorder);
        header[5] = static_cast<int>(options.verbosity);
    }
    krona::mpi::broadcast(comm, std::span{header});

    // allocate memory on child ranks
    if (comm.rank() != 0) {
        options.input.resize(static_cast<std::size_t>(header[0]));
        options.output.resize(static_cast<std::size_t>(header[1]));
        
        if (header[2] > 0) {
            options.boundary_output.resize(static_cast<std::size_t>(header[2]));
        }

        if (header[3] > 0) {
            options.vtu_directory.resize(static_cast<std::size_t>(header[3]));
        }
    }

    // exchange data
    krona::mpi::broadcast(comm, std::span{options.input});
    krona::mpi::broadcast(comm, std::span{options.output});
    if (header[2] > 0) {
        krona::mpi::broadcast(comm, std::span{options.boundary_output});
    }
    if (header[3] > 0) {
        krona::mpi::broadcast(comm, std::span{options.vtu_directory});
    }
    options.reorder = static_cast<krona::mesh::ReorderMethod>(header[4]);
    options.verbosity = static_cast<Verbosity>(header[5]);
}

// Import the mesh and extrude 2D input to one 3D layer.
[[nodiscard]] int run_partitioner(const Options& options, krona::mpi::CommView comm)
{   
    const auto global_start = std::chrono::steady_clock::now();
    // set verbosity aux flags
    const bool report = comm.rank() == 0 && options.verbosity != Verbosity::QUIET;
    const bool detailed = report && options.verbosity > Verbosity::SUMMARY;

    // output general mesh partitioner information
    if (report) {
        std::fprintf(stderr,
            "KRONA mesh_partition: input=%s output=%s ranks=%d reorder=%s verbosity=%d\n"
            "  bc=%s\n  vtu=%s\n",
            options.input.c_str(), options.output.c_str(), comm.size(), krona::mesh::reorder_method_to_string(options.reorder),
            static_cast<int>(options.verbosity),
            !options.boundary_output.empty() ? options.boundary_output.c_str() : "<disabled>",
            !options.vtu_directory.empty() ? options.vtu_directory.c_str() : "<disabled>");
    }

   
    


    // Parallel CGNS import with automatic conversion of planar 2D meshes to 3D.
    //
    // Collective contract:
    //  - Every rank in comm participates, including ranks with empty owned slices.
    //  - All ranks use the same path and ReaderOptions, and call on the MPI main thread.
    //  - The default reader uses collective I/O; do not overlap other CGNS calls.
    //  - This call reads base 1 / zone 1 unless explicit ReaderOptions are supplied.
    //    Other bases or zones may exist; they are not imported by this call.
    //
    // Input contract:
    //  - The selected zone is a nonempty HDF5-backed Unstructured zone.
    //  - Supported linear cells are TRI/QUAD in 2D and TET/PYRA/PRISM/HEXA in 3D.
    //  - Uniform and MIXED sections are supported. An imported section cannot cross
    //    the cell/boundary ID boundary; MIXED record dimensions are checked on reading.
    //  - Cartesian coordinate arrays are required for the source physical dimension.
    //  - Element-based BCs use FaceCenter in 3D or EdgeCenter in 2D; legacy element
    //    lists/ranges are also supported. Unsupported BC representations are rejected
    //    by default, or recorded as ignored when explicitly requested in ReaderOptions.
    //  - Missing BC definitions are allowed; unassigned elements retain invalid_patch_id.
    //
    // Import algorithm:
    //  1. Read small replicated metadata and establish balanced block distributions
    //     for cells, nodes and dense boundary ordinals.
    //  2. Read each rank's coordinate and connectivity slices directly into owned
    //     storage. Boundary ownership is independent of cell/node ownership and uses
    //     one dense distribution across imported boundary sections, not one per section.
    //  3. Read MIXED offsets using their own stored integer type: I4 connectivity may
    //     have I8 offsets. Use offset endpoints to plan final connectivity storage.
    //  4. Apply BC ranges locally and route distributed BC lists to boundary owners.
    //     Global BC point lists are never replicated on every rank.
    //  5. Close the file. If the source is 2D, extrude the planar TRI/QUAD mesh into
    //     one PRISM/HEXA layer, convert BAR boundaries to QUAD faces, and add end caps.
    //     Preserve cell ownership; rebalance output nodes and boundary ordinals.
    //
    // After import:
    //  - imported.mesh is always 3D; imported.source retains source dimensions/names.
    //  - imported.extrusion contains displacement information only for a 2D source.
    //  - Each rank owns only its cell, coordinate and boundary slices. Connectivity
    //    references zero-based global node IDs, which may belong to other ranks.
    //  - Counts, block-distribution parameters and patch labels are small replicated
    //    metadata. Boundary element IDs may be sparse and are not dense ordinals.
    //  - This is initial block ownership, not a graph partition or a halo mesh.
    //    Migration/adaptation must establish the next ownership and ID mapping.

    if (report) std::fputs("KRONA mesh_partition: importing mesh...\n", stderr);

    const auto import_started = std::chrono::steady_clock::now();

    krona::io::cgns::ReadResult imported = krona::io::cgns::read_parallel(options.input, comm);
    krona::mesh::RawMesh& mesh = imported.mesh;

    if (report) {
        // Measure rank 0's elapsed time without adding a barrier or a timing reduction.
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - import_started).count();

        std::fprintf(stderr,
            "  Mesh: cells=%lld nodes=%lld boundary_elements=%lld patches=%zu ranks=%d\n",
            static_cast<long long>(mesh.distribution.cells.global_size()),
            static_cast<long long>(mesh.distribution.nodes.global_size()),
            static_cast<long long>(mesh.distribution.boundary_ordinals.global_size()),
            mesh.patches.size(), comm.size());

        if (imported.extrusion) {
            const auto& extrusion = *imported.extrusion;
            std::fprintf(stderr,
                "  Extrusion: 2D -> 3D, one layer, thickness=%.6g normal=(%.6g, %.6g, %.6g)\n",
                static_cast<double>(extrusion.thickness), extrusion.normal[0],
                extrusion.normal[1], extrusion.normal[2]);
        }

        if (detailed) {
            std::fprintf(stderr,
                "  Source: CGNS=%.2f base=\"%s\" zone=\"%s\" cell_dim=%d physical_dim=%d\n",
                static_cast<double>(imported.source.version),
                imported.source.base_name.c_str(),
                imported.source.zone_name.c_str(),
                imported.source.cell_dimension,
                imported.source.physical_dimension);

            if (!mesh.patches.empty())
                std::fputs("  Patches:\n", stderr);

            for (std::size_t i = 0; i < mesh.patches.size(); ++i) {
                const auto& patch = mesh.patches[i];
                std::fprintf(stderr,
                    "    [%zu] name=\"%s\" family=\"%s\" source=\"%s\"\n",
                    i, patch.name.c_str(),
                    patch.family.empty() ? "-" : patch.family.c_str(),
                    patch.source_type.empty() ? "-" : patch.source_type.c_str());
            }

            for (const auto& name : imported.source.ignored_boundary_conditions)
                std::fprintf(stderr, "  Ignored BC: \"%s\"\n", name.c_str());
        }

        std::fprintf(stderr,
            "KRONA mesh_partition: import completed in %.3f s.\n", seconds);
    }





    // Distributed construction of unique faces and a symmetric cell dual graph.
    //
    // Collective contract and preconditions:
    //  - Every rank in comm participates with identical BuildFacesOptions.
    //  - mesh is a conforming 3D snapshot with owned cells in its current block order;
    //    ghosts are not included in mesh.cells. Node IDs are globally consistent.
    //  - Supported cells: TET/PYRA/PRISM/HEXA in canonical CGNS node order.
    //    Optional boundary elements are TRI/QUAD with corresponding patch IDs.
    //  - Coordinates and input-file metadata are not used. A generated or moved mesh
    //    works under the same ownership/connectivity contract as an imported mesh.
    //  - More than two cells per face, duplicate faces of one cell, duplicate boundary
    //    records, orphan boundary elements and boundary labels on interior faces abort.
    //    Nonconforming AMR interfaces need a different matching scheme and are not supported.
    //
    // Algorithm:
    //  1. Count local half-faces and boundary labels. One maximum Allreduce determines
    //     a common round count. Store compact source references instead of retaining
    //     a full message for every local face.
    //  2. Build orientation-independent keys from sorted global node IDs. Hash each
    //     key into a bucket selecting both a round and a rendezvous rank. Equal keys
    //     always meet in the same round, so unmatched records need not persist.
    //  3. Materialize the current round and exchange it using packed Alltoallv.
    //     Sort received records and match equal keys: two cells form an interior face;
    //     one cell forms an exterior face, with a boundary patch when available.
    //     Missing labels leave invalid_patch_id; a boundary-only group is an error.
    //  4. Send each unique face to the owner of its smaller cell ID, cell_a.
    //     For cross-rank interior faces, send cell_b -> cell_a to cell_b's owner.
    //     Add both directed adjacencies immediately to reserved local CSR rows;
    //     reverse-edge receive buffers are not accumulated between rounds.
    //  5. Sort and deduplicate each CSR row, compact adjacency in place, consolidate
    //     face chunks, and obtain all global totals with one final sum Allreduce.
    //
    // Communication and memory:
    //  - Each round uses three packed exchanges: half-faces, unique faces and reverse
    //    edges. Each exchange needs one counts Alltoall and one payload Alltoallv.
    //    The stage has two Allreduces total; there are no status/diagnostic collectives.
    //  - Default batch_bytes is 64 MiB: an average per-rank half-face payload target,
    //    not a strict receive-buffer or total-memory cap under hash imbalance.
    //  - CSR initially reserves at most one neighbour slot per local cell face;
    //    unused capacity may remain after compaction. Temporary buffers are released
    //    after their last use. Faces and graph remain distributed; mesh is read-only.
    //
    // After construction:
    //  - topology.faces stores one globally unique record per face on cell_a's owner.
    //    cell_b == invalid_cell_id identifies an exterior face, regardless of its patch.
    //  - Face keys are sorted matching keys, not oriented connectivity. Recover node
    //    order from cell_a and local_face_a using the canonical cell face tables.
    //  - topology.graph has one row per owned cell, offsets of type size_t, and sorted
    //    unique neighbouring global cell IDs. Boundary faces do not create graph arcs.
    //  - topology.stats contains replicated global counts; graph_arcs counts directed
    //    adjacencies, not unique faces or undirected graph edges.
    //  - Face-vector order is unspecified and is not a persistent global face numbering.
    //    Pure coordinate motion preserves this topology; connectivity changes require
    //    rebuilding after restoring the ownership contract and boundary connectivity.

    if (report) std::fputs("KRONA mesh_partition: building faces and dual graph...\n", stderr);

    const auto topology_started = std::chrono::steady_clock::now();

    krona::mesh::BuildFacesResult topology = krona::mesh::build_faces(mesh, comm);
    std::vector<krona::mesh::FaceRecord>& faces = topology.faces;
    krona::mesh::DualGraph& graph = topology.graph;

    if (report) {
        // Global statistics already exist; logging requires no additional MPI calls.
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - topology_started).count();
        const auto& stats = topology.stats;

        std::fprintf(stderr, "  Faces: total=%llu interior=%llu boundary=%llu\n",
            static_cast<unsigned long long>(stats.faces),
            static_cast<unsigned long long>(stats.interior_faces),
            static_cast<unsigned long long>(stats.boundary_faces));

        std::fprintf(stderr, "  Dual graph: vertices=%lld directed_arcs=%llu\n",
            static_cast<long long>(mesh.distribution.cells.global_size()),
            static_cast<unsigned long long>(stats.graph_arcs));

        if (detailed) {
            std::fprintf(stderr,
                "  Rank 0 owns: faces=%zu graph_rows=%zu graph_arcs=%zu\n",
                faces.size(), mesh.cells.size(), graph.adjacency.size());
            std::fputs(
                "  Face ownership: rank owning cell_a (smaller cell ID).\n", stderr);
            std::fputs(
                "  CSR neighbours: sorted, unique global cell IDs.\n", stderr);
        }

        std::fprintf(stderr,
            "KRONA mesh_partition: topology completed in %.3f s.\n", seconds);
    }
    




    // Distributed partitioning of the cell dual graph with dKaMinPar.
    // This stage computes migration destinations; it does not move mesh data.
    //
    // Collective contract:
    //  - Every rank in comm participates, including ranks with no owned cells.
    //  - Ownership metadata and DKaMinParOptions are identical on all ranks.
    //  - Call on the MPI main thread; do not overlap other dKaMinPar calls in a process.
    //  - Multiple worker threads need MPI_THREAD_FUNNELED or stronger support.
    //
    // Input and balancing contract:
    //  - graph is the symmetric CSR produced by build_faces(), with one row per
    //    currently owned cell and zero-based global neighbour IDs.
    //  - Cells still follow mesh.distribution.cells. No ghost rows are passed.
    //  - The explicit-endpoint overload also accepts unequal contiguous rank ranges;
    //    after migration/adaptation, renumber vertices and remap adjacency consistently.
    //  - The partitioner creates exactly comm.size() blocks, one per destination rank.
    //    All cells and graph edges have unit weight: balance means cell-count balance.
    //  - imbalance_tolerance must be finite and positive; integer block sizes can make
    //    the measured max/average imbalance exceed epsilon on small graphs.
    //  - dKaMinPar's configured local ID and weight widths must fit the graph, including
    //    the ghost vertices created internally during distributed import.
    //
    // Algorithm:
    //  1. Build a bijective block-to-rank placement. With shared-memory grouping,
    //     use MPI_Comm_split_type to find rank groups, broadcast each group's smallest
    //     parent rank, and Allgather these small leader IDs. Group destination ranks
    //     by leader ID, then by parent rank. No leader communicator is needed.
    //  2. Format distributed CSR for dKaMinPar. Borrow arrays whose integer types match
    //     the public API and convert the others into temporary storage. copy_graph()
    //     creates the library-owned representation; release conversion buffers before
    //     entering the multilevel solver. The original graph remains unchanged.
    //  3. Partition into P blocks with the default dKaMinPar context and requested
    //     imbalance tolerance. Obtain the global undirected edge cut from its return
    //     value; do not perform an additional halo exchange to recompute it.
    //  4. Map each local block ID through part2rank to obtain cell_target_rank.
    //     Count planned cells per destination and release solver/raw-partition storage.
    //  5. Sum the P destination counts with one Allreduce for application diagnostics.
    //     There are no wrapper status collectives; dKaMinPar has its own internal MPI.
    //     One rank or an empty global graph bypasses the solver and this reduction.
    //
    // Placement semantics:
    //  - Shared-memory grouping places consecutive block IDs on co-located ranks.
    //    It is a heuristic, not a topology-aware optimization of the block graph.
    //  - Local MPI order is not a physical CPU/core ID. This stage changes no affinity.
    //  - No guarantee is made that communicating blocks share a server, that network
    //    traffic disappears, or that the placement minimizes the inter-node cut.
    //  - global_edge_cut is unchanged by the bijective block-to-rank permutation.
    //    It counts cut graph edges, not necessarily individual physical mesh faces.
    //    The inter-node cut is not measured and is not represented by a fictitious zero.
    //
    // After partitioning:
    //  - pr.cell_target_rank has one valid destination rank per input CSR row.
    //  - pr.part2rank maps block IDs to ranks; pr.rank2part is its inverse.
    //    These tables describe this partition only, not persistent cell/face numbering.
    //  - pr.cells_per_rank contains replicated global planned ownership counts.
    //    Actual ownership, cell IDs, mesh arrays and graph arrays are still unchanged.
    //  - Coordinates and CGNS metadata are irrelevant to this stage. Coordinate motion
    //    alone does not require repartitioning. Topology changes may require rebuilding
    //    the graph; a later repartition is full partitioning, not incremental migration
    //    minimization or a reuse of the previous partition.

    const krona::mesh_partition::DKaMinParOptions partition_options{
        .imbalance_tolerance = 0.03,
        .threads_per_rank = 1,
        .seed = 0,
        .quiet = true, // Application diagnostics below control verbosity independently.
        .group_by_shared_memory = true
    };

    if (report) std::fputs("KRONA mesh_partition: partitioning dual graph...\n", stderr);
    const auto partition_started = std::chrono::steady_clock::now();

    krona::mesh_partition::PartitionResult pr = krona::mesh_partition::partition_cells(
        graph, mesh.distribution.cells, comm, partition_options);

    if (report) {
        // All counts are already global; logging adds no MPI calls or barriers.
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - partition_started).count();
        const auto [minimum, maximum] =
            std::minmax_element(pr.cells_per_rank.begin(), pr.cells_per_rank.end());
        const double average =
            static_cast<double>(mesh.distribution.cells.global_size()) / comm.size();
        const double imbalance = average > 0
            ? 100.0 * (static_cast<double>(*maximum) / average - 1.0) : 0.0;

        std::fprintf(stderr, "  Partition: blocks=%d edge_cut=%llu\n", comm.size(),
            static_cast<unsigned long long>(pr.global_edge_cut));
        std::fprintf(stderr,
            "  Planned cells/rank: min=%llu avg=%.1f max=%llu imbalance=%.2f%%\n",
            static_cast<unsigned long long>(*minimum), average,
            static_cast<unsigned long long>(*maximum), imbalance);
        if (pr.shared_memory_groups > 0)
            std::fprintf(stderr, "  Placement: shared-memory grouping, groups=%d\n",
                        pr.shared_memory_groups);
        else
            std::fputs("  Placement: MPI rank order.\n", stderr);

        if (detailed) {
            // Bound output on large communicators; do not print one line per cell.
            const auto shown = std::min<std::size_t>(pr.cells_per_rank.size(), 16);
            std::fprintf(stderr, "  Settings: epsilon=%.4f threads/rank=%d seed=%d\n",
                partition_options.imbalance_tolerance, partition_options.threads_per_rank,
                partition_options.seed);
            std::fprintf(stderr, "  Rank placement and planned load (first %zu ranks):\n", shown);
            for (std::size_t rank = 0; rank < shown; ++rank)
                std::fprintf(stderr, "    rank=%zu block=%u cells=%llu\n", rank,
                    static_cast<unsigned int>(pr.rank2part[rank]),
                    static_cast<unsigned long long>(pr.cells_per_rank[rank]));
            if (shown < pr.cells_per_rank.size())
                std::fprintf(stderr, "    ... %zu further ranks omitted.\n",
                            pr.cells_per_rank.size() - shown);
            std::fputs("  Destinations computed; mesh migration has not started.\n", stderr);
        }
        // This is rank 0's elapsed time, not a reduced maximum across ranks.
        std::fprintf(stderr, "KRONA mesh_partition: partitioning completed in %.3f s.\n",
                    seconds);
    }





    // Distributed migration to the target ranks, one-layer face-neighbour halo,
    // local zero-based connectivity and field-exchange maps for the solver.
    // Every rank in comm participates, including ranks with no source/target cells.
    // Run on the MPI main thread with matching source ownership metadata on all ranks.
    //
    // Input contract:
    //  - mesh is the 3D import snapshot, with block-owned cells and coordinates.
    //  - faces contains one globally unique record per physical face, stored on
    //    the original owner of cell_a. Local facet indices follow CGNS node ordering.
    //  - pr.cell_target_rank has one destination rank per current local source cell.
    //  - Only linear TET/PYRA/PRISM/HEXA volume cells are supported. The mesh is
    //    conforming and manifold; vertex-sharing neighbours do not enter this halo.
    //  - The general MigrationSource overload also accepts unequal contiguous source
    //    ranges. It requires unique coordinate ownership and dense global IDs;
    //    adapted inputs must be renumbered and all references remapped consistently.
    //
    // Algorithm:
    //  1. Deduplicate remote cell_b IDs and query their current source owners for
    //     destination ranks. Local targets are read directly from the supplied span.
    //  2. Redistribute volume cells with small (GID,type) headers and exact-length
    //     node connectivity. Header arities determine connectivity receive counts.
    //     There is no fixed eight-node padding for tetrahedra, pyramids or prisms.
    //  3. Redistribute compact faces. A boundary or same-destination interior face
    //     is sent once; a cut face is sent to both adjacent destination ranks.
    //     The second copy uses cell_b as owner and its own local facet ordering.
    //  4. Sort/unique required ghosts by (new owner rank, cell GID). Request their
    //     complete topology from the new owners. The same requests establish halo
    //     send_owned_local and receive order; no additional handshake is performed.
    //     Known inverse layouts deliver replies without another count exchange.
    //  5. Resolve face references to local cell indices, then release cell-GID
    //     lookups. Collect unique nodes used by owned cells first and ghost-only
    //     nodes second; sort each group by GID, or scan compact dense ranges.
    //  6. Assemble cell CSR with owned cells in [0,n_own) and ghosts after them.
    //     Convert node GIDs to LocalIndex and release global connectivity/lookups.
    //     Reconstruct face nodes from oriented owner facets; no geometry is computed.
    //  7. Copy locally held coordinates and request only missing triples from the
    //     original coordinate owners. A one-rank full-node import moves coordinates.
    //  8. Build patch-face CSR and reduce actual face/boundary/cut/halo totals.
    //     Unique face counts come from source FaceRecords, not partition graph cuts:
    //     one graph edge can represent several physical faces. Bounds include all
    //     nodes referenced by owned cells; a completely empty mesh has zero bounds.
    //
    // Memory and communication:
    //  - This call consumes mesh cell/coordinate/boundary payloads and faces.
    //    The target-rank span is borrowed and must remain valid until return.
    //  - Reusable storage is moved; dead buffers are released at their last use.
    //    The source dual graph is released below before migration starts.
    //  - There is no gather to rank 0 and no collective status preflight.
    //    MPI 3.1 int counts/displacements limit each packed exchange; migration
    //    is not batched and offers no fixed bound on peak process memory.
    //  - One-rank transfers bypass MPI. Multi-rank ranks execute matching calls
    //    even if their local buffers are empty. Statistics need one SUM reduction
    //    and one MIN reduction for nonempty global meshes, with no logging barriers.
    //
    // Output and subsequent stages:
    //  - mp contains local cells/nodes/faces, patch labels and halo exchange maps.
    //    Nodes used by owned cells are not exclusively owned globally; ranks can
    //    store the same node. Ghosts are exactly one layer of face neighbours.
    //  - Each face owner is owned locally. An interior neighbour may be owned or
    //    ghost; a boundary neighbour is invalid_local_index. Cut faces occur on both
    //    adjacent ranks. Facet orientation assumes positively oriented input cells.
    //  - Geometry arrays remain empty. RCM/SFC and face grouping are separate stages.
    //    mp.faces_grouped is false; n_internal_faces/n_inner_faces are not valid range
    //    boundaries yet. The grouping stage must form owned-owned, owned-ghost and
    //    boundary sections, update their boundaries and set faces_grouped true.
    //  - Reordering must remap halo indices and patch-face indices with connectivity;
    //    topology changes require rebuilding them. Coordinate motion alone does not.

    if (report) std::fputs("KRONA mesh_partition: migrating mesh and building halo...\n", stderr);
    const auto migration_started = std::chrono::steady_clock::now();

    // The old CSR describes source ownership and is no longer needed here.
    graph = {};
    krona::mesh::MeshPart mp = krona::mesh::migrate_local_mesh(
        std::move(mesh), std::move(faces), std::span<const int>{pr.cell_target_rank}, comm);
    std::vector<int>{}.swap(pr.cell_target_rank);

    if (report) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - migration_started).count();
        const double halo_percent = mp.n_cells_g > 0
            ? 100.0 * static_cast<double>(mp.n_ghost_copies_g) / static_cast<double>(mp.n_cells_g) : 0.0;
        std::fprintf(stderr,
            "  Migrated: cells=%llu source_nodes=%llu unique_faces=%llu\n",
            static_cast<unsigned long long>(mp.n_cells_g),
            static_cast<unsigned long long>(mp.n_nodes_g),
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
            // These are rank 0's local counts, not a global distribution summary.
            std::fprintf(stderr, "  Rank 0: owned=%lld ghosts=%lld nodes=%lld faces=%lld peers=%d\n",
                static_cast<long long>(mp.n_own), static_cast<long long>(mp.n_cells - mp.n_own),
                static_cast<long long>(mp.n_nodes), static_cast<long long>(mp.n_faces), mp.n_neighbors());
            std::fputs("  Topology and halo maps ready; geometry and face grouping are pending.\n", stderr);
        }
        // Root-local elapsed time; diagnostics add no collective operations.
        std::fprintf(stderr, "KRONA mesh_partition: migration completed in %.3f s.\n", seconds);
    }
    




    // Distributed geometric integration and verification of outward face winding.
    // All ranks in comm participate, including empty ranks, on the MPI main thread.
    // The explicit communicator must match the ranks used to migrate mp.
    //
    // Input contract:
    //  - mp has valid local CSR connectivity and complete coordinates for owned and
    //    ghost cells. Supported linear volume cells are TET/PYRA/PRISM/HEXA; faces
    //    are TRI/QUAD with owner-relative outward loops from the CGNS face tables.
    //  - Every face owner belongs to [0,n_own); an interior neighbour may be owned
    //    or ghost. Boundary neighbours are invalid_local_index.
    //  - Cells have positive orientation and the usual valid convex CFD geometry.
    //    Strict centroid-side checks are not a general concave-cell validator or a
    //    proof that a cell/face has no self-intersections.
    //  - Coordinates may have changed since migration, but all shared/ghost copies
    //    must already be synchronized. Geometry computation does not exchange them.
    //
    // Algorithm and geometric convention:
    //  1. Resize existing metric arrays; no per-cell/per-face heap buffers are used.
    //     Compute every owned/ghost cell in double arithmetic from at most 8 vertices.
    //  2. Shift coordinates by the first vertex, then integrate around the local
    //     vertex mean. Outward facet triangles and this anchor define signed
    //     tetrahedra. Sum their volumes and first moments for the true volume centre
    //     of the represented piecewise-planar polyhedron, up to floating-point error.
    //  3. TRI faces are integrated directly. QUAD faces use four triangles joining
    //     consecutive vertices to their vertex mean. The identical fan defines the
    //     quad surface in both the cell and face integrations; a different diagonal
    //     or a bilinear warped-quad mapping would define different geometry.
    //  4. Compute face centres with actual subtriangle-area weights. Sum oriented
    //     triangle area vectors into S; store face_area = |S| and normal = S/|S|.
    //     Thus area*normal reconstructs the flux area vector. For a warped quad,
    //     |S| generally differs from the sum of its scalar triangle areas.
    //  5. Check positive finite volumes and flux areas, finite centres/normals and
    //     representable metric storage. No unit-dependent 1e-15 cutoff is used:
    //     a small valid cell is accepted while zero, inverted or nonfinite geometry
    //     aborts with its rank and local entity IDs (also cell GID/type for cells).
    //  6. Verify n dot (face_centre-owner_centre) > 0 on every face. Also verify
    //     n dot (neighbour_centre-face_centre) > 0 on interior faces. No face/cell
    //     is silently flipped. The normal points outward from its local owner;
    //     the two copies of a partition-cut face consequently have opposite normals.
    //     Winding checks use local offsets before conversion to world centroids,
    //     avoiding subtraction of rounded absolute centres on highly shifted meshes.
    //  7. Accumulate volume totals/extrema over owned cells only: ghosts do not
    //     duplicate the physical domain volume. Face extrema include local copies,
    //     whose duplication does not affect MIN/MAX. Bounds use nodes referenced by
    //     owned cells, excluding ghost-only nodes. Empty ranks use neutral extrema.
    //  8. Reduce packed extrema/bounds once with MIN and total owned volume once
    //     with SUM; refresh mp.bbox_lo/hi. No logging barriers or status preflights
    //     are added. A one-rank communicator skips both MPI reductions; a globally
    //     empty mesh reports zero total volume, extrema and bounds.
    //
    // Output and reuse:
    //  - mp contains volumes/centres for all local cells, including ghosts, and
    //    areas/centres/normals for every local face. Storage uses GeometryReal;
    //    double arithmetic cannot recover coordinate detail already lost on input.
    //  - Existing array capacity is reused. Cell-centroid arrays temporarily hold
    //    local offsets during orientation checks, then contain absolute centres.
    //    Connectivity, halo maps, patch indices and the three face regions are not
    //    reordered or changed. The function does not set faces_grouped.
    //  - Coordinate motion alone can reuse this call after coordinate synchronization.
    //    Topology adaptation also requires rebuilding topology/halo/connectivity first.
    //  - compute_mesh_geometry(mp) is the local overload: it computes the same metrics
    //    with no reductions, returns local statistics and leaves global bounds alone.
    //    Empty local minima/bounds in that overload retain infinity sentinels.
    //  - Geometry is ready after this stage. The separate RCM/SFC and face-grouping
    //    stage is still required before serializing the final grouped solver mesh.

    if (report) std::fputs("KRONA mesh_partition: computing geometry...\n", stderr);
    const auto geometry_started = std::chrono::steady_clock::now();

    const auto geometry_stats = krona::mesh::compute_mesh_geometry(mp, comm);

    if (report) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - geometry_started).count();
        std::fprintf(stderr, "  Geometry: total_volume=%.12g cell_volume=[%.6g, %.6g]\n",
            geometry_stats.total_volume, geometry_stats.min_cell_volume, geometry_stats.max_cell_volume);
        std::fprintf(stderr, "  Face flux area: min=%.6g max=%.6g\n",
            geometry_stats.min_face_area, geometry_stats.max_face_area);
        if (detailed) {
            std::fprintf(stderr, "  Bounds: x=[%.12g, %.12g] y=[%.12g, %.12g] z=[%.12g, %.12g]\n",
                geometry_stats.bbox_lo[0], geometry_stats.bbox_hi[0],
                geometry_stats.bbox_lo[1], geometry_stats.bbox_hi[1],
                geometry_stats.bbox_lo[2], geometry_stats.bbox_hi[2]);
            std::fprintf(stderr, "  Rank 0: metric_cells=%lld metric_faces=%lld storage=%zu-bit\n",
                static_cast<long long>(mp.n_cells), static_cast<long long>(mp.n_faces),
                sizeof(krona::GeometryReal) * 8);
            if (mp.n_faces_g > 0)
                std::fputs("  Positive metrics and owner/neighbour face-side checks passed.\n", stderr);
        }
        // Root-local elapsed time; diagnostics require no additional reductions.
        std::fprintf(stderr, "KRONA mesh_partition: geometry completed in %.3f s.\n", seconds);
    }

    // Local owned-cell ordering and three-region face grouping for solver loops.
    // Each rank processes its own MeshPart; this stage performs no MPI exchanges,
    // reductions or barriers. Run on the MPI main thread when MPI is active.
    // Existing nonblocking requests using mesh buffers must have completed first.
    //
    // Input contract:
    //  - mp contains valid local cell/face CSR, owner-relative face winding,
    //    halo maps and patch labels from migration. Node IDs remain local indices.
    //  - Owned cells occupy [0,n_own); ghosts occupy [n_own,n_cells). Face owners
    //    are owned; neighbours are owned, ghost, or invalid_local_index at a boundary.
    //  - Geometry arrays may be empty or populated with one value per local entity.
    //    HILBERT_SFC requires populated finite cell centroids when n_own > 1.
    //    RCM and NONE can be used before geometry computation.
    //  - Labelled faces are boundary faces with a valid patch index. Unlabelled
    //    boundaries retain invalid_patch_id and remain outside the patch-face CSR.
    //  - This stage updates every entity-indexed array currently defined in MeshPart.
    //    External solution fields would also need the corresponding permutations;
    //    in this preprocessing pipeline those fields have not been allocated yet.
    //
    // Owned-cell ordering:
    //  1. NONE preserves the current cell order, but still groups and sorts faces.
    //     Empty ranks and ranks with one owned cell follow the same grouping path.
    //  2. HILBERT_SFC quantizes local owned centroids to 21 bits per axis and sorts
    //     their 63-bit 3D Hilbert keys. All axes use the same largest bounding-box
    //     extent, preserving the physical aspect ratio rather than stretching each
    //     axis independently. Equal keys are ordered by cell GID; coincident
    //     centroids and zero extents are supported. Quantization can merge nearby
    //     points, and locality improvement depends on the mesh and solver kernel.
    //  3. RCM builds symmetric adjacency from owned-owned faces only. For each
    //     connected component, at most three BFS passes choose a pseudo-peripheral
    //     start. Cuthill-McKee traversal visits new neighbours in increasing degree
    //     order, breaks degree ties by current local index, and reverses that
    //     component's order. This is a bandwidth/profile heuristic, not an optimal
    //     ordering guarantee. Cross-rank matrix structure is not reordered here.
    //     Flat work queues are reused; temporary visitation marks are cleared only
    //     for touched components, avoiding full-mesh clearing per isolated component.
    //  4. Apply the owned permutation to types, GIDs, donors, connectivity and any
    //     populated metrics. Rebuild the owned CSR prefix with size_t offsets;
    //     ghost connectivity, offsets, metadata and metrics keep their old positions.
    //  5. Remap owned references in face_owner/face_neigh and send_owned_local.
    //     Preserve peer groups and the order of send entries inside every group:
    //     independently sorting those entries would break the remote receive order.
    //     recv_ghost_local and all halo offsets remain valid because ghosts keep
    //     their original local indices. Node numbering and coordinates are unchanged.
    //
    // Face grouping and orientation:
    //  6. Separate faces in a linear pass, then sort each region independently:
    //       [0,n_internal_faces): owned-owned, sorted by (owner,neighbour);
    //       [n_internal_faces,n_inner_faces): owned-ghost, sorted by (owner,neighbour);
    //       [n_inner_faces,n_faces): boundary, grouped by patch and sorted by owner.
    //     Unlabelled boundary faces follow all labelled patches. Previous face
    //     indices break otherwise equal keys, making repeated NONE grouping stable.
    //  7. Move owner/neighbour, type, patch, local facet index, node loops and all
    //     populated face metrics through the same face permutation. Each face keeps
    //     its physical owner and owner-relative winding; normals need no sign change
    //     when that owner's numeric local index becomes larger than its neighbour's.
    //  8. Rebuild patch_face_offsets/patch_faces from the patch-sorted boundary region.
    //     Empty patches have empty CSR rows. Set the two region boundaries and mark
    //     faces_grouped true only after all face and patch arrays are ready.
    //
    // Memory and extension:
    //  - Scalar SoA fields are permuted in place by cycles with one reused byte per
    //    row, rather than retaining replacement copies of all metric arrays.
    //  - Variable-length CSR rows use temporary offsets and connectivity for the
    //    reordered prefix. Cell-order planning storage is released before face
    //    sorting starts; there is no fixed peak-memory bound or per-entity allocation.
    //  - The overload taking span<const LocalIndex> accepts a custom owned new-to-old
    //    permutation. It checks the bijection and uses the same topology/halo/metric
    //    update path; the span must remain valid and must not alias mesh storage.
    //  - Local indices and face-vector positions change. Any external lookup/cache
    //    keyed by those positions must be remapped or rebuilt. Global IDs, global
    //    mesh statistics, bounds, patch metadata and ghost donor ordering persist.
    //
    // Solver use of the resulting regions:
    //  - Start halo exchange, evaluate owned-owned faces, complete the exchange,
    //    then evaluate owned-ghost faces. Boundary faces form a separate BC loop.
    //  - This layout enables overlap; the solver must implement communication,
    //    MPI progress and synchronization. Local SIMD loops still need safe handling
    //    of scatter updates to cells shared by several faces.
    //    Owned-owned describes face adjacency, not an entire reconstruction stencil:
    //    higher-order/viscous kernels must have their required gradient/halo inputs ready.
    //  - Moving coordinates alone preserve this layout. Topology adaptation requires
    //    updating connectivity/halo first and rerunning geometry/grouping as needed.

    if (report)
        std::fprintf(stderr, "KRONA mesh_partition: reordering mesh (%s)...\n",
            krona::mesh::reorder_method_to_string(static_cast<krona::mesh::ReorderMethod>(options.reorder)));
    const auto reorder_started = std::chrono::steady_clock::now();

    krona::mesh::reorder_local_mesh(mp, static_cast<krona::mesh::ReorderMethod>(options.reorder));

    if (report) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - reorder_started).count();
        // Counts describe rank 0's local storage; logging adds no global reductions.
        std::fprintf(stderr, "  Rank 0 face regions: owned-owned=%lld owned-ghost=%lld boundary=%lld\n",
            static_cast<long long>(mp.n_internal_faces),
            static_cast<long long>(mp.n_inner_faces - mp.n_internal_faces),
            static_cast<long long>(mp.n_faces - mp.n_inner_faces));
        if (detailed) {
            std::fprintf(stderr, "  Rank 0 cells: owned=%lld ghosts=%lld\n",
                static_cast<long long>(mp.n_own), static_cast<long long>(mp.n_cells - mp.n_own));
            std::fprintf(stderr, "  Rank 0 boundary faces: labelled=%zu unlabelled=%zu\n",
                mp.patch_faces.size(), static_cast<std::size_t>(mp.n_faces - mp.n_inner_faces) - mp.patch_faces.size());
            for (std::size_t p = 0; p < mp.patches.size(); ++p)
                std::fprintf(stderr, "  Patch %zu: name=%s local_faces=%zu\n", p,
                    mp.patches[p].name.c_str(), mp.patch_face_offsets[p + 1] - mp.patch_face_offsets[p]);
            std::fputs("  Three face regions, halo indices and patch-face tables ready.\n", stderr);
        }
        std::fprintf(stderr, "KRONA mesh_partition: reordering completed in %.3f s.\n", seconds);
    }





    // Final local structural validation, collective conservation/statistics,
    // and optional generation of the editable boundary-condition template.
    // This stage follows migration, geometry and face grouping/reordering.
    //
    // Validation contract and scope:
    //  - mp must have complete geometry and faces_grouped == true. All ranks use
    //    the same replicated patch metadata and the communicator used for migration.
    //  - validate_meshpart(mp) is local: it checks array sizes before indexing,
    //    monotone size_t CSR offsets, supported cell/face arities, local/global ID
    //    ranges, owned-node references, donor ranks and finite stored coordinates.
    //  - Checks positive finite volumes/areas, finite centroids and unit normals
    //    with a tolerance derived from GeometryReal precision, without an absolute
    //    1e-15 measure cutoff. It checks oriented face loops against the owner's
    //    canonical local facet and verifies all three face regions and sort keys.
    //  - Equal (owner,neighbour) keys are permitted: sorting is nondecreasing.
    //    Boundary patch IDs may be invalid_patch_id; these faces sort after labelled
    //    patches and are deliberately absent from patch_face_offsets/patch_faces.
    //  - Patch CSR must list every labelled boundary face exactly once in sorted
    //    order. Halo offsets/peers are checked, sends reference owned cells, and
    //    receives cover every ghost exactly once in its donor's peer group.
    //    The only mesh-sized scratch array is one byte per ghost for receive coverage.
    //  - This is a structural check, not a second mesh reconstruction. Geometric
    //    outward winding was checked by compute_mesh_geometry() using local offsets.
    //    Repeating the old dot product on rounded absolute centroids could reject
    //    valid thin cells on a strongly shifted mesh, so it is not repeated here.
    //  - Remote halo GID/order matching comes from migration's request/reply protocol.
    //    No neighbour Sendrecv handshake, global-ID uniqueness search or geometric
    //    self-intersection scan is added. Local malformed data aborts WORLD immediately;
    //    there is no collective status reduction before the actual statistics.
    //
    // Collective statistics and conservation:
    //  - Every rank calls collect_mesh_statistics(mp,comm), including empty ranks
    //    and quiet runs. It requires the local validation above to have succeeded.
    //  - One uint64 SUM reduces cell/ghost/face-region totals and all patch counts.
    //    One double SUM reduces owned-cell volume and all labelled-patch flux areas.
    //    One uint64 MIN reduces load extrema; complemented maxima share that call
    //    without converting integer counts to floating point. One-rank runs skip MPI.
    //  - Ghost volumes do not contribute to domain volume. Boundary faces occur
    //    once globally; cut faces occur twice. Conservation checks require:
    //      summed owned cells == mp.n_cells_g;
    //      summed ghost copies == mp.n_ghost_copies_g;
    //      owned-owned faces + cut copies/2 + boundary faces == mp.n_faces_g;
    //      boundary/labelled/cut totals agree with the migration snapshot metadata.
    //    Cut-copy count must be even. Unique-node count is retained source metadata;
    //    shared coordinate copies are not summed as globally unique nodes.
    //  - Patch areas sum face_area = |S|: for warped faces this is effective flux
    //    area, not a reconstructed scalar area of the triangulated surface. Units
    //    follow the input coordinates; diagnostics do not assume metres.
    //  - Statistics are returned on all ranks. Logging is only on rank 0 and adds
    //    no collectives or barriers. Quiet mode changes output, not validation.
    //
    // Optional BC template:
    //  - A nonempty options.boundary_config requests the TOML output. Rank 0 writes
    //    small replicated patch metadata with the already reduced patch counts;
    //    there is no extra reduction, gather or mesh payload written in this step.
    //  - The writer also accepts an empty path as a no-op. The requested file is
    //    replaced; its parent directory must exist. Open/write/close errors abort
    //    with a diagnostic, so a requested export cannot silently fail.
    //  - Entries retain patch_id, name, family, source_type and global_face_count.
    //    source_type is an import label, replacing the old CGNS-specific cgns_type
    //    field. TOML strings escape quotes, backslashes and ASCII controls; source
    //    metadata must already be valid UTF-8. Empty patches are included.
    //  - Solver types are suggestions from explicit import labels, not inferred
    //    from patch names. Supersonic inlet/outlet use their own template types;
    //    ambiguous labels become USER_DEFINED. Review the suggestions and uncomment
    //    the required parameter examples before using the configuration in a solver.
    //    Unlabelled boundary faces have no patch entry; their count appears in the header.
    //
    // After this stage:
    //  - mp remains unchanged; grouped topology, geometry and halo maps are ready
    //    for the separate distributed mesh writer. This stage does not serialize it.
    //  - mesh_stats contains replicated final diagnostics that later stages may reuse.

    if (report) std::fputs("KRONA mesh_partition: validating final mesh...\n", stderr);
    const auto validation_started = std::chrono::steady_clock::now();

    krona::mesh::validate_meshpart(mp);
    const auto mesh_stats = krona::mesh::collect_mesh_statistics(mp, comm);

    if (report) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - validation_started).count();
        const double average = static_cast<double>(mesh_stats.owned_cells) / static_cast<double>(comm.size());
        const double imbalance = average > 0 ? (static_cast<double>(mesh_stats.max_owned) / average - 1) * 100 : 0;
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
                    static_cast<unsigned long long>(mesh_stats.patch_face_counts[p]), mesh_stats.patch_flux_areas[p]);
        }
        std::fprintf(stderr, "KRONA mesh_partition: validation completed in %.3f s.\n", seconds);
    }

    // File creation is independent of verbosity; it is controlled only by the optional path.
    if (!options.boundary_output.empty() && comm.rank() == 0) {
        krona::mesh::generate_bc_template_config(mp, options.boundary_output, mesh_stats.patch_face_counts);
        if (report)
            std::fputs("KRONA mesh_partition: boundary-condition template written.\n", stderr);
    }




    
    // Distributed serialization of the final solver snapshot and optional VTK export.
    // This stage follows migration, geometry, reordering and final mesh validation.
    // All ranks in comm must call export_mesh_hdf5(), including ranks without cells.
    // All ranks must use identical paths, optional VTU enablement and partition mapping.
    // The communicator is passed explicitly; MPI_COMM_WORLD is not hardcoded into I/O.
    //
    // Solver-file contract:
    //  - Writes one shared HDF5 file with schema version 2. Requires parallel HDF5
    //    >= 1.10 and an existing parent directory. The requested file is replaced.
    //  - Mesh payloads remain distributed. One small Allgather collects each rank's
    //    payload lengths; prefixes are constructed in partition order using
    //    pr.part2rank. No rank receives another rank's mesh connectivity or metrics.
    //  - Each rank writes a non-overlapping hyperslab into every SoA dataset.
    //    Empty ranks select none but still participate in every collective call.
    //    Shared mapping, labels and global metadata are written once by rank 0,
    //    with the other ranks participating through empty selections.
    //  - All dataset transfers request H5FD_MPIO_COLLECTIVE. HDF5 may choose a
    //    different underlying MPI-IO path when datatype conversion is necessary;
    //    file ordering alone does not guarantee filesystem locality or throughput.
    //  - Portable file types are little-endian integers/floats with explicit widths.
    //    CSR offsets are always UInt64; local/global indices and geometry retain
    //    the current build's widths. File layout does not depend on struct padding.
    //  - Stores owned/ghost cell data, required nodes, all local faces, full local
    //    CSR offsets including their terminal entries, metrics, patch strings,
    //    patch-face lists, local facet indices, halo maps and snapshot statistics.
    //  - The three face-region counts and faces_grouped are stored explicitly.
    //    Existing cell/face ordering is preserved; this step does not reorder mesh data.
    //  - Patch names, families and import labels have no fixed-length truncation.
    //    No variable-length HDF5 datatype or cell-sized conversion buffer is used.
    //  - The partition directory stores every payload length in partition order.
    //    The reader derives all prefixes from this directory, including halo and
    //    patch-face arrays: using an MPI rank-order Exscan here would be incorrect.
    //  - Open/write/close errors abort the job. Export relies on the validation
    //    already performed by the preceding stage instead of repeating its full scan.
    //
    // Solver restart contract:
    //  - import_mesh_hdf5(path,comm) returns a fully populated MeshPart and performs
    //    one local structural validation before returning it to solver kernels.
    //  - The reader requires the saved communicator size and rank numbering because
    //    donor/peer ranks are persisted. Physical host placement may change, but a
    //    different MPI size or reassignment of saved subdomains needs migration.
    //  - The saved permutation identifies this rank's partition; every mesh-sized
    //    read selects only that partition's payload. Small file metadata is replicated.
    //  - Wider index builds can read narrower files. Geometry precision must match:
    //    converting stored FP32 normals to FP64 does not make their original errors
    //    satisfy FP64 validation tolerances. Recompute geometry for precision changes.
    //    Unsafe index narrowing is rejected before loading payloads.
    //  - Legacy files written by the previous implementation are not schema-compatible;
    //    regenerate them rather than inferring the missing counts and dataset layouts.
    //
    // Optional visualization:
    //  - A nonempty vtudir requests two datasets: part.pvtu for owned volume cells
    //    and part_bnd.pvtu for physical boundary faces, each referencing one VTU per rank.
    //  - Pieces use appended raw binary with UInt64 block headers. Volume ghosts and
    //    partition-cut faces are omitted; shared points may occur in multiple pieces.
    //    Only nodes referenced by owned cells are exported, including boundary nodes.
    //  - Default arrays include rank, global cell/node IDs, volume, cell centroid,
    //    boundary patch ID, effective flux area and the owner's outward unit normal.
    //    Unlabelled boundary faces retain patch_id == -1.
    //  - Numeric point/cell/boundary fields can be passed through MeshFields.
    //    Scalars and interleaved vectors/tensors write directly; SoA vectors use a
    //    fixed 64-KiB packing buffer. Field spans and names are borrowed for the call.
    //  - MeshFields accepts full local arrays or arrays trimmed to the exported
    //    association. Field names, types and component counts must match across ranks.
    //  - The reusable GridView API also exports other linear unstructured grids
    //    with CGNS vertex ordering, independently of MeshPart and precomputed geometry.
    //  - Two barriers coordinate directory creation and completed piece files.
    //    Rank 0 writes only the small PVTU indexes; no mesh payload is gathered there.
    //
    // After this stage mp remains unchanged. The solver file contains the complete
    // current topology/geometry snapshot; optional VTU files serve visualization only.

    if (report)
        std::fputs("KRONA mesh_partition: writing solver mesh...\n", stderr);
    const auto mesh_write_started = std::chrono::steady_clock::now();

    krona::io::solver_mesh::export_mesh_hdf5(mp, options.output, comm, pr.part2rank);

    if (report) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mesh_write_started).count();
        std::fprintf(stderr, "  Solver mesh: %s\n", options.output.c_str());
        std::fprintf(stderr, "  Snapshot: cells=%llu nodes=%llu faces=%llu partitions=%d\n",
            static_cast<unsigned long long>(mp.n_cells_g),
            static_cast<unsigned long long>(mp.n_nodes_g),
            static_cast<unsigned long long>(mp.n_faces_g), comm.size());
        if (detailed)
            std::fprintf(stderr, "  Format: KRONA HDF5 v2, indices=%zu/%zu bits, geometry=%zu bits\n",
                sizeof(krona::LocalIndex) * 8, sizeof(krona::GlobalIndex) * 8,
                sizeof(krona::GeometryReal) * 8);
        std::fprintf(stderr, "KRONA mesh_partition: solver mesh written in %.3f s (rank 0).\n", seconds);
    }

    if (!options.vtu_directory.empty()) {
        if (report)
            std::fputs("KRONA mesh_partition: writing visualization...\n", stderr);
        const auto vtk_write_started = std::chrono::steady_clock::now();
        krona::io::vtk::write_vtu(mp, options.vtu_directory, "part", comm);
        if (report) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - vtk_write_started).count();
            std::fprintf(stderr, "  VTK datasets: %s/part.pvtu and %s/part_bnd.pvtu\n",
                        options.vtu_directory.c_str(), options.vtu_directory.c_str());
            std::fprintf(stderr, "KRONA mesh_partition: visualization written in %.3f s (rank 0).\n", seconds);
        }
    }

    const double total_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - global_start).count();

    if (report) 
        std::fprintf(stderr, "KRONA mesh_partition: mesh partiton is done successfully, total executional time %.3f s (rank 0).\n", total_time);
    
    return EXIT_SUCCESS;
}

[[nodiscard]] int execute(int argc, char* const* argv, krona::mpi::CommView comm)
{
    // parsing input arguments on master-rank
    ParseResult parsed;
    krona::mpi::LocalError local_err{};

    try {
        if (comm.rank() == 0) parsed = parse_arguments(argc, argv);
    } catch (const std::exception& error) {
        krona::mpi::set_local_error(local_err, error.what());
    } catch (...) {
        krona::mpi::set_local_error(local_err, "Unhandled arguments parsing exception");
    }
    krona::mpi::require_collective_success(comm, local_err, "Parsing Command Line Arguments");

    // broadcast data to other ranks
    int action = static_cast<int>(parsed.action);
    krona::mpi::broadcast(comm, action);
    if (action == static_cast<int>(CliAction::HELP)) {
        if (comm.rank() == 0) usage(stdout);
        return EXIT_SUCCESS;
    }
    if (action == static_cast<int>(CliAction::ERROR)) {
        if (comm.rank() == 0) {
            constexpr std::size_t limit = 512;
            std::fprintf(stderr, "mesh_partition: %.*s",
                static_cast<int>(std::min(parsed.error.size(), limit)), parsed.error.data());
            if (!parsed.argument.empty()) {
                std::fprintf(stderr, ": '%.*s'%s",
                    static_cast<int>(std::min(parsed.argument.size(), limit)), parsed.argument.data(),
                    parsed.argument.size() > limit ? " [truncated]" : "");
            }
            std::fputs("\n\n", stderr);
            usage(stderr);
        }
        return EXIT_FAILURE;
    }

    broadcast_options(parsed.options, comm);
    return run_partitioner(parsed.options, comm);
}

} // namespace

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
    const krona::mpi::CommView comm = communicator.view();
    try {
        return execute(argc, argv, comm);
    } catch (const std::exception& error) {
        krona::mpi::fatal(error.what());
    } catch (...) {
        krona::mpi::fatal("Unhandled mesh_partitioner exception");
    }
}
