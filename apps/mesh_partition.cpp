#include "krona/mpi/collective_status.hpp"
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <krona/io/cgns/reader.hpp>
#include <krona/mesh/extrusion.hpp>
#include <krona/mesh/faces.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <chrono>

namespace {

enum class ReorderMethod : int { none, sfc, rcm };
enum class Verbosity : int { quiet, summary, detailed };
enum class CliAction : int { run, help, error };

struct Options {
    std::string input;
    std::string output;
    std::optional<std::string> boundary_output;
    std::optional<std::string> vtu_directory;
    ReorderMethod reorder = ReorderMethod::none;
    Verbosity verbosity = Verbosity::quiet;
};

struct ParseResult {
    Options options;
    CliAction action = CliAction::run;
    std::string_view error;
    std::string_view argument;
};

[[nodiscard]] constexpr const char* reorder_name(ReorderMethod method) noexcept
{
    switch (method) {
    case ReorderMethod::none: return "NONE";
    case ReorderMethod::sfc:  return "SFC";
    case ReorderMethod::rcm:  return "RCM";
    }
    return "unknown";
}

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
        "  mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5 --bc mesh.bc\n"
        "  mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5 --vtu vtk --reorder RCM -v\n"
        "\n"
        "Development stage: CGNS import and automatic 2D extrusion; export is not implemented.\n",
        stream);
}

[[nodiscard]] bool option_token(std::string_view token) noexcept
{
    return token.size() > 1 && token.front() == '-';
}

// Parse only on rank 0; expected CLI errors are returned, not thrown.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv)
{
    ParseResult result;
    auto fail = [&](std::string_view error, std::string_view argument = {}) {
        result.action = CliAction::error;
        result.error = error;
        result.argument = argument;
    };
    if (argc <= 1) {
        result.action = CliAction::help;
        return result;
    }

    int positional_count = 0;
    bool positional_only = false;
    bool reorder_seen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view token{argv[i]};
        if (!positional_only && token == "--") {
            positional_only = true;
            continue;
        }
        if (!positional_only && (token == "-h" || token == "--help")) {
            result.action = CliAction::help;
            return result;
        }
        if (!positional_only && (token == "-v" || token == "--verbose")) {
            if (result.options.verbosity == Verbosity::quiet) {
                result.options.verbosity = Verbosity::summary;
            }
            continue;
        }
        if (!positional_only && token == "-vv") {
            result.options.verbosity = Verbosity::detailed;
            continue;
        }

        if (!positional_only && option_token(token)) {
            const auto equal = token.find('=');
            const auto name = token.substr(0, equal);
            if (name != "--bc" && name != "--vtu" && name != "--reorder") {
                fail("Unknown option", token);
                return result;
            }
            if ((name == "--bc" && result.options.boundary_output.has_value()) ||
                (name == "--vtu" && result.options.vtu_directory.has_value()) ||
                (name == "--reorder" && reorder_seen)) {
                fail("Duplicate option", name);
                return result;
            }

            std::string_view value;
            if (equal != std::string_view::npos) {
                value = token.substr(equal + 1);
            } else {
                if (i + 1 >= argc || option_token(argv[i + 1])) {
                    fail("Missing value for option", name);
                    return result;
                }
                value = argv[++i];
            }
            if (value.empty()) {
                fail("Empty value for option", name);
                return result;
            }

            if (name == "--bc") {
                result.options.boundary_output.emplace(value);
            } else if (name == "--vtu") {
                result.options.vtu_directory.emplace(value);
            } else {
                reorder_seen = true;
                if (value == "NONE") result.options.reorder = ReorderMethod::none;
                else if (value == "SFC") result.options.reorder = ReorderMethod::sfc;
                else if (value == "RCM") result.options.reorder = ReorderMethod::rcm;
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
        header[2] = options.boundary_output ? static_cast<int>(options.boundary_output->size()) : 0;
        header[3] = options.vtu_directory ? static_cast<int>(options.vtu_directory->size()) : 0;
        header[4] = static_cast<int>(options.reorder);
        header[5] = static_cast<int>(options.verbosity);
    }
    krona::mpi::broadcast(comm, std::span{header});

    if (comm.rank() != 0) {
        options.input.resize(static_cast<std::size_t>(header[0]));
        options.output.resize(static_cast<std::size_t>(header[1]));
        
        if (header[2] > 0) {
            options.boundary_output.emplace(static_cast<std::size_t>(header[2]), '\0');
        } else {
            options.boundary_output = std::nullopt;
        }

        if (header[3] > 0) {
            options.vtu_directory.emplace(static_cast<std::size_t>(header[3]), '\0');
        } else {
            options.vtu_directory = std::nullopt;
        }
    }

    krona::mpi::broadcast(comm, std::span{options.input});
    krona::mpi::broadcast(comm, std::span{options.output});
    if (header[2] > 0) {
        krona::mpi::broadcast(comm, std::span{*options.boundary_output});
    }
    if (header[3] > 0) {
        krona::mpi::broadcast(comm, std::span{*options.vtu_directory});
    }
    options.reorder = static_cast<ReorderMethod>(header[4]);
    options.verbosity = static_cast<Verbosity>(header[5]);
}

// Import the mesh and extrude 2D input to one 3D layer.
[[nodiscard]] int run_partitioner(const Options& options, krona::mpi::CommView comm)
{
    if (comm.rank() == 0) {
        if (options.verbosity != Verbosity::quiet) {
            std::fprintf(stderr,
                "KRONA mesh_partition: input=%s output=%s ranks=%d reorder=%s verbosity=%d\n"
                "  bc=%s\n  vtu=%s\n",
                options.input.c_str(), options.output.c_str(), comm.size(), reorder_name(options.reorder),
                static_cast<int>(options.verbosity),
                options.boundary_output ? options.boundary_output->c_str() : "<disabled>",
                options.vtu_directory ? options.vtu_directory->c_str() : "<disabled>");
        }
    }

    const bool report = comm.rank() == 0 && options.verbosity != Verbosity::quiet;
    const bool detailed = report && options.verbosity > Verbosity::summary;

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

    if (report)
        std::fputs("KRONA mesh_partition: importing mesh...\n", stderr);

    const auto import_started = std::chrono::steady_clock::now();

    auto imported = krona::io::cgns::read_parallel(options.input, comm);
    auto& mesh = imported.mesh;

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

            std::fprintf(stderr,
                "  Rank 0 owns: cells=%zu nodes=%zu boundary_elements=%zu\n",
                mesh.cells.size(), mesh.coordinates.size(), mesh.boundary.size());

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

    if (report)
        std::fputs("KRONA mesh_partition: building faces and dual graph...\n", stderr);

    const auto topology_started = std::chrono::steady_clock::now();

    auto topology = krona::mesh::build_faces(mesh, comm);
    auto& faces = topology.faces;
    auto& graph = topology.graph;

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
    if (action == static_cast<int>(CliAction::help)) {
        if (comm.rank() == 0) usage(stdout);
        return EXIT_SUCCESS;
    }
    if (action == static_cast<int>(CliAction::error)) {
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
