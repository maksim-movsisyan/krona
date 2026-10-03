// Distributed mesh preprocessing entry point.

#include <krona/io/cgns/reader.hpp>
#include <krona/mesh/extrusion.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/runtime.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>

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
    std::array<int, 6> header{};
    const auto prepared = krona::mpi::LocalStatus::capture([&]() -> krona::mpi::LocalStatus {
        if (comm.rank() != 0) return {};
        const std::array<std::size_t, 4> sizes{
            options.input.size(), options.output.size(),
            options.boundary_output ? options.boundary_output->size() : 0,
            options.vtu_directory ? options.vtu_directory->size() : 0
        };
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                return krona::mpi::LocalStatus::failure("Command-line path exceeds MPI 3.1 count limit");
            }
            header[i] = static_cast<int>(sizes[i]);
        }
        header[4] = static_cast<int>(options.reorder);
        header[5] = static_cast<int>(options.verbosity);
        return {};
    });
    krona::mpi::require_collective_success(comm, prepared, "Command-line serialization");
    krona::mpi::broadcast(comm, std::span{header});

    const auto allocated = krona::mpi::LocalStatus::capture([&] {
        if (comm.rank() == 0) return;
        options.input.resize(static_cast<std::size_t>(header[0]));
        options.output.resize(static_cast<std::size_t>(header[1]));
        if (header[2] != 0) options.boundary_output.emplace(static_cast<std::size_t>(header[2]), '\0');
        else options.boundary_output.reset();
        if (header[3] != 0) options.vtu_directory.emplace(static_cast<std::size_t>(header[3]), '\0');
        else options.vtu_directory.reset();
    });
    krona::mpi::require_collective_success(comm, allocated, "Command-line receive allocation");

    krona::mpi::broadcast(comm, std::span{options.input.data(), options.input.size()});
    krona::mpi::broadcast(comm, std::span{options.output.data(), options.output.size()});
    if (options.boundary_output) {
        krona::mpi::broadcast(comm, std::span{options.boundary_output->data(), options.boundary_output->size()});
    }
    if (options.vtu_directory) {
        krona::mpi::broadcast(comm, std::span{options.vtu_directory->data(), options.vtu_directory->size()});
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

    auto imported = krona::io::cgns::read_parallel(options.input, comm);
    auto &mesh = imported.mesh;
    if (mesh.cell_dimension == 2) {
        const auto extrusion = krona::mesh::extrude_one_layer(mesh, comm);
        if (comm.rank() == 0 && options.verbosity != Verbosity::quiet) {
            std::fprintf(stderr, "  extrusion: thickness=%.17g normal=%c%s\n",
                static_cast<double>(extrusion.thickness), extrusion.normal_sign > 0 ? '+' : '-',
                extrusion.normal_axis == 0 ? "X" : extrusion.normal_axis == 1 ? "Y" : "Z");
        }
    }
    if (comm.rank() == 0) {
        if (options.verbosity != Verbosity::quiet) {
            std::fprintf(stderr, "  Imported: cells=%lld nodes=%lld boundary=%lld\n",
                static_cast<long long>(mesh.distribution.cells.global_size()),
                static_cast<long long>(mesh.distribution.nodes.global_size()),
                static_cast<long long>(mesh.distribution.boundary_ordinals.global_size()));

            if (options.verbosity > Verbosity::summary) {
                std::fprintf(stderr, "  Boundary zones: \n");

                for (std::size_t i = 0; i < mesh.patches.size(); ++i) {
                    std::fprintf(stderr, "  boundary: name=%s family=%s source=%s\n",
                        mesh.patches[i].name.c_str(),
                        mesh.patches[i].family.c_str(),
                        mesh.patches[i].source_type.c_str());
                }
            }
        }
        std::fputs("KRONA mesh_partition: import completed; partitioning/export are not implemented.\n", stderr);
    }
    // TODO: topology, partitioning, migration, geometry and export.
    return EXIT_SUCCESS;
}

[[nodiscard]] int execute(int argc, char* const* argv, krona::mpi::CommView comm)
{
    ParseResult parsed;
    const auto prepared = krona::mpi::LocalStatus::capture([&] {
        if (comm.rank() == 0) parsed = parse_arguments(argc, argv);
    });
    krona::mpi::require_collective_success(comm, prepared, "Command-line parsing");

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
    auto communicator = krona::mpi::Communicator::duplicate(krona::mpi::CommView::world());
    const auto comm = communicator.view();
    try {
        return execute(argc, argv, comm);
    } catch (const std::exception& error) {
        krona::mpi::fatal(comm.abort_scope(), error.what());
    } catch (...) {
        krona::mpi::fatal(comm.abort_scope(), "Unhandled mesh_partitioner exception");
    }
}
