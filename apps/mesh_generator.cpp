#include "krona/mpi/collective_status.hpp"
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mesh_generation/config.hpp>
#include <krona/mesh_generation/topology.hpp>
#include <krona/mesh_generation/writer.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <string>
#include <string_view>

namespace {

enum class Verbosity : int { quiet, summary, detailed };
enum class CliAction : int { run, help, error };

struct Options {
    std::string input;
    std::string output;
    Verbosity verbosity = Verbosity::quiet;
};

struct ParseResult {
    Options options;
    CliAction action = CliAction::run;
    std::string_view error;
    std::string_view argument;
};

// Print on rank 0; no arguments and --help are successful help requests.
void usage(std::FILE* stream) noexcept {
    std::fputs(
        "KRONA distributed mesh generator\n"
        "\n"
        "Usage:\n"
        "  mpirun -np N mesh_generation <in.toml> <out.cgns> [options]\n"
        "  mesh_generation --help\n"
        "\n"
        "Options:\n"
        "  -v, --verbose      Summary diagnostics.\n"
        "  -vv                Detailed diagnostics.\n"
        "  -h, --help         Show this help.\n"
        "  --                 End options; remaining arguments are paths.\n"
        "\n"
        "Examples:\n"
        "  mpirun -np 4 mesh_generation mesh_config.toml result_mesh.cgns\n"
        "  mpirun -np 4 mesh_generation mesh_config.toml result_mesh.cgns --verbose\n",
        stream);
}

// Parse only on rank 0; expected CLI errors are returned, not thrown.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv) {
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
        fail("Expected an input TOML config and an output CGNS path");
    }
    return result;
}

// Replicate root's settings; absence of optional paths is encoded by length zero.
void broadcast_options(Options& options, krona::mpi::CommView comm) {
    std::array<int, 3> header{};

    if (comm.rank() == 0) {
        header[0] = static_cast<int>(options.input.size());
        header[1] = static_cast<int>(options.output.size());
        header[2] = static_cast<int>(options.verbosity);
    }
    krona::mpi::broadcast(comm, std::span{header});

    if (comm.rank() != 0) {
        options.input.resize(static_cast<std::size_t>(header[0]));
        options.output.resize(static_cast<std::size_t>(header[1]));
    }

    krona::mpi::broadcast(comm, std::span{options.input.data(), options.input.size()});
    krona::mpi::broadcast(comm, std::span{options.output.data(), options.output.size()});
    options.verbosity = static_cast<Verbosity>(header[2]);
}

[[nodiscard]] int run_generator(const Options& options, krona::mpi::CommView comm) {
    if (comm.rank() == 0) {
        if (options.verbosity > Verbosity::quiet) {
            std::fprintf(stderr,
                "KRONA mesh_generation: input=%s output=%s ranks=%d verbosity=%d\n",
                options.input.c_str(), options.output.c_str(), comm.size(),
                static_cast<int>(options.verbosity));
        }
    }


    // Parse configuration and validate all design contracts collectively
    const krona::mesh_generation::GeneratorConfig cfg 
        = krona::mesh_generation::parse_generator_config(options.input, comm.native());

    if (options.verbosity >= Verbosity::summary) {
        // Compute mesh volume statistics
        std::uint64_t total_volume_cells = 0;
        for (const auto& block : cfg.blocks) {
            total_volume_cells += static_cast<std::uint64_t>(block.cells[0]) *
                                static_cast<std::uint64_t>(block.cells[1]) *
                                static_cast<std::uint64_t>(block.cells[2]);
        }

        std::uint64_t total_boundary_faces = 0;
        for (const auto& patch : cfg.boundaries) {
            total_boundary_faces += patch.faces.size();
        }

        // Report results on Rank 0
        if (comm.rank() == 0) {
            std::fprintf(stdout, "\n[Config Validation Succeeded]\n");
            std::fprintf(stdout, "  Mesh Name:         %s\n", cfg.name.c_str());
            std::fprintf(stdout, "  Scaling Factor:    %f\n", cfg.scale); // Используйте %.f, если нужна точность
            std::fprintf(stdout, "  Macro Vertices:    %zu\n", cfg.vertices.size());
            std::fprintf(stdout, "  Macro Blocks:      %zu\n", cfg.blocks.size());
            std::fprintf(stdout, "  Boundary Patches:  %zu\n", cfg.boundaries.size());
            std::fprintf(stdout, "  Total Hex Cells:   %llu\n", static_cast<unsigned long long>(total_volume_cells));
            std::fprintf(stdout, "  Macro Bnd Faces:   %llu\n", static_cast<unsigned long long>(total_boundary_faces));


            if (options.verbosity == Verbosity::detailed) {
                std::fprintf(stdout, "\n === Blocks Breakdown ===\n");
                for (std::size_t i = 0; i < cfg.blocks.size(); ++i) {
                    const auto& b = cfg.blocks[i];
                    const auto b_cells = static_cast<std::uint64_t>(b.cells[0]) * b.cells[1] * b.cells[2];
                    std::fprintf(stdout, "  [%zu] %s: %llux%llux%llu (%llu cells)\n",
                         i, b.name.c_str(), static_cast<unsigned long long>(b.cells[0]), 
                                            static_cast<unsigned long long>(b.cells[1]), 
                                            static_cast<unsigned long long>(b.cells[2]), 
                         static_cast<unsigned long long>(b_cells));

                    std::string type0{ krona::mesh_generation::to_string(b.grading_type[0]) };
                    std::string type1{ krona::mesh_generation::to_string(b.grading_type[1]) };
                    std::string type2{ krona::mesh_generation::to_string(b.grading_type[2]) };

                    std::fprintf(stdout, "      Grading Types: [%s, %s, %s]\n",
                                 type0.c_str(), type1.c_str(), type2.c_str());
            
                    std::fprintf(stdout, "      Grading Values: [%f, %f, %f]\n",
                                b.grading[0], b.grading[1], b.grading[2]);
                }

                std::fprintf(stdout, "\n === Boundary Patches ===\n");
                for (const auto& patch : cfg.boundaries) {
                    std::fprintf(stdout, "  Patch '%s': %zu macro-faces\n", patch.name.c_str(), patch.faces.size());
                }
            }
        }
    }

    // Building mesh macro topology
    krona::mesh_generation::MacroTopology topo;
    topo.build(cfg, MPI_COMM_WORLD);

    // Output mesh
    krona::mesh_generation::write(options.output, cfg, topo, MPI_COMM_WORLD);

    return EXIT_SUCCESS;
}

[[nodiscard]] int execute(int argc, char* const* argv, krona::mpi::CommView comm) {
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
    return run_generator(parsed.options, comm);
}

} // namespace

int main(int argc, char** argv) {
    // Initialize RAII MPI runtime wrapper
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};

    // Create MPI_COMM_WORLD communicator duplicate
    krona::mpi::Comm communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
    const krona::mpi::CommView comm = communicator.view();

    // Execution try-catch
    try {
        return execute(argc, argv, comm);
    } catch (const std::exception& error) {
        krona::mpi::fatal(error.what());
    } catch (...) {
        krona::mpi::fatal("Unhandled mesh_partitioner exception");
    }
}
