#include "driver.hpp"
#include "cli.hpp"

#include <krona/toml_config/file.hpp>
#include <krona/mesh/generation/config.hpp>
#include <krona/mesh/generation/topology.hpp>
#include <krona/mesh/generation/writer.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace krona::apps::mesh_generator {
namespace {

/// @brief Replicates rank-zero CLI settings using the common string broadcast.
void broadcast_options(Options &options, krona::mpi::CommView comm) {
    int verbosity = static_cast<int>(options.verbosity);
    krona::mpi::broadcast(comm, verbosity);
    krona::toml_config::broadcast(options.input, comm);
    krona::toml_config::broadcast(options.output, comm);
    options.verbosity = static_cast<Verbosity>(verbosity);
}

/// @brief Prints mesh sizes and configuration metadata on rank zero.
void print_summary(const krona::mesh::generation::GeneratorConfig &config,
                   const krona::mesh::generation::MacroTopology &topo) {
    std::size_t macro_faces = 0;
    for (const auto &patch : topo.boundary_patches())
        macro_faces += patch.faces.size();
    std::fprintf(stdout,
                 "\n[Config Validation Succeeded]\n"
                 "  Mesh Name:         %s\n"
                 "  Scaling Factor:    %g\n"
                 "  Macro Vertices:    %zu\n"
                 "  Macro Blocks:      %zu\n"
                 "  Boundary Patches:  %zu\n"
                 "  Total Hex Cells:   %llu\n"
                 "  Macro Bnd Faces:   %zu\n",
                 config.name.c_str(), config.scale, config.vertices.size(), config.blocks.size(),
                 topo.boundary_patches().size(), static_cast<unsigned long long>(topo.total_cells()),
                 macro_faces);
}

/// @brief Prints each block's grading and each patch's macro-face count.
void print_details(const krona::mesh::generation::GeneratorConfig &config,
                   const krona::mesh::generation::MacroTopology &topo) {
    std::fputs("\n === Blocks Breakdown ===\n", stdout);
    for (std::size_t i = 0; i < config.blocks.size(); ++i) {
        const auto &block = config.blocks[i];
        const auto cells = static_cast<std::uint64_t>(block.cells[0]) * block.cells[1] * block.cells[2];
        std::fprintf(stdout, "  [%zu] %s: %zux%zux%zu (%llu cells)\n", i, block.name.c_str(), block.cells[0],
                     block.cells[1], block.cells[2], static_cast<unsigned long long>(cells));
        std::fprintf(stdout, "      Grading Types: [%s, %s, %s]\n",
                     krona::mesh::generation::to_string(block.grading_type[0]).data(),
                     krona::mesh::generation::to_string(block.grading_type[1]).data(),
                     krona::mesh::generation::to_string(block.grading_type[2]).data());
        std::fprintf(stdout, "      Grading Values: [%g, %g, %g]\n", block.grading[0], block.grading[1],
                     block.grading[2]);
    }
    std::fputs("\n === Boundary Patches ===\n", stdout);
    for (const auto &patch : topo.boundary_patches())
        std::fprintf(stdout, "  Patch '%s': %zu macro-faces\n", patch.name.c_str(), patch.faces.size());
}

/// @brief Reads the mesh, builds shared topology and writes CGNS collectively.
[[nodiscard]] int run_generator(const Options &options, krona::mpi::CommView comm) {
    if (comm.rank() == 0 && options.verbosity > Verbosity::quiet)
        std::fprintf(stderr, "KRONA mesh generator: input=%s output=%s ranks=%d verbosity=%d\n",
                     options.input.c_str(), options.output.c_str(), comm.size(),
                     static_cast<int>(options.verbosity));

    // Reading is collective; topology construction is deterministic and local.
    const auto config = krona::mesh::generation::parse_generator_config(options.input, comm);
    krona::mesh::generation::MacroTopology topo;
    topo.build(config, comm);

    if (comm.rank() == 0 && options.verbosity >= Verbosity::summary) {
        print_summary(config, topo);
        if (options.verbosity == Verbosity::detailed)
            print_details(config, topo);
    }

    // All ranks participate, including ranks with empty write intervals.
    krona::mesh::generation::write(options.output, config, topo, comm);
    return EXIT_SUCCESS;
}

} // namespace

/// @brief Keeps CLI failures and collective generation inside the MPI error guard.
int run(int argc, char *const *argv, mpi::CommView comm) noexcept {
    return mpi::guarded([&] {
        mpi::require(!comm.is_null(), "Mesh generator app requires a live communicator");
        // Parse paths and options only on the master rank.
        ParseResult parsed;
        if (comm.rank() == 0)
            parsed = parse_arguments(argc, argv);

        // Every rank takes the same help, error or run branch.
        int action = static_cast<int>(parsed.action);
        krona::mpi::broadcast(comm, action);
        if (action == static_cast<int>(CliAction::help)) {
            if (comm.rank() == 0)
                usage(stdout);
            return EXIT_SUCCESS;
        }
        if (action == static_cast<int>(CliAction::error)) {
            if (comm.rank() == 0)
                print_error(parsed, stderr);
            return EXIT_FAILURE;
        }

        broadcast_options(parsed.options, comm);
        return run_generator(parsed.options, comm);
    });
}

} // namespace krona::apps::mesh_generator
