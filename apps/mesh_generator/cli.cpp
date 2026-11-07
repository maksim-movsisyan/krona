#include "cli.hpp"

#include <algorithm>
#include <string_view>

namespace krona::apps::mesh_generator {

/// @brief Prints command-line syntax to the selected stream.
void usage(std::FILE *stream) noexcept {
    std::fputs("KRONA distributed mesh generator\n"
               "\n"
               "Usage:\n"
               "  mpirun -np N krona_mesh_generator <in.toml> <out.cgns> [options]\n"
               "  krona_mesh_generator --help\n"
               "\n"
               "Options:\n"
               "  -v, --verbose      Summary diagnostics.\n"
               "  -vv                Detailed diagnostics.\n"
               "  -h, --help         Show this help.\n"
               "  --                 End options; remaining arguments are paths.\n"
               "\n"
               "Examples:\n"
               "  mpirun -np 4 krona_mesh_generator mesh_config.toml result_mesh.cgns\n"
               "  mpirun -np 4 krona_mesh_generator mesh_config.toml result_mesh.cgns --verbose\n",
               stream);
}

/// @brief Parses paths and options without consulting MPI or opening files.
[[nodiscard]] ParseResult parse_arguments(int argc, char *const *argv) {
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

    // Options may appear around the two paths until an explicit -- delimiter.
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

        if (!positional_only && token.starts_with('-')) {
            fail("Unknown option", token);
            return result;
        }
        if (token.empty()) {
            fail("Input and output paths must not be empty");
            return result;
        }
        if (positional_count == 0)
            result.options.input = token;
        else if (positional_count == 1)
            result.options.output = token;
        else {
            fail("Unexpected positional argument", token);
            return result;
        }
        ++positional_count;
    }
    // A run needs exactly one input TOML file and one output CGNS path.
    if (positional_count != 2) {
        fail("Expected an input TOML config and an output CGNS path");
    }
    return result;
}

/// @brief Prints bounded diagnostics so a long argument cannot flood the log.
void print_error(const ParseResult &result, std::FILE *stream) noexcept {
    constexpr std::size_t limit = 512;
    std::fprintf(stream, "krona_mesh_generator: %.*s", static_cast<int>(std::min(result.error.size(), limit)),
                 result.error.data());
    if (!result.argument.empty()) {
        std::fprintf(stream, ": '%.*s'%s", static_cast<int>(std::min(result.argument.size(), limit)),
                     result.argument.data(), result.argument.size() > limit ? " [truncated]" : "");
    }
    std::fputs("\n\n", stream);
    usage(stream);
}

} // namespace krona::apps::mesh_generator
