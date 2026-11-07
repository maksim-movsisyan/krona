#include "cli.hpp"

#include <algorithm>
#include <string_view>

namespace krona::apps::mesh_partitioner {

/// @brief Prints help without reading or creating mesh files.
void usage(std::FILE* stream) noexcept {
    std::fputs("KRONA distributed mesh partitioner\n"
               "\n"
               "Usage:\n"
               "  mpirun -np N krona_mesh_partitioner <in.cgns> <out.h5> [options]\n"
               "  krona_mesh_partitioner --help\n"
               "\n"
               "Options:\n"
               "  --bc <bc.toml>       Write a boundary-condition template (optional).\n"
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
               "  mpirun -np 4 krona_mesh_partitioner mesh.cgns mesh.h5\n"
               "  mpirun -np 4 krona_mesh_partitioner mesh.cgns mesh.h5 --bc mesh.toml\n"
               "  mpirun -np 4 krona_mesh_partitioner mesh.cgns mesh.h5 --vtu vtk --reorder RCM -v\n"
               "\n",
               stream);
}

/// @brief Collects paths and options, returning expected CLI errors as data.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv) {
    ParseResult result;

    auto is_optional = [](std::string_view token) { return token.size() > 1 && token.front() == '-'; };

    auto fail = [&](std::string_view error, std::string_view argument = {}) {
        result.action = CliAction::error;
        result.error = error;
        result.argument = argument;
    };

    if (argc <= 1) {
        result.action = CliAction::help;
        return result;
    }

    // Separate positional paths from options until an explicit -- delimiter.
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
            result.options.verbosity = Verbosity::summary;
            continue;
        }

        if (!positional_only && token == "-vv") {
            result.options.verbosity = Verbosity::detailed;
            continue;
        }

        if (!positional_only && is_optional(token)) {
            // Named options accept both --option value and --option=value.
            const auto equal = token.find('=');
            const auto name = token.substr(0, equal);

            if (name != "--bc" && name != "--vtu" && name != "--reorder") {
                fail("Unknown option", token);
                return result;
            }

            if ((name == "--bc" && !result.options.boundary_output.empty()) ||
                (name == "--vtu" && !result.options.vtu_directory.empty()) ||
                (name == "--reorder" && reorder_seen)) {
                fail("Duplicate option", name);
                return result;
            }

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
                if (value == "NONE")
                    result.options.reorder = krona::mesh::host::ReorderMethod::NONE;
                else if (value == "SFC")
                    result.options.reorder = krona::mesh::host::ReorderMethod::HILBERT_SFC;
                else if (value == "RCM")
                    result.options.reorder = krona::mesh::host::ReorderMethod::RCM;
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
    if (positional_count != 2) {
        fail("Expected an input CGNS path and an output HDF5 path");
    }
    return result;
}

/// @brief Limits diagnostic lengths before writing rank-zero error output.
void print_error(const ParseResult& result, std::FILE* stream) noexcept {
    constexpr std::size_t limit = 512;
    std::fprintf(stream, "mesh_partitioner: %.*s", static_cast<int>(std::min(result.error.size(), limit)),
                 result.error.data());
    if (!result.argument.empty())
        std::fprintf(stream, ": '%.*s'%s", static_cast<int>(std::min(result.argument.size(), limit)),
                     result.argument.data(), result.argument.size() > limit ? " [truncated]" : "");
    std::fputs("\n\n", stream);
    usage(stream);
}

} // namespace krona::apps::mesh_partitioner
