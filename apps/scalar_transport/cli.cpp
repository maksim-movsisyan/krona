#include "cli.hpp"

#include <algorithm>
#include <string_view>

namespace krona::apps::scalar_transport {

/// @brief Prints help without accessing files.
void usage(std::FILE* stream) noexcept {
    std::fputs("KRONA scalar convection-diffusion solver\n"
               "\n"
               "Usage:\n"
               "  mpirun -np N krona_scalar_transport <mesh.h5> <transport.toml> <bc.toml> [options]\n"
               "  krona_scalar_transport --help\n"
               "\n"
               "Options:\n"
               "  --field=<name>             Select a scalar (default: phi).\n"
               "  --case=<name>              C++ setup: custom, diffusion, gaussian (default: custom).\n"
               "  --vtu=<dir>                Write the final field to VTU files.\n"
               "  -v, --verbose              Summary diagnostics.\n"
               "  -vv                        Step and iteration diagnostics.\n"
               "  -h, --help                 Show this help.\n"
               "  --                         End options; remaining arguments are paths.\n"
               "\n"
               "Named options also accept --option value.\n"
               "Use the MPI size that produced mesh.h5.\n",
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
    bool field_seen = false;
    bool case_seen = false;
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

            if (name != "--field" && name != "--vtu" && name != "--case") {
                fail("Unknown option", token);
                return result;
            }

            if ((name == "--field" && field_seen) || (name == "--case" && case_seen) ||
                (name == "--vtu" && !result.options.vtu_directory.empty())) {
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

            if (name == "--field") {
                field_seen = true;
                result.options.field = value;
            } else if (name == "--vtu") {
                result.options.vtu_directory = value;
            } else {
                case_seen = true;
                if (value == "custom") result.options.problem = Case::custom;
                else if (value == "diffusion") result.options.problem = Case::diffusion;
                else if (value == "gaussian") result.options.problem = Case::gaussian;
                else {
                    fail("Invalid case; expected custom, diffusion or gaussian", value);
                    return result;
                }
            }
            continue;
        }

        if (token.empty()) {
            fail("Paths must not be empty");
            return result;
        }
        if (positional_count == 0)
            result.options.mesh = token;
        else if (positional_count == 1)
            result.options.config = token;
        else if (positional_count == 2)
            result.options.boundary = token;
        else {
            fail("Unexpected positional argument", token);
            return result;
        }
        ++positional_count;
    }
    if (positional_count != 3) {
        fail("Expected solver mesh, transport configuration and boundary configuration paths");
    }
    return result;
}

/// @brief Limits diagnostic lengths before writing rank-zero error output.
void print_error(const ParseResult& result, std::FILE* stream) noexcept {
    constexpr std::size_t limit = 512;
    std::fprintf(stream, "scalar_transport: %.*s", static_cast<int>(std::min(result.error.size(), limit)),
                 result.error.data());
    if (!result.argument.empty())
        std::fprintf(stream, ": '%.*s'%s", static_cast<int>(std::min(result.argument.size(), limit)),
                     result.argument.data(), result.argument.size() > limit ? " [truncated]" : "");
    std::fputs("\n\n", stream);
    usage(stream);
}

} // namespace krona::apps::scalar_transport
