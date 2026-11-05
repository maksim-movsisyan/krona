#include "cli.hpp"
#include <string_view>

namespace krona::apps::scalar_transport {
void usage(std::FILE* stream) noexcept
{
    std::fputs(
        "KRONA scalar transport\n\n"
        "Usage:\n"
        "  mpirun -np N krona_scalar_transport <case.toml> [options]\n\n"
        "Options:\n"
        "  -v, --verbose    Per-solve diagnostics.\n"
        "  -vv              Detailed configuration and diagnostics.\n"
        "  --check-config   Validate the case file without loading a mesh or solving.\n"
        "  -h, --help       Show this help. No arguments also show help.\n"
        "  --               End options; allows a case path starting with '-'.\n\n"
        "Steady/BE/BDF2; upwind/linear/MUSCL/TVD; optional cell limiting and\n"
        "nonorthogonal diffusion correction. N must match the saved mesh.\n"
        "Case paths are resolved relative to the case file.\n\n", stream);
}
ParseResult parse_arguments(int argc, char* const* argv)
{
    ParseResult result;
    if (argc <= 1) { result.action = CliAction::HELP; return result; }
    bool positional_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view token{argv[i]};
        if (!positional_only && token == "--") { positional_only = true; continue; }
        if (!positional_only && (token == "-h" || token == "--help")) {
            result.action = CliAction::HELP; return result;
        }
        if (!positional_only && (token == "-v" || token == "--verbose")) {
            if (result.options.verbosity == Verbosity::QUIET) result.options.verbosity = Verbosity::SUMMARY;
        } else if (!positional_only && token == "-vv") {
            result.options.verbosity = Verbosity::DETAILED;
        } else if (!positional_only && token == "--check-config") {
            result.options.check_config = true;
        } else if (!positional_only && token.size() > 1 && token.front() == '-') {
            result.action = CliAction::ERROR; result.error = "Unknown option: " + std::string{token}; return result;
        } else if (token.empty() || !result.options.case_file.empty()) {
            result.action = CliAction::ERROR; result.error = "Expected exactly one nonempty case-file path"; return result;
        } else result.options.case_file = token;
    }
    if (result.options.case_file.empty()) {
        result.action = CliAction::ERROR; result.error = "Missing case-file path";
    }
    return result;
}
} // namespace krona::apps::scalar_transport
