#pragma once
#include <cstdio>
#include <string>

namespace krona::apps::scalar_transport {
enum class CliAction : int { RUN, HELP, ERROR };
enum class Verbosity : int { QUIET, SUMMARY, DETAILED };
struct CliOptions {
    std::string case_file;
    Verbosity verbosity = Verbosity::QUIET;
    bool check_config = false;
};
struct ParseResult {
    CliOptions options;
    CliAction action = CliAction::RUN;
    std::string error;
};
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv);
void usage(std::FILE* stream) noexcept;
} // namespace krona::apps::scalar_transport
