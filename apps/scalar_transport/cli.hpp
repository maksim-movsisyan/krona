#pragma once

#include <cstdio>
#include <string>

namespace krona::apps::scalar_transport {

enum class Verbosity : int { quiet, summary, detailed };
enum class CliAction : int { run, help, error };
enum class Case : int { custom, diffusion, gaussian };

struct Options {
    std::string mesh;
    std::string config;
    std::string boundary;
    std::string vtu_directory;
    std::string field = "phi";
    Case problem = Case::custom;
    Verbosity verbosity = Verbosity::quiet;
};

struct ParseResult {
    Options options;
    CliAction action = CliAction::run;
    std::string error;
    std::string argument;
};

/// @brief Parses command-line syntax locally without file or MPI access.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv);
void usage(std::FILE* stream) noexcept;
void print_error(const ParseResult& result, std::FILE* stream) noexcept;

} // namespace krona::apps::scalar_transport
