#pragma once

#include <cstdio>
#include <string>

namespace krona::apps::mesh_generator {

/// @brief Selects the amount of rank-zero diagnostic output.
enum class Verbosity : int { quiet, summary, detailed };

/// @brief Selects execution, successful help or a command-line error.
enum class CliAction : int { run, help, error };

/// @brief Stores the input/output paths and requested diagnostics.
struct Options {
    std::string input;
    std::string output;
    Verbosity verbosity = Verbosity::quiet;
};

/// @brief Owns parsed options and error diagnostics independently of argv.
struct ParseResult {
    Options options;
    CliAction action = CliAction::run;
    std::string error;
    std::string argument;
};

/// @brief Parses CLI syntax locally without MPI or file access.
[[nodiscard]] ParseResult parse_arguments(int argc, char *const *argv);

/// @brief Prints the command-line syntax to the chosen stream.
void usage(std::FILE *stream) noexcept;

/// @brief Prints a bounded CLI error and usage to the chosen stream.
void print_error(const ParseResult &result, std::FILE *stream) noexcept;

} // namespace krona::apps::mesh_generator
