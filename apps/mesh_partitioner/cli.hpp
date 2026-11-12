#pragma once

#include <krona/mesh/host/reorder.hpp>

#include <cstdio>
#include <string>

namespace krona::apps::mesh_partitioner {

/// @brief Selects the amount of rank-zero diagnostic output. 
enum class Verbosity : int { quiet, summary, detailed };

/// @brief Selects execution, successful help or a command-line error.
enum class CliAction : int { run, help, error };

/// @brief Stores mesh paths, optional exports and the local ordering method.
struct Options {
    std::string input;
    std::string output;
    std::string boundary_output;
    std::string vtu_directory;
    std::string patches_template_directory;
    mesh::host::ReorderMethod reorder = mesh::host::ReorderMethod::NONE;
    Verbosity verbosity = Verbosity::quiet;
};

/// @brief Owns parsed options and diagnostic text independently of argv.
struct ParseResult {
    Options options;
    CliAction action = CliAction::run;
    std::string error;
    std::string argument;
};

/// @brief Parses command-line syntax locally without MPI or file access.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv);

/// @brief Prints command-line syntax and supported options.
void usage(std::FILE* stream) noexcept;

/// @brief Prints a bounded argument diagnostic followed by usage.
void print_error(const ParseResult& result, std::FILE* stream) noexcept;

} // namespace krona::apps::mesh_partitioner
