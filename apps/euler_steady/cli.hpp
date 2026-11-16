#pragma once

#include <cstdio>
#include <string>

namespace krona::apps::euler_steady {

enum class Verbosity : int { quiet, summary, detailed };
enum class CliAction : int { run, help, error };
enum class Case : int { custom, channel, wedge };

struct Options {
    std::string mesh;
    std::string config;
    std::string boundary;
    std::string vtu_directory;
    Case problem = Case::custom;
    Verbosity verbosity = Verbosity::quiet;
};

struct ParseResult {
    Options options;
    CliAction action = CliAction::run;
    std::string error;
    std::string argument;
};

/// @brief Parses local command-line syntax. Rank zero broadcasts the result in the driver.
[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv);
void usage(std::FILE* stream) noexcept;
void print_error(const ParseResult& result, std::FILE* stream) noexcept;

} // namespace krona::apps::euler_steady
