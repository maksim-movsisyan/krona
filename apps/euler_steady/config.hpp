#pragma once

#include <krona/mpi/communicator.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <string_view>

namespace krona::apps::euler_steady {

enum class Flux { Hllc, Hll, Rusanov };
enum class ReconstructionOrder { First, Second };
enum class Limiter { None, BarthJespersen };

struct InitialOptions {
    double pressure = 0;
    double temperature = 0;
    std::array<double, 3> velocity{};
};

struct PseudoTimeOptions {
    double cfl = 1.0;
    double relaxation = 1.0;
    double absolute_tolerance = 1e-8;
    double relative_tolerance = 1e-8;
    int maximum_iterations = 200;
};

struct LinearOptions {
    std::string method = "GMRES";
    std::string preconditioner = "RELAXATION";
    double relative_tolerance = 1e-6;
    double absolute_tolerance = 0;
    int maximum_iterations = 100;
    int restart_length = 30;
    bool verify_residual = false;
};

struct Configuration {
    double gamma = 1.4;
    double gas_constant = 287.05;
    InitialOptions initial;
    PseudoTimeOptions pseudo_time;
    LinearOptions linear;
    Flux flux = Flux::Hllc;
    ReconstructionOrder reconstruction = ReconstructionOrder::First;
    Limiter limiter = Limiter::None;
    std::filesystem::path output_directory;
};

/// @brief Parses and validates the solver TOML locally.
[[nodiscard]] Configuration parse_config(std::string_view text);

/// @brief Reads on rank zero, broadcasts the text and parses on every rank.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, mpi::CommView comm);

} // namespace krona::apps::euler_steady
