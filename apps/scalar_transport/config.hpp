#pragma once
#include <krona/transport/scalar/config.hpp>
#include <filesystem>
#include <string_view>
namespace krona::apps::scalar_transport {
using transport::scalar::ConstantInitial;
using transport::scalar::LinearInitial;
using transport::scalar::GaussianInitial;
enum class Mode { Steady, Transient };
struct Configuration : transport::scalar::Configuration {
    std::filesystem::path mesh_file, boundary_file, output_directory;
    Mode mode = Mode::Steady;
    double dt = 0.01, end_time = 1;
    int maximum_steps = 100000, output_interval = 1;
    bool output_enabled = true;
};
[[nodiscard]] Configuration parse_config(std::string_view text, const std::filesystem::path& case_file);
} // namespace krona::apps::scalar_transport
