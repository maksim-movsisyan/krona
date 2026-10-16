#pragma once

#include <krona/boundary/physical/conditions.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace krona::boundary::physical {

enum class Kind {
    VelocityInlet, TotalInlet, SupersonicInlet, PressureOutlet, InletOutlet,
    SupersonicOutlet, AdiabaticWall, IsothermalWall, SlipWall, Symmetry, Extrapolate
};

/// @brief Physical data from one shared boundary_condition entry.
struct ConditionConfig {
    Kind kind = Kind::Symmetry;
    std::array<double, 3> velocity{}, direction{};
    std::optional<double> pressure, temperature, total_pressure, total_temperature;
};
struct PatchConfig {
    std::int64_t patch_id = 0;
    std::string name;
    ConditionConfig condition;
};
struct Configuration {
    std::vector<PatchConfig> patches;
    std::optional<ConditionConfig> unlabelled;
};

/// @brief Reads boundary_condition metadata/type and physical parameters; ignores scalar sections.
[[nodiscard]] Configuration parse_config(std::string_view text);
/// @brief Local read on each caller; no broadcast or other MPI communication.
[[nodiscard]] Configuration read_config(const std::filesystem::path& path);

} // namespace krona::boundary::physical
