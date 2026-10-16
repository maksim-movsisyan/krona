#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace krona::boundary::physical {

using ConfigCondition = std::variant<Symmetry,SlipWall<double>,AdiabaticWall<double>,
    IsothermalWall<double>,HeatFluxWall<double>,PressureOutlet<double>,TotalConditionsInlet<double>,
    VelocityInlet<double>,SupersonicInlet<double>,SupersonicOutlet>;

struct PatchConfig { std::int64_t patch_id = 0; std::string name; ConfigCondition condition; };
struct Configuration { std::vector<PatchConfig> patches; std::optional<ConfigCondition> unlabelled; };

/// Local host parsing/reading, with the same failure convention as scalar::parse_config/read_config.
[[nodiscard]] Configuration parse_config(std::string_view text);
[[nodiscard]] Configuration read_config(const std::filesystem::path& file);

} // namespace krona::boundary::physical
