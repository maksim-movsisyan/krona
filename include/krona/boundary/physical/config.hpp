#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace krona::boundary::physical {

namespace detail {
template<class... Additional>
using ConfigVariant = std::variant<Symmetry,SlipWall<double>,AdiabaticWall<double>,
    IsothermalWall<double>,HeatFluxWall<double>,PressureOutlet<double>,TotalConditionsInlet<double>,
    VelocityInlet<double>,SupersonicInlet<double>,SupersonicOutlet,Additional...>;
} // namespace detail

enum class ProfileParameter { Pressure, Temperature, TotalPressure, TotalTemperature, Flux, Mach, Velocity, Direction };

struct ParameterProfile {
    ProfileParameter parameter;
    std::size_t component;
    std::filesystem::path path;
};

/// @brief Numeric condition with CSV overrides for selected parameters or vector components.
/// For a pressure outlet, direction and thermal parameters belong to its backflow data.
struct TabulatedConditionConfig {
    detail::ConfigVariant<> numeric;
    std::vector<ParameterProfile> profiles;
};

using ConfigCondition = detail::ConfigVariant<TabulatedConditionConfig>;

struct PatchConfig { std::int64_t patch_id = 0; std::string name; ConfigCondition condition; };
struct Configuration { std::vector<PatchConfig> patches; std::optional<ConfigCondition> unlabelled; };

/// @brief Parses TOML locally.
[[nodiscard]] Configuration parse_config(std::string_view text);
/// @brief Reads TOML on rank 0, broadcasts the text and parses on every rank.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, mpi::CommView comm);

} // namespace krona::boundary::physical