#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace krona::boundary::physical {

/// @brief Identifies which parameter of a numeric condition a CSV file supplies.
enum class ProfileParameter { Pressure, Temperature, TotalPressure, TotalTemperature, Flux, Mach, Velocity, Direction };

/// @brief One CSV override for a scalar parameter or a component of a vector parameter.
struct ParameterProfile {
    ProfileParameter parameter; ///< Datum supplied by the file.
    std::size_t component;      ///< Zero for a scalar, or the Cartesian index for a vector.
    std::filesystem::path path; ///< CSV path to open on the root rank.
};

/// @brief Numeric condition with CSV overrides for selected parameters or vector components.
/// For a pressure outlet, direction and thermal parameters belong to its backflow data.
struct TabulatedConditionConfig {
    detail::ConditionVariant<double> numeric;
    std::vector<ParameterProfile> profiles;
};

/// @brief Condition parsed from TOML before it is converted to the manager's arithmetic type.
using ConfigCondition = detail::ConditionVariant<double, TabulatedConditionConfig>;

/// @brief One labelled patch's configuration, matched to the mesh using both ID and name.
struct PatchConfig {
    std::int64_t patch_id = 0;
    std::string name;
    ConfigCondition condition;
};

/// @brief Holds conditions for the mesh's labelled patches and any unlabelled boundary faces.
struct Configuration {
    std::vector<PatchConfig> patches;
    std::optional<ConfigCondition> unlabelled;
};

/// @brief Parses the physical boundary configuration on the calling rank.
/// @param text TOML text containing patch definitions.
[[nodiscard]] Configuration parse_config(std::string_view text);

/// @brief Reads TOML on rank 0, broadcasts the text and parses on every rank.
/// @param file Path opened by rank 0 of comm.
/// @param comm Communicator whose ranks all participate in the read.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, mpi::CommView comm);

} // namespace krona::boundary::physical
