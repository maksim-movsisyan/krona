#pragma once

#include <krona/config.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace krona::boundary::scalar {

enum class Kind { FixedValue, NormalGradient, FixedFlux };

/// @brief Host configuration; interpretation of parameter depends on kind.
struct ConditionConfig {
    Kind kind = Kind::FixedValue;
    double parameter = 0;
};

struct PatchConfig {
    std::int64_t patch_id = 0;
    std::string name;
    ConditionConfig condition;
};

/// @brief One scalar field's conditions; patch IDs are checked when binding a mesh.
struct Configuration {
    std::string field;
    std::vector<PatchConfig> patches;
    std::optional<ConditionConfig> unlabelled;
};

/// @brief Parses one field from TOML text; malformed input terminates through mpi::fatal.
[[nodiscard]] Configuration parse_config(std::string_view text, std::string_view field);

/// @brief Reads locally on each caller; performs no MPI communication or broadcast.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, std::string_view field);

} // namespace krona::boundary::scalar
