#pragma once

#include <krona/config.hpp>
#include <krona/mpi/communicator.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::scalar {

enum class Kind { FixedValue, NormalGradient, FixedFlux };

/// @brief Host configuration; interpretation of parameter depends on kind.
struct ConditionConfig {
    using Parameter = std::variant<double, std::filesystem::path>;

    Kind kind = Kind::FixedValue;
    Parameter parameter = 0.0;

    ConditionConfig() = default;
    ConditionConfig(Kind type, double value) : kind(type), parameter(value) {}
    ConditionConfig(Kind type, std::filesystem::path profile) : kind(type), parameter(std::move(profile)) {}
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

/// @brief Reads TOML on rank 0, broadcasts the text and parses one field on every rank.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, std::string_view field,
                                        mpi::CommView comm);

} // namespace krona::boundary::scalar