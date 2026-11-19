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

/// @brief Condition kinds accepted in the scalar TOML configuration.
enum class Kind {
    FixedValue,     ///< Prescribed scalar value at the face.
    NormalGradient, ///< Prescribed outward normal derivative. ZERO_GRADIENT uses this kind with zero.
    FixedFlux       ///< Prescribed outward flux per unit area.
};

/// @brief Stores the condition kind and its parameter before the configuration is bound to a mesh.
struct ConditionConfig {
    /// @brief A numeric constant or the path to a CSV profile.
    using Parameter = std::variant<double, std::filesystem::path>;

    Kind kind = Kind::FixedValue; ///< Determines whether the parameter is a value, derivative or flux.
    Parameter parameter = 0.0;   ///< Data supplied for the selected kind.

    /// @brief Starts with a prescribed face value of zero.
    ConditionConfig() = default;

    /// @brief Builds a condition with a constant parameter.
    /// @param type Meaning of the prescribed number.
    /// @param value Constant applied across the patch.
    ConditionConfig(Kind type, double value) : kind(type), parameter(value) {}

    /// @brief Builds a condition whose parameter will be loaded from a CSV profile.
    /// @param type Meaning of the values in the profile.
    /// @param profile Path to the CSV file to load when binding the configuration.
    ConditionConfig(Kind type, std::filesystem::path profile) : kind(type), parameter(std::move(profile)) {}
};

/// @brief Binds one labelled patch to a condition. The ID and name must match the mesh catalog.
struct PatchConfig {
    std::int64_t patch_id = 0;  ///< Numeric patch ID in the mesh catalog.
    std::string name;          ///< Exact patch name used to verify the ID.
    ConditionConfig condition; ///< Condition for this patch and scalar field.
};

/// @brief Conditions for one scalar field. The manager checks coverage when binding the configuration.
struct Configuration {
    std::string field;                       ///< Scalar name used to select its TOML tables.
    std::vector<PatchConfig> patches;         ///< One entry per labelled mesh patch.
    std::optional<ConditionConfig> unlabelled; ///< Condition for boundary faces without a label.
};

/// @brief Parses one scalar field's conditions from TOML text on the calling rank.
/// @param text Contents of the configuration file.
/// @param field Name to select under each patch's scalar table.
[[nodiscard]] Configuration parse_config(std::string_view text, std::string_view field);

/// @brief Reads TOML on rank 0, broadcasts the text and parses one field on every rank.
/// @param file Path opened by rank 0 of comm.
/// @param field Name to select under each patch's scalar table.
/// @param comm Communicator whose ranks all take part in the read.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, std::string_view field,
                                        mpi::CommView comm);

} // namespace krona::boundary::scalar
