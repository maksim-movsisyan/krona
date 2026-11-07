#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <krona/mesh/generation/vec3.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mpi/error.hpp>

namespace krona::mesh::generation {

/// @brief Selects the normalized node distribution along one block axis.
enum class GradingType : std::uint8_t {
    UNIFORM = 0,
    GEOMETRIC = 1,
    TANH_START = 2,
    TANH_END = 3,
    TANH_BOTH = 4
};

/// @brief Returns the TOML spelling of a grading type, or "unknown".
[[nodiscard]] inline std::string_view to_string(GradingType type) noexcept {
    switch (type) {
    case GradingType::UNIFORM: return "uniform";
    case GradingType::GEOMETRIC: return "geometric";
    case GradingType::TANH_START: return "tanh_start";
    case GradingType::TANH_END: return "tanh_end";
    case GradingType::TANH_BOTH: return "tanh_both";
    }
    return "unknown";
}

/// @brief Parses a grading name; unknown spellings terminate the MPI job.
[[nodiscard]] inline GradingType parse_grading_type(std::string_view str) noexcept {
    if (str == "uniform") return GradingType::UNIFORM;
    if (str == "geometric") return GradingType::GEOMETRIC;
    if (str == "tanh_start") return GradingType::TANH_START;
    if (str == "tanh_end") return GradingType::TANH_END;
    if (str == "tanh_both") return GradingType::TANH_BOTH;
    mpi::fatal("Unknown grading type (expected uniform, geometric, tanh_start, tanh_end or tanh_both)");
}

/// @brief Defines a trilinear HEXA_8 block and its three node distributions.
struct MacroBlock {
    std::string name;
    std::array<std::size_t, 8> vertices{};
    std::array<std::size_t, 3> cells{};                       // Nx, Ny, Nz
    std::array<GradingType, 3> grading_type{};                // Along xi, eta, zeta
    std::array<double, 3> grading{1.0, 1.0, 1.0}; // Strengths / ratios
};

/// @brief Selects a macro face by shared vertex IDs in perimeter order.
/// Either traversal is accepted; the writer emits outward-oriented quads.
struct MacroFace {
    std::array<std::size_t, 4> vertices{};
};

/// @brief Groups complete outer macro faces under one CGNS patch name.
struct BoundaryPatch {
    std::string name;
    std::vector<MacroFace> faces;
};

/// @brief Stores physical coordinates, block subdivisions and named faces.
struct GeneratorConfig {
    std::string name{"Mesh"};
    double scale{1.0}; // Applied while decoding TOML; retained as input metadata.
    std::vector<Vec3> vertices;
    std::vector<MacroBlock> blocks;
    std::vector<BoundaryPatch> boundaries;
};

/// @brief Reads on rank 0, broadcasts text and decodes it on all ranks.
[[nodiscard]] GeneratorConfig parse_generator_config(const std::string &path, mpi::CommView comm) noexcept;

/// @brief Validates physical geometry and configuration values locally.
/// C++ callers supply physical coordinates; no scaling is applied here.
void validate_generator_config(const GeneratorConfig &config) noexcept;

} // namespace krona::mesh::generation