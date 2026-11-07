#include <krona/mesh/generation/config.hpp>

#include <krona/toml_config/file.hpp>
#include <krona/toml_config/toml.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace krona::mesh::generation {
namespace {

/// @brief Reads a fixed-size list of nonnegative integer indices or counts.
template <std::size_t N>
[[nodiscard]] std::array<std::size_t, N> read_indices(const toml_config::TomlReader &reader,
                                                      const toml::array &entries, std::string_view key) 
{
    if (entries.size() != N)
        reader.error("'" + std::string{key} + "' must contain " + std::to_string(N) + " integers");

    std::array<std::size_t, N> values{};
    for (std::size_t i = 0; i < N; ++i) {
        const auto *integer = entries[i].as_integer();
        if (!integer || !std::in_range<std::size_t>(integer->get()))
            reader.error("'" + std::string{key} + "[" + std::to_string(i) +
                         "]' must be a nonnegative integer representable by size_t");
        values[i] = static_cast<std::size_t>(integer->get());
    }
    return values;
}

/// @brief Reads and scales one Cartesian macro vertex.
[[nodiscard]] Vec3 read_vertex(const toml::node &node, std::size_t index, double scale,
                               const toml_config::TomlReader &reader) 
{
    const auto *coordinates = node.as_array();
    if (!coordinates || coordinates->size() != 3)
        reader.error("vertices[" + std::to_string(index) + "] must contain three numbers");

    std::array<double, 3> values{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto &component = (*coordinates)[axis];
        const auto value = component.value<double>();
        if (!component.is_number() || !value || !std::isfinite(*value))
            reader.error("vertices[" + std::to_string(index) + "] must contain finite numbers");
        values[axis] = *value * scale;
    }
    return {values[0], values[1], values[2]};
}

/// @brief Decodes mesh metadata and physical macro coordinates.
void read_mesh(const toml::table &root, GeneratorConfig &result) 
{
    const toml_config::TomlReader reader{"Mesh generator [mesh]"};
    const auto &mesh = reader.table(root, "mesh", true);
    reader.check_keys(mesh, {"name", "scale", "vertices"});
    result.name = reader.string(mesh, "name", result.name);
    result.scale = reader.number(mesh, "scale", result.scale);
    if (!(result.scale > 0)) reader.error("scale must be positive");

    const auto &vertices = reader.array(mesh, "vertices");
    result.vertices.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i)
        result.vertices.push_back(read_vertex(vertices[i], i, result.scale, reader));
}

/// @brief Decodes a block's corner IDs, cell counts and optional grading.
[[nodiscard]] MacroBlock read_block(const toml::node &node, std::size_t index) 
{
    const toml_config::TomlReader reader{"Mesh generator blocks[" + std::to_string(index) + "]"};
    const auto *table = node.as_table();
    if (!table) reader.error("entry must be a table");
    reader.check_keys(*table, {"name", "vertices", "cells", "grading_type", "grading"});

    MacroBlock block;
    block.name = reader.string(*table, "name", "block_" + std::to_string(index));
    block.vertices = read_indices<8>(reader, reader.array(*table, "vertices"), "vertices");
    block.cells = read_indices<3>(reader, reader.array(*table, "cells"), "cells");
    block.grading = reader.vector(*table, "grading", block.grading);

    // Missing grading types keep the block's uniform defaults.
    if (table->contains("grading_type")) {
        const auto &types = reader.array(*table, "grading_type");
        if (types.size() != 3) reader.error("grading_type must contain three strings");
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto *type = types[axis].as_string();
            if (!type) reader.error("grading_type must contain three strings");
            block.grading_type[axis] = parse_grading_type(type->get());
        }
    }
    return block;
}

/// @brief Decodes one named patch and its macro-face vertex lists.
[[nodiscard]] BoundaryPatch read_patch(const toml::node &node, std::size_t index) 
{
    const toml_config::TomlReader reader{"Mesh generator boundaries[" + std::to_string(index) + "]"};
    const auto *table = node.as_table();
    if (!table) reader.error("entry must be a table");
    reader.check_keys(*table, {"name", "faces"});

    BoundaryPatch patch;
    patch.name = reader.string(*table, "name", "patch_" + std::to_string(index));
    const auto &faces = reader.array(*table, "faces");
    patch.faces.reserve(faces.size());
    for (const auto &node_face : faces) {
        const auto *vertices = node_face.as_array();
        if (!vertices) reader.error("each face must be an array of four vertex indices");
        patch.faces.push_back({read_indices<4>(reader, *vertices, "faces")});
    }
    return patch;
}

/// @brief Decodes the three sections and validates their combined values.
[[nodiscard]] GeneratorConfig decode(const toml::table &root) 
{
    const toml_config::TomlReader reader{"Mesh generator"};
    reader.check_keys(root, {"mesh", "blocks", "boundaries"});

    GeneratorConfig result;
    read_mesh(root, result);

    const auto &blocks = reader.array(root, "blocks");
    result.blocks.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
        result.blocks.push_back(read_block(blocks[i], i));

    const auto &patches = reader.array(root, "boundaries", false);
    result.boundaries.reserve(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
        result.boundaries.push_back(read_patch(patches[i], i));

    validate_generator_config(result);
    return result;
}

} // namespace

GeneratorConfig parse_generator_config(const std::string &path, mpi::CommView comm) noexcept 
{
    return mpi::guarded([&] {
        // Only the master accesses the file; every rank decodes identical text.
        const auto text = toml_config::read_broadcast(path, comm);
        const toml_config::TomlReader reader{"Mesh generator"};
        return decode(reader.parse_text(text, path));
    });
}

} // namespace krona::mesh::generation
