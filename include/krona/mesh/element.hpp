#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace krona::mesh {

// Linear elements; node order follows CGNS SIDS.
/// @brief Identifies supported linear volume, boundary and point elements.
enum class CellType : std::uint8_t { TET, PYRA, PRISM, HEXA, TRI, QUAD, BAR, NODE };
/// @brief Maps each element type to its fixed node count.
inline constexpr std::array<int, 8> nodes_per_type{4, 5, 6, 8, 3, 4, 2, 1};
/// @brief Maps each element type to its topological dimension.
inline constexpr std::array<int, 8> dimension_per_type{3, 3, 3, 3, 2, 2, 1, 0};
/// @brief Maps each element type to its number of boundary facets.
inline constexpr std::array<int, 8> faces_per_type{4, 5, 5, 6, 3, 4, 2, 0};
/// @brief Stores stable diagnostic names in CellType order.
inline constexpr std::array<std::string_view, 8> type_names{"TET", "PYRA", "PRISM", "HEXA",
                                                            "TRI", "QUAD", "BAR",   "NODE"};

/// @brief Returns the fixed node count of a supported linear element.
[[nodiscard]] constexpr int nodes_per_element(CellType type) noexcept {
    return nodes_per_type[static_cast<std::size_t>(type)];
}
/// @brief Returns an element's topological dimension.
[[nodiscard]] constexpr int dimension(CellType type) noexcept {
    return dimension_per_type[static_cast<std::size_t>(type)];
}
/// @brief Returns the short internal name of an element type.
[[nodiscard]] constexpr std::string_view cell_type_name(CellType type) noexcept {
    return type_names[static_cast<std::size_t>(type)];
}

// Outward facet ordering for a positively oriented reference element.
/// @brief Stores the node count of every supported local facet.
inline constexpr int face_nodes[8][6] = {{3, 3, 3, 3, 0, 0}, {4, 3, 3, 3, 3, 0}, {3, 4, 4, 4, 3, 0},
                                         {4, 4, 4, 4, 4, 4}, {2, 2, 2, 0, 0, 0}, {2, 2, 2, 2, 0, 0},
                                         {1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
/// @brief Stores outward local-node loops for positively oriented reference elements.
inline constexpr int face_table[8][6][4] = {
    {{1, 2, 3, -1}, {0, 3, 2, -1}, {0, 1, 3, -1}, {0, 2, 1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
    {{0, 3, 2, 1}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}, {-1, -1, -1, -1}},
    {{0, 2, 1, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}, {3, 4, 5, -1}, {-1, -1, -1, -1}},
    {{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}},
    {{0, 1, -1, -1}, {1, 2, -1, -1}, {2, 0, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
    {{0, 1, -1, -1}, {1, 2, -1, -1}, {2, 3, -1, -1}, {3, 0, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
    {{0, -1, -1, -1},
     {1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1}},
    {{-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1}}};
/// @brief Reverses an element orientation while retaining its supported node layout.
inline constexpr int orientation_flip[8][8] = {
    {0, 2, 1, 3, -1, -1, -1, -1},   {0, 3, 2, 1, 4, -1, -1, -1},    {0, 2, 1, 3, 5, 4, -1, -1},
    {0, 3, 2, 1, 4, 7, 6, 5},       {0, 2, 1, -1, -1, -1, -1, -1},  {0, 3, 2, 1, -1, -1, -1, -1},
    {1, 0, -1, -1, -1, -1, -1, -1}, {0, -1, -1, -1, -1, -1, -1, -1}};

} // namespace krona::mesh
