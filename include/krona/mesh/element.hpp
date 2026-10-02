#pragma once

#include "krona/config.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

namespace krona::mesh
{

// Linear elements; node order follows CGNS SIDS.
enum class CellType : std::uint8_t
{
    TET,
    PYRA,
    PRISM,
    HEXA,
    TRI,
    QUAD,
    BAR,
    NODE
};
inline constexpr std::array<int, 8> nodes_per_type{4, 5, 6, 8, 3, 4, 2, 1};
inline constexpr std::array<int, 8> dimension_per_type{3, 3, 3, 3, 2, 2, 1, 0};
inline constexpr std::array<int, 8> faces_per_type{4, 5, 5, 6, 3, 4, 2, 0};
inline constexpr std::array<std::string_view, 8> type_names{"TET", "PYRA", "PRISM", "HEXA",
                                                            "TRI", "QUAD", "BAR",   "NODE"};

[[nodiscard]] constexpr int nodes_per_element(CellType type) noexcept
{
    return nodes_per_type[static_cast<std::size_t>(type)];
}
[[nodiscard]] constexpr int dimension(CellType type) noexcept
{
    return dimension_per_type[static_cast<std::size_t>(type)];
}
[[nodiscard]] constexpr std::string_view cell_type_name(CellType type) noexcept
{
    return type_names[static_cast<std::size_t>(type)];
}

// Outward facet ordering for a positively oriented reference element.
inline constexpr int face_nodes[8][6] = {{3, 3, 3, 3, 0, 0}, {4, 3, 3, 3, 3, 0}, {3, 4, 4, 4, 3, 0},
                                         {4, 4, 4, 4, 4, 4}, {2, 2, 2, 0, 0, 0}, {2, 2, 2, 2, 0, 0},
                                         {1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
inline constexpr int face_table[8][6][4] = {
    {{1, 2, 3, -1},
     {0, 3, 2, -1},
     {0, 1, 3, -1},
     {0, 2, 1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1}},
    {{0, 3, 2, 1}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}, {-1, -1, -1, -1}},
    {{0, 2, 1, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}, {3, 4, 5, -1}, {-1, -1, -1, -1}},
    {{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}},
    {{0, 1, -1, -1},
     {1, 2, -1, -1},
     {2, 0, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1}},
    {{0, 1, -1, -1},
     {1, 2, -1, -1},
     {2, 3, -1, -1},
     {3, 0, -1, -1},
     {-1, -1, -1, -1},
     {-1, -1, -1, -1}},
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
inline constexpr int orientation_flip[8][8] = {
    {0, 2, 1, 3, -1, -1, -1, -1},   {0, 3, 2, 1, 4, -1, -1, -1},    {0, 2, 1, 3, 5, 4, -1, -1},
    {0, 3, 2, 1, 4, 7, 6, 5},       {0, 2, 1, -1, -1, -1, -1, -1},  {0, 3, 2, 1, -1, -1, -1, -1},
    {1, 0, -1, -1, -1, -1, -1, -1}, {0, -1, -1, -1, -1, -1, -1, -1}};

// A matching key only; retain oriented connectivity separately.
struct FaceKey
{
    std::array<GlobalIndex, 4> nodes{-1, -1, -1, -1};
    auto operator<=>(const FaceKey &) const = default;
};
[[nodiscard]] inline FaceKey make_face_key(std::span<const GlobalIndex> nodes)
{
    if (nodes.empty() || nodes.size() > 4)
    {
        throw std::invalid_argument("Invalid facet arity");
    }
    FaceKey key;
    std::copy(nodes.begin(), nodes.end(), key.nodes.begin());
    std::sort(key.nodes.begin(), key.nodes.begin() + static_cast<std::ptrdiff_t>(nodes.size()));
    return key;
}

} // namespace krona::mesh
