#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace krona::mesh::generation::hex {

/// @brief Local HEXA_8 corners in xi, eta, zeta coordinates.
inline constexpr std::array<std::array<std::size_t, 3>, 8> corners = {
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};

/// @brief Edges directed along increasing xi, eta and zeta, four per axis.
inline constexpr std::array<std::pair<std::size_t, std::size_t>, 12> edges = {
    {{0, 1}, {3, 2}, {4, 5}, {7, 6}, {0, 3}, {1, 2}, {4, 7}, {5, 6}, {0, 4}, {1, 5}, {2, 6}, {3, 7}}};

/// @brief Face corners in local (u,v) order: (0,0), (1,0), (1,1), (0,1).
inline constexpr std::array<std::array<std::size_t, 4>, 6> faces = {
    {{0, 1, 2, 3}, {0, 1, 5, 4}, {1, 2, 6, 5}, {3, 2, 6, 7}, {0, 3, 7, 4}, {4, 5, 6, 7}}};

/// @brief Outward winding for bottom, front, right, back, left and top.
inline constexpr std::array<std::array<std::size_t, 4>, 6> outward_faces = {
    {{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}}};

/// @brief Maps tangential axes and a fixed face side to block axes.
struct FaceLayout {
    std::size_t u;
    std::size_t v;
    std::size_t normal;
    bool upper;
};

/// @brief Axis layouts in the same order as faces and outward_faces.
inline constexpr std::array<FaceLayout, 6> layouts = {{{0, 1, 2, false},
                                                       {0, 2, 1, false},
                                                       {1, 2, 0, true},
                                                       {0, 2, 1, true},
                                                       {1, 2, 0, false},
                                                       {0, 1, 2, true}}};

/// @brief Identifies a face independently of its winding and starting corner.
using FaceKey = std::array<std::size_t, 4>;

/// @brief Sorts vertex IDs for lookup; perimeter is checked separately.
[[nodiscard]] inline FaceKey face_key(FaceKey vertices) noexcept {
    std::sort(vertices.begin(), vertices.end());
    return vertices;
} 

/// @brief Checks the same quad perimeter in either traversal direction.
[[nodiscard]] inline bool same_perimeter(const FaceKey &a, const FaceKey &b) noexcept {
    for (std::size_t shift = 0; shift < 4; ++shift) {
        bool forward = true;
        bool backward = true;
        for (std::size_t i = 0; i < 4; ++i) {
            forward = forward && a[i] == b[(shift + i) % 4];
            backward = backward && a[i] == b[(shift + 4 - i) % 4];
        }
        if (forward || backward)
            return true;
    }
    return false;
}

/// @brief Returns a block face in parametric order; face must be in [0,6).
[[nodiscard]] inline FaceKey face_vertices(const std::array<std::size_t, 8> &vertices,
                                           std::size_t face) noexcept {
    const auto &local = faces[face];
    return {vertices[local[0]], vertices[local[1]], vertices[local[2]], vertices[local[3]]};
}

/// @brief Returns the boundary cell at valid tangential indices u and v.
[[nodiscard]] inline std::array<std::size_t, 3> boundary_cell(const std::array<std::size_t, 3> &cells,
                                                              std::size_t face, std::size_t u,
                                                              std::size_t v) noexcept {
    const auto &layout = layouts[face];
    std::array<std::size_t, 3> index{};
    index[layout.u] = u;
    index[layout.v] = v;
    index[layout.normal] = layout.upper ? cells[layout.normal] - 1 : 0;
    return index;
}

} // namespace krona::mesh::generation::hex
