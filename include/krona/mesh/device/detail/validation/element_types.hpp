#pragma once

#include <krona/mesh/device/mesh_view.hpp>

#include <cstddef>

namespace krona::mesh::device::validation {

/// @brief Checks that each cell's node count matches its element type.
/// @tparam Space Kokkos memory space of the mesh view.
/// @tparam Rules Object with a device-callable cell_nodes(type) that returns the
/// expected count, for example 8 for a hexahedron, or zero for an unsupported type.
/// Check CellNodes shapes and CSR before running over all local cells.
template <class Space, class Rules>
struct ValidateCellTypes {
    MeshView<Space> mesh;
    Rules rules;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t c, std::size_t& bad) const {
        const auto& a = mesh.connectivity;
        const auto b = a.cell_nodes_offsets(c), e = a.cell_nodes_offsets(c + 1);
        const auto expected = rules.cell_nodes(a.cell_type(c));
        if (expected == 0 || b > e || e > a.cell_nodes.extent(0) || e - b != expected)
            ++bad;
    }
};

/// @brief Checks the face node count and the face number within its owner cell.
/// @tparam Space Kokkos memory space of the mesh view.
/// @tparam Rules Object with device-callable face_nodes(type) and cell_faces(type)
/// methods. Each returns the expected count or zero for an unsupported type.
/// Check FaceNodes, FaceCells and CellNodes before running over all local faces.
template <class Space, class Rules>
struct ValidateFaceTypes {
    MeshView<Space> mesh;
    Rules rules;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t f, std::size_t& bad) const {
        const auto& a = mesh.connectivity;
        const auto b = a.face_nodes_offsets(f), e = a.face_nodes_offsets(f + 1);
        const auto expected = rules.face_nodes(a.face_type(f));
        const auto owner = a.face_owner(f);
        if (expected == 0 || b > e || e > a.face_nodes.extent(0) || e - b != expected || owner < 0 ||
            owner >= mesh.sizes.n_own) {
            ++bad;
            return;
        }

        // The local face number must be within the owner's faces.
        if (a.face_local_index(f) >= rules.cell_faces(a.cell_type(static_cast<std::size_t>(owner))))
            ++bad;
    }
};

} // namespace krona::mesh::device::validation
