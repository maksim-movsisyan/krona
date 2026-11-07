#pragma once

#include <krona/mesh/device/mesh_view.hpp>
#include <krona/mesh/aux_types.hpp>

namespace krona::mesh::device {

namespace detail {

/// @brief Groups optional CSR descriptors for local cell and node stencils.
template <class Space, bool Owning> 
struct AuxConnectivityArrays {
    using memory_space = Space;

    AuxConnType active_mask = AuxConnType::None;
    LocalIndex n_faces = 0, n_cells_own = 0, n_nodes_own = 0;

    MeshArray<std::size_t, Space, Owning> cell_faces_offsets;
    MeshArray<LocalIndex, Space, Owning> cell_faces;
    MeshArray<std::size_t, Space, Owning> cell_cells_face_offsets;
    MeshArray<LocalIndex, Space, Owning> cell_cells_face;
    MeshArray<std::size_t, Space, Owning> node_cells_offsets;
    MeshArray<LocalIndex, Space, Owning> node_cells;
    MeshArray<std::size_t, Space, Owning> node_faces_offsets;
    MeshArray<LocalIndex, Space, Owning> node_faces;
    MeshArray<std::size_t, Space, Owning> cell_cells_node_offsets;
    MeshArray<LocalIndex, Space, Owning> cell_cells_node;

    /// @brief Reports whether every requested data group is ready.
    KOKKOS_INLINE_FUNCTION bool has(AuxConnType flags) const noexcept {
        const std::uint64_t bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
    }

    /// @brief Reports availability of owned-cell-to-face rows.
    KOKKOS_INLINE_FUNCTION bool has_cell_faces() const noexcept { return has(AuxConnType::CellFaces); }
    /// @brief Reports availability of owned-cell face-neighbour rows.
    KOKKOS_INLINE_FUNCTION bool has_cell_cells_face() const noexcept {
        return has(AuxConnType::CellCellsByFace);
    }
    /// @brief Reports availability of owned-used-node-to-cell rows.
    KOKKOS_INLINE_FUNCTION bool has_node_cells() const noexcept { return has(AuxConnType::NodeCells); }
    /// @brief Reports availability of owned-used-node-to-face rows.
    KOKKOS_INLINE_FUNCTION bool has_node_faces() const noexcept { return has(AuxConnType::NodeFaces); }
    /// @brief Reports availability of unique local vertex-neighbour rows.
    KOKKOS_INLINE_FUNCTION bool has_cell_cells_node() const noexcept {
        return has(AuxConnType::CellCellsByNode);
    }
};

/// @brief Groups optional face metrics and cell wall-distance descriptors.
template <class Space, bool Owning> 
struct AuxGeometryArrays {
    using memory_space = Space;

    AuxGeomType active_mask = AuxGeomType::None;
    LocalIndex n_faces = 0, n_cells = 0;

    MeshArray<GeometryReal, Space, Owning> face_cell_dist;
    MeshArray<GeometryReal, Space, Owning> face_cell_dist_x;
    MeshArray<GeometryReal, Space, Owning> face_cell_dist_y;
    MeshArray<GeometryReal, Space, Owning> face_cell_dist_z;
    MeshArray<GeometryReal, Space, Owning> face_cell_dist_inv;
    MeshArray<GeometryReal, Space, Owning> face_interp_weight;
    MeshArray<GeometryReal, Space, Owning> face_non_ortho_x;
    MeshArray<GeometryReal, Space, Owning> face_non_ortho_y;
    MeshArray<GeometryReal, Space, Owning> face_non_ortho_z;
    MeshArray<GeometryReal, Space, Owning> face_normal_dist;
    MeshArray<GeometryReal, Space, Owning> face_normal_dist_inv;
    MeshArray<GeometryReal, Space, Owning> face_area_over_normal_dist;
    MeshArray<GeometryReal, Space, Owning> cell_wall_dist;

    /// @brief Reports whether every requested data group is ready.
    KOKKOS_INLINE_FUNCTION bool has(AuxGeomType flags) const noexcept {
        const std::uint64_t bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
    }
    /// @brief Reports availability of centre-separation magnitudes.
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist() const noexcept {
        return has(AuxGeomType::FaceCellDistance);
    }
    /// @brief Reports availability of centre-displacement components.
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_vector() const noexcept {
        return has(AuxGeomType::FaceCellDistanceVector);
    }
    /// @brief Reports availability of reciprocal centre separations.
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_inv() const noexcept {
        return has(AuxGeomType::FaceCellDistanceInv);
    }
    /// @brief Reports availability of owner-side interpolation weights.
    KOKKOS_INLINE_FUNCTION bool has_face_interp_weights() const noexcept {
        return has(AuxGeomType::FaceInterpolationWeights);
    }
    /// @brief Reports availability of nonorthogonal area corrections.
    KOKKOS_INLINE_FUNCTION bool has_face_non_ortho() const noexcept {
        return has(AuxGeomType::FaceNonOrthogonalityVector);
    }
    /// @brief Reports availability of normal centre separations.
    KOKKOS_INLINE_FUNCTION bool has_face_normal_dist() const noexcept {
        return has(AuxGeomType::FaceNormalDistance);
    }
    /// @brief Reports availability of reciprocal normal separations.
    KOKKOS_INLINE_FUNCTION bool has_face_normal_dist_inv() const noexcept {
        return has(AuxGeomType::FaceNormalDistanceInv);
    }
    /// @brief Reports availability of area divided by normal separation.
    KOKKOS_INLINE_FUNCTION bool has_face_area_over_normal_dist() const noexcept {
        return has(AuxGeomType::FaceAreaOverNormalDistance);
    }
    /// @brief Reports availability of the collective graph wall-distance field.
    KOKKOS_INLINE_FUNCTION bool has_wall_dist() const noexcept { return has(AuxGeomType::WallDistance); }
};

} // namespace detail

// Optional data stays outside MeshView: kernels capture only needed descriptors.
/// @brief Borrows read-only auxiliary connectivity for capture by kernels.
template <class Space = execution::MemorySpace>
using AuxConnectivityView = detail::AuxConnectivityArrays<Space, false>;

/// @brief Borrows read-only auxiliary metrics for capture by kernels.
template <class Space = execution::MemorySpace>
using AuxGeometryView = detail::AuxGeometryArrays<Space, false>;

} // namespace krona::mesh::device
