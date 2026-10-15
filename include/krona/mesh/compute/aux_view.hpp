#pragma once

#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/mesh/aux_connectivity.hpp>
#include <krona/mesh/aux_geometry.hpp>

namespace krona::mesh::compute {

namespace detail {

template<class Space, bool Owning>
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

    KOKKOS_INLINE_FUNCTION bool has(AuxConnType flags) const noexcept
    {
        const std::uint64_t bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
    }

    KOKKOS_INLINE_FUNCTION bool has_cell_faces() const noexcept { return has(AuxConnType::CellFaces); }
    KOKKOS_INLINE_FUNCTION bool has_cell_cells_face() const noexcept { return has(AuxConnType::CellCellsByFace); }
    KOKKOS_INLINE_FUNCTION bool has_node_cells() const noexcept { return has(AuxConnType::NodeCells); }
    KOKKOS_INLINE_FUNCTION bool has_node_faces() const noexcept { return has(AuxConnType::NodeFaces); }
    KOKKOS_INLINE_FUNCTION bool has_cell_cells_node() const noexcept { return has(AuxConnType::CellCellsByNode); }
};

template<class Space, bool Owning>
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

    KOKKOS_INLINE_FUNCTION bool has(AuxGeomType flags) const noexcept
    {
        const std::uint64_t bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
    }
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist() const noexcept { return has(AuxGeomType::FaceCellDistance); }
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_vector() const noexcept { return has(AuxGeomType::FaceCellDistanceVector); }
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_inv() const noexcept { return has(AuxGeomType::FaceCellDistanceInv); }
    KOKKOS_INLINE_FUNCTION bool has_face_interp_weights() const noexcept { return has(AuxGeomType::FaceInterpolationWeights); }
    KOKKOS_INLINE_FUNCTION bool has_face_non_ortho() const noexcept { return has(AuxGeomType::FaceNonOrthogonalityVector); }
    KOKKOS_INLINE_FUNCTION bool has_face_normal_dist() const noexcept { return has(AuxGeomType::FaceNormalDistance); }
    KOKKOS_INLINE_FUNCTION bool has_face_normal_dist_inv() const noexcept { return has(AuxGeomType::FaceNormalDistanceInv); }
    KOKKOS_INLINE_FUNCTION bool has_face_area_over_normal_dist() const noexcept { return has(AuxGeomType::FaceAreaOverNormalDistance); }
    KOKKOS_INLINE_FUNCTION bool has_wall_dist() const noexcept { return has(AuxGeomType::WallDistance); }
};

} // namespace detail

// Optional data stays outside MeshView: kernels capture only needed descriptors. 
template<class Space = execution::MemorySpace>
using AuxConnectivityView = detail::AuxConnectivityArrays<Space, false>;

template<class Space = execution::MemorySpace>
using AuxGeometryView = detail::AuxGeometryArrays<Space, false>;

} // namespace krona::mesh::compute
