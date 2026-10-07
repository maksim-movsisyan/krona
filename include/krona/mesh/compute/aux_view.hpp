#pragma once

#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/mesh/aux_connectivity.hpp>
#include <krona/mesh/aux_geometry.hpp>

namespace krona::mesh::compute {

/// @brief Non-owning read-only CSR row, usable on host and in Kokkos kernels.
struct IndexRow {
    const LocalIndex* data = nullptr;
    std::size_t length = 0;

    KOKKOS_INLINE_FUNCTION std::size_t size() const noexcept { return length; }
    KOKKOS_INLINE_FUNCTION bool empty() const noexcept { return length == 0; }
    KOKKOS_INLINE_FUNCTION LocalIndex operator()(std::size_t i) const noexcept
    {
        KOKKOS_ASSERT(i < length);
        return data[i];
    }
    KOKKOS_INLINE_FUNCTION LocalIndex operator[](std::size_t i) const noexcept { return (*this)(i); }
};

namespace detail {

template<class Space, bool Owning>
struct AuxConnectivityArrays {
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
        const auto bits = static_cast<std::uint32_t>(flags);
        return (static_cast<std::uint32_t>(active_mask) & bits) == bits;
    }

    KOKKOS_INLINE_FUNCTION bool has_cell_faces() const noexcept { return has(AuxConnType::CellFaces); }
    KOKKOS_INLINE_FUNCTION IndexRow cell_faces_of(LocalIndex index) const noexcept
    {
        KOKKOS_ASSERT(has_cell_faces() && index >= 0 && index < n_cells_own);
        return row(cell_faces_offsets, cell_faces, index);
    }

    KOKKOS_INLINE_FUNCTION bool has_cell_cells_face() const noexcept { return has(AuxConnType::CellCellsByFace); }
    KOKKOS_INLINE_FUNCTION IndexRow cell_neighbors_face_of(LocalIndex index) const noexcept
    {
        KOKKOS_ASSERT(has_cell_cells_face() && index >= 0 && index < n_cells_own);
        return row(cell_cells_face_offsets, cell_cells_face, index);
    }

    KOKKOS_INLINE_FUNCTION bool has_node_cells() const noexcept { return has(AuxConnType::NodeCells); }
    KOKKOS_INLINE_FUNCTION IndexRow node_cells_of(LocalIndex index) const noexcept
    {
        KOKKOS_ASSERT(has_node_cells() && index >= 0 && index < n_nodes_own);
        return row(node_cells_offsets, node_cells, index);
    }

    KOKKOS_INLINE_FUNCTION bool has_node_faces() const noexcept { return has(AuxConnType::NodeFaces); }
    KOKKOS_INLINE_FUNCTION IndexRow node_faces_of(LocalIndex index) const noexcept
    {
        KOKKOS_ASSERT(has_node_faces() && index >= 0 && index < n_nodes_own);
        return row(node_faces_offsets, node_faces, index);
    }

    KOKKOS_INLINE_FUNCTION bool has_cell_cells_node() const noexcept { return has(AuxConnType::CellCellsByNode); }
    KOKKOS_INLINE_FUNCTION IndexRow cell_neighbors_node_of(LocalIndex index) const noexcept
    {
        KOKKOS_ASSERT(has_cell_cells_node() && index >= 0 && index < n_cells_own);
        return row(cell_cells_node_offsets, cell_cells_node, index);
    }

private:
    KOKKOS_INLINE_FUNCTION static IndexRow row(
        const MeshArray<std::size_t, Space, Owning>& offsets,
        const MeshArray<LocalIndex, Space, Owning>& values, LocalIndex index) noexcept
    {
        const auto i = static_cast<std::size_t>(index);
        const auto begin = offsets(i), end = offsets(i + 1);
        // Empty rows also work when the values allocation is null.
        return {values.data() ? values.data() + begin : nullptr, end - begin};
    }
};

template<class Space, bool Owning>
struct AuxGeometryArrays {
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
    MeshArray<GeometryReal, Space, Owning> cell_wall_dist;

    KOKKOS_INLINE_FUNCTION bool has(AuxGeomType flags) const noexcept
    {
        const auto bits = static_cast<std::uint32_t>(flags);
        return (static_cast<std::uint32_t>(active_mask) & bits) == bits;
    }
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist() const noexcept { return has(AuxGeomType::FaceCellDistance); }
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_vector() const noexcept { return has(AuxGeomType::FaceCellDistanceVector); }
    KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_inv() const noexcept { return has(AuxGeomType::FaceCellDistanceInv); }
    KOKKOS_INLINE_FUNCTION bool has_face_interp_weights() const noexcept { return has(AuxGeomType::FaceInterpolationWeights); }
    KOKKOS_INLINE_FUNCTION bool has_face_non_ortho() const noexcept { return has(AuxGeomType::FaceNonOrthogonalityVector); }
    KOKKOS_INLINE_FUNCTION bool has_wall_dist() const noexcept { return has(AuxGeomType::WallDistance); }
};

} // namespace detail

// Optional data stays outside MeshView: kernels capture only needed descriptors.
template<class Space = execution::MemorySpace>
using AuxConnectivityView = detail::AuxConnectivityArrays<Space, false>;

template<class Space = execution::MemorySpace>
using AuxGeometryView = detail::AuxGeometryArrays<Space, false>;

} // namespace krona::mesh::compute
