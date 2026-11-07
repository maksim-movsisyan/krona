#pragma once

#include <krona/execution/types.hpp>
#include <krona/mesh/types.hpp>
#include <krona/mesh/element.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace krona::mesh::device {

/// @brief Local counts; face intervals are valid only when faces_grouped is true.
struct MeshSizes {
    LocalIndex n_own = 0, n_cells = 0;
    LocalIndex n_nodes_own = 0, n_nodes = 0;
    LocalIndex n_faces = 0, n_internal_faces = 0, n_inner_faces = 0;
    bool faces_grouped = false;
};

/// @brief Independently selectable array groups; sizes are always present.
enum class MeshData : std::uint64_t {
    None                = 0,
    FaceCells           = std::uint64_t{1} << 0,
    CellNodes           = std::uint64_t{1} << 1,
    FaceNodes           = std::uint64_t{1} << 2,
    NodeCoordinates     = std::uint64_t{1} << 3,
    CellGeometry        = std::uint64_t{1} << 4,
    FaceGeometry        = std::uint64_t{1} << 5,
    Halo                = std::uint64_t{1} << 6,
    Connectivity        = FaceCells | CellNodes | FaceNodes,
    Geometry            = CellGeometry | FaceGeometry,
    FiniteVolume        = FaceCells | Geometry | Halo,
    All                 = Connectivity | NodeCoordinates | Geometry | Halo
};

[[nodiscard]] constexpr MeshData operator|(MeshData a, MeshData b) noexcept
{
    return static_cast<MeshData>(static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b));
}

namespace detail {

// Owning buffers are writable; kernel views are const and do not own memory.
template<class T, class Space, bool Owning>
using MeshArray = Kokkos::View<std::conditional_t<Owning, T, const T>*, Space, Kokkos::MemoryTraits<Owning ? 0 : Kokkos::Unmanaged>>;

template<class Space, bool Owning>
struct ConnectivityArrays {
    MeshArray<LocalIndex, Space, Owning> face_owner;
    MeshArray<LocalIndex, Space, Owning> face_neigh;
    MeshArray<CellType, Space, Owning> cell_type;
    MeshArray<std::size_t, Space, Owning> cell_nodes_offsets;
    MeshArray<LocalIndex, Space, Owning> cell_nodes;
    MeshArray<CellType, Space, Owning> face_type;
    MeshArray<std::uint8_t, Space, Owning> face_local_index;
    MeshArray<std::size_t, Space, Owning> face_nodes_offsets;
    MeshArray<LocalIndex, Space, Owning> face_nodes;
};

template<class Space, bool Owning>
struct GeometryArrays {
    MeshArray<GeometryReal, Space, Owning> node_x, node_y, node_z;
    MeshArray<GeometryReal, Space, Owning> cell_centroid_x, cell_centroid_y, cell_centroid_z;
    MeshArray<GeometryReal, Space, Owning> cell_volume;
    MeshArray<GeometryReal, Space, Owning> face_centroid_x, face_centroid_y, face_centroid_z;
    MeshArray<GeometryReal, Space, Owning> face_normal_x, face_normal_y, face_normal_z;
    MeshArray<GeometryReal, Space, Owning> face_area;
};

// Kernel gather/scatter maps. MPI peers and offsets live in the host HaloPlanView.
template<class Space, bool Owning>
struct HaloArrays {
    MeshArray<LocalIndex, Space, Owning> send_owned_local;
    MeshArray<LocalIndex, Space, Owning> recv_ghost_local;
};

template<class Space, bool Owning>
struct MeshArrays {
    using memory_space = Space;

    MeshSizes sizes;
    MeshData active_mask = MeshData::None;
    ConnectivityArrays<Space, Owning> connectivity;
    GeometryArrays<Space, Owning> geometry;
    HaloArrays<Space, Owning> halo;

    KOKKOS_INLINE_FUNCTION bool has(MeshData flags) const noexcept
    {
        const std::uint64_t bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
    }
};

} // namespace detail

/// @brief Read-only, non-owning descriptors passed by value to Kokkos kernels.
template<class Space = execution::MemorySpace>
using MeshView = detail::MeshArrays<Space, false>; 

} // namespace krona::mesh::device
