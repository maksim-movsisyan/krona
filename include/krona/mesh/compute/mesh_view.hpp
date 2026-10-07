#pragma once

#include <krona/execution/types.hpp>
#include <krona/mesh/raw_mesh.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace krona::mesh::compute {
/// @brief Groups copied or borrowed from MeshPart; counts are always available.
enum class MeshData : std::uint32_t {
    None = 0, Topology = 1U << 0, Geometry = 1U << 1, Halo = 1U << 2,
    CellConnectivity = 1U << 3, FaceConnectivity = 1U << 4, NodeCoordinates = 1U << 5,
    Core = 0x07U, Connectivity = 0x38U, All = 0x3FU
};

[[nodiscard]] constexpr MeshData operator|(MeshData a, MeshData b) noexcept
{
    return static_cast<MeshData>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

namespace detail {

// Owning buffers are writable; kernel views are const and do not own memory.
template<class T, class Space, bool Owning>
using MeshArray = Kokkos::View<std::conditional_t<Owning, T, const T>*, Space,
    Kokkos::MemoryTraits<Owning ? 0 : Kokkos::Unmanaged>>;

template<class Space, bool Owning>
struct TopologyArrays {
    LocalIndex n_own = 0, n_cells = 0, n_nodes_own = 0, n_nodes = 0, n_faces = 0;
    LocalIndex n_internal_faces = 0, n_inner_faces = 0;

    MeshArray<LocalIndex, Space, Owning> face_owner;
    MeshArray<LocalIndex, Space, Owning> face_neigh;
    MeshArray<PatchId, Space, Owning> face_patch;
    MeshArray<std::size_t, Space, Owning> patch_face_offsets;
    MeshArray<LocalIndex, Space, Owning> patch_faces;
};

template<class Space, bool Owning>
struct GeometryArrays {
    MeshArray<GeometryReal, Space, Owning> cell_centroid_x;
    MeshArray<GeometryReal, Space, Owning> cell_centroid_y;
    MeshArray<GeometryReal, Space, Owning> cell_centroid_z;
    MeshArray<GeometryReal, Space, Owning> cell_volume;
    MeshArray<GeometryReal, Space, Owning> face_centroid_x;
    MeshArray<GeometryReal, Space, Owning> face_centroid_y;
    MeshArray<GeometryReal, Space, Owning> face_centroid_z;
    MeshArray<GeometryReal, Space, Owning> face_normal_x;
    MeshArray<GeometryReal, Space, Owning> face_normal_y;
    MeshArray<GeometryReal, Space, Owning> face_normal_z;
    MeshArray<GeometryReal, Space, Owning> face_area;
};

template<class Space, bool Owning>
struct HaloArrays {
    MeshArray<LocalIndex, Space, Owning> send_owned_local;
    MeshArray<LocalIndex, Space, Owning> recv_ghost_local;
};

template<class Space, bool Owning>
struct ConnectivityArrays {
    MeshArray<CellType, Space, Owning> cell_type;
    MeshArray<std::size_t, Space, Owning> cell_nodes_offsets;
    MeshArray<LocalIndex, Space, Owning> cell_nodes;
    MeshArray<GeometryReal, Space, Owning> node_x;
    MeshArray<GeometryReal, Space, Owning> node_y;
    MeshArray<GeometryReal, Space, Owning> node_z;
    MeshArray<CellType, Space, Owning> face_type;
    MeshArray<std::uint8_t, Space, Owning> face_local_index;
    MeshArray<std::size_t, Space, Owning> face_nodes_offsets;
    MeshArray<LocalIndex, Space, Owning> face_nodes;
};

template<class Space, bool Owning>
struct MeshArrays {
    MeshData active_mask = MeshData::None;

    KOKKOS_INLINE_FUNCTION bool has(MeshData flags) const noexcept
    {
        const auto bits = static_cast<std::uint32_t>(flags);
        return (static_cast<std::uint32_t>(active_mask) & bits) == bits;
    }

    TopologyArrays<Space, Owning> topology;
    GeometryArrays<Space, Owning> geometry;
    HaloArrays<Space, Owning> halo;
    ConnectivityArrays<Space, Owning> connectivity;
};

} // namespace detail

/// @brief Read-only, non-owning descriptors passed by value to Kokkos kernels.
template<class Space = execution::MemorySpace>
using MeshView = detail::MeshArrays<Space, false>;

} // namespace krona::mesh::compute
