#pragma once

#include <krona/mesh/device/detail/validation/common.hpp>
#include <krona/mesh/device/aux_view.hpp>

namespace krona::mesh::device::validation {

/// @brief Checks one cell's selected geometry, including cells in the MPI halo.
/// @tparam Data Base mesh groups. CellGeometry checks the centre and positive volume.
/// @tparam Geometry Auxiliary geometry groups. WallDistance checks the stored wall distance.
/// @tparam Space Kokkos memory space of the mesh and geometry views.
/// Run over all local cells after checking that the selected arrays have the expected sizes.
template <MeshData Data, AuxGeomType Geometry, class Space>
struct ValidateCells {
    MeshView<Space> mesh;
    AuxGeometryView<Space> geometry;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t c, std::size_t& bad) const {
        bool valid = true;
        if constexpr (has(Data, MeshData::CellGeometry)) {
            const auto& g = mesh.geometry;
            valid &= detail::positive(g.cell_volume(c)) &&
                     detail::finite3(g.cell_centroid_x(c), g.cell_centroid_y(c), g.cell_centroid_z(c));
        }

        if constexpr (has(Geometry, AuxGeomType::WallDistance)) {
            const GeometryReal d = geometry.cell_wall_dist(c);
            // A cell that cannot reach a wall has distance +infinity.
            valid &= d >= 0 && (Kokkos::isfinite(d) || d == std::numeric_limits<GeometryReal>::infinity());
        }

        if (!valid)
            ++bad;
    }
};

/// @brief Checks that all three coordinates of a node are finite.
/// @tparam Data Base mesh groups. This check runs when NodeCoordinates is selected.
/// @tparam Space Kokkos memory space of the mesh view.
template <MeshData Data, class Space>
struct ValidateNodes {
    MeshView<Space> mesh;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t n, std::size_t& bad) const {
        if constexpr (has(Data, MeshData::NodeCoordinates))
            if (!detail::finite3(mesh.geometry.node_x(n), mesh.geometry.node_y(n), mesh.geometry.node_z(n)))
                ++bad;
    }
};

} // namespace krona::mesh::device::validation
