#pragma once

#include "cli.hpp"

#include <krona/mesh/device/mesh_view.hpp>

namespace krona::apps::scalar_transport {

/// @brief Source density at a cell centroid and time. Customize the custom case here.
template<class Real>
KOKKOS_INLINE_FUNCTION Real source_density(Case problem, Real /*x*/, Real /*y*/, Real /*z*/,
                                          Real /*time*/, Real uniform) {
    if (problem == Case::custom) return uniform;
    return Real{0};
}

/// @brief Evaluates the prescribed source for one cell at the bound time.
template<class Real, class Coordinate>
struct SourceAtTime {
    Coordinate x, y, z;
    Case problem;
    Real time, uniform;

    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t cell) const {
        return source_density(problem, static_cast<Real>(x(cell)), static_cast<Real>(y(cell)),
                              static_cast<Real>(z(cell)), time, uniform);
    }
};

/// @brief Binds the source to owned cell coordinates and one step's evaluation time.
template<class Real, class Space>
[[nodiscard]] auto make_source(Case problem, mesh::device::MeshView<Space> mesh, Real time, Real uniform) {
    const auto& g = mesh.geometry;
    return SourceAtTime<Real, decltype(g.cell_centroid_x)>{
        g.cell_centroid_x, g.cell_centroid_y, g.cell_centroid_z, problem, time, uniform};
}

} // namespace krona::apps::scalar_transport