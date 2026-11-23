#pragma once

#include <krona/mpi/error.hpp>

namespace krona::transport::detail {

/// @brief Requires a boundary manager to borrow the same geometry and patch ranges as the operator.
template<class Plan, class Boundary>
void validate_face_boundary(const Plan& plan, const Boundary& boundary) {
    const auto b = boundary.geometry_view();
    const auto m = plan.mesh_view();
    const auto& g = m.geometry;
    mpi::require(b.n_cells == m.sizes.n_cells && b.n_inner_faces == m.sizes.n_inner_faces &&
        b.face_owner.data() == m.connectivity.face_owner.data() &&
        b.cell_x.data() == g.cell_centroid_x.data() && b.cell_y.data() == g.cell_centroid_y.data() &&
        b.cell_z.data() == g.cell_centroid_z.data() &&
        b.face_x.data() == g.face_centroid_x.data() && b.face_y.data() == g.face_centroid_y.data() &&
        b.face_z.data() == g.face_centroid_z.data() &&
        b.normal_x.data() == g.face_normal_x.data() && b.normal_y.data() == g.face_normal_y.data() &&
        b.normal_z.data() == g.face_normal_z.data() && b.face_area.data() == g.face_area.data(),
        "Transport: boundary manager must use the same compute mesh");
    const auto& catalog = plan.catalog();
    const auto patches = catalog.patches();
    boundary.for_each_patch([&](auto range, const auto&) {
        const auto id = range.patch_id;
        mpi::require(id == -1 || (id >= 0 && static_cast<std::size_t>(id) < patches.size()),
                     "Transport: unknown boundary patch");
        const auto expected = id == -1 ? catalog.unlabelled() : patches[static_cast<std::size_t>(id)].range;
        mpi::require(expected.begin == range.begin && expected.end == range.end,
                     "Transport: boundary catalog differs from plan");
    });
}

} // namespace krona::transport::detail
