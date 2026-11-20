#pragma once

#include <krona/transport/diffusion/isotropic.hpp>
#include <krona/transport/scalar/source/providers.hpp>
#include <krona/transport/scalar/diffusion/plan.hpp>
#include <krona/transport/scalar/diffusion/detail/correction.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Checks that a correction can read its owned and ghost gradients.
template<class Exec, class Real, class Correction>
void validate_correction(const DiffusionPlan<Exec, Real>& plan, const Correction& correction) {
    if constexpr (corrected<Correction>) {
        const auto mesh = plan.mesh_view();
        const auto cells = static_cast<std::size_t>(mesh.sizes.n_cells);
        if constexpr (requires { correction.gradients; })
            validate_gradients<Exec>(correction.gradients, cells);
        else if constexpr (requires { correction.template validate<Exec>(cells); })
            mpi::guarded([&] { correction.template validate<Exec>(cells); });
    }
}

template<class Plan, class Coefficient> void validate_coefficient(const Plan&, const Coefficient&) {}
template<class Exec, class Real, class View>
void validate_coefficient(const DiffusionPlan<Exec, Real>& plan, FaceCoefficient<View> c)
{ scalar::detail::validate_view<Exec>(c.values, static_cast<std::size_t>(plan.mesh_view().sizes.n_faces)); }
template<class Exec, class Real, class View>
void validate_coefficient(const DiffusionPlan<Exec, Real>& plan, OwnerCellCoefficient<View> c) {
    mpi::require(plan.mesh_view().sizes.n_inner_faces == 0,
        "Scalar diffusion: OwnerCellCoefficient is exterior-only; use InterpolatedCellCoefficient for interior faces");
    scalar::detail::validate_view<Exec>(c.values, static_cast<std::size_t>(plan.mesh_view().sizes.n_own));
}
template<class Exec, class Real, class View, class Interpolation>
void validate_coefficient(const DiffusionPlan<Exec, Real>& plan, InterpolatedCellCoefficient<View, Interpolation> c) {
    scalar::detail::validate_view<Exec>(c.values, static_cast<std::size_t>(plan.mesh_view().sizes.n_cells));
}

/// @brief Checks that the boundary manager borrows the same mesh geometry as the plan.
template<class Exec, class Real, class Boundary>
void validate_boundary(const DiffusionPlan<Exec, Real>& plan, const Boundary& boundary) {
    const auto m = plan.mesh_view();
    const auto b = boundary.geometry_view();
    const auto& g = m.geometry;
    mpi::require(b.n_cells == m.sizes.n_cells && b.n_inner_faces == m.sizes.n_inner_faces &&
        b.face_owner.data() == m.connectivity.face_owner.data() &&
        b.cell_x.data() == g.cell_centroid_x.data() && b.cell_y.data() == g.cell_centroid_y.data() && b.cell_z.data() == g.cell_centroid_z.data() &&
        b.face_x.data() == g.face_centroid_x.data() && b.face_y.data() == g.face_centroid_y.data() && b.face_z.data() == g.face_centroid_z.data() &&
        b.normal_x.data() == g.face_normal_x.data() && b.normal_y.data() == g.face_normal_y.data() && b.normal_z.data() == g.face_normal_z.data() &&
        b.face_area.data() == g.face_area.data(), "Scalar diffusion: boundary manager must use the same compute mesh");
}

} // namespace krona::transport::scalar::diffusion::detail
