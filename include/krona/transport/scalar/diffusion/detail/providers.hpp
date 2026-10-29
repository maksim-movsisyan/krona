#pragma once

#include <krona/transport/diffusion/isotropic.hpp>
#include <krona/transport/scalar/source.hpp>
#include <krona/transport/scalar/diffusion/plan.hpp>
#include <krona/transport/scalar/diffusion/correction.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Canonical owner-to-neighbour orientation, also on the neighbour's row.
template<class Real>
struct InteriorFaceContext {
    LocalIndex face = 0, owner = 0, neighbour = -1;
    Real x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0;
    Real distance = 0, area = 0, time = 0;
    Real owner_weight = -1; // Available when FaceInterpolationWeights is attached.
};

/// @brief Kernel captures only arrays used by row assembly; descriptors borrow existing storage.
template<class Space>
struct RowData {
    using Index = mesh::device::detail::MeshArray<LocalIndex, Space, false>;
    using Metric = mesh::device::detail::MeshArray<GeometryReal, Space, false>;
    using Offset = mesh::device::detail::MeshArray<std::size_t, Space, false>;
    LocalIndex n_inner_faces = 0;
    Index face_owner, face_neigh, faces;
    Offset offsets;
    Metric cell_volume, face_x, face_y, face_z, normal_x, normal_y, normal_z, area;
    Metric distance, area_over_distance, owner_weight;
    Metric non_ortho_x, non_ortho_y, non_ortho_z;
    bool has_weights = false;
};

template<class Space>
auto row_data(mesh::device::MeshView<Space> mesh, mesh::device::AuxConnectivityView<Space> connectivity,
              mesh::device::AuxGeometryView<Space> geometry)
{
    const auto& g = mesh.geometry;
    return RowData<Space>{mesh.sizes.n_inner_faces, mesh.connectivity.face_owner, mesh.connectivity.face_neigh,
        connectivity.cell_faces, connectivity.cell_faces_offsets, g.cell_volume,
        g.face_centroid_x, g.face_centroid_y, g.face_centroid_z, g.face_normal_x, g.face_normal_y, g.face_normal_z,
        g.face_area, geometry.face_normal_dist, geometry.face_area_over_normal_dist, geometry.face_interp_weight,
        geometry.face_non_ortho_x, geometry.face_non_ortho_y, geometry.face_non_ortho_z,
        geometry.has_face_interp_weights()};
}

template<class Exec, class Real, class Correction>
void validate_correction(const DiffusionPlan<Exec, Real>& plan, const Correction& correction)
{
    if constexpr (corrected<Correction>) {
        const auto mesh = plan.mesh_view();
        const auto geometry = plan.geometry_view();
        mpi::require(geometry.has_face_non_ortho(),
            "Scalar diffusion: correction requires attached FaceNonOrthogonalityVector");
        mpi::require(mesh.sizes.n_inner_faces == 0 || geometry.has_face_interp_weights(),
            "Scalar diffusion: correction requires attached FaceInterpolationWeights");
        validate_gradients<Exec>(correction.gradients, static_cast<std::size_t>(mesh.sizes.n_cells));
    }
}

template<class Real, class Space>
KOKKOS_INLINE_FUNCTION auto interior_context(const RowData<Space>& data, LocalIndex face, Real time)
{
    const std::size_t f = static_cast<std::size_t>(face);
    return InteriorFaceContext<Real>{face, data.face_owner(f), data.face_neigh(f),
        static_cast<Real>(data.face_x(f)), static_cast<Real>(data.face_y(f)), static_cast<Real>(data.face_z(f)),
        static_cast<Real>(data.normal_x(f)), static_cast<Real>(data.normal_y(f)), static_cast<Real>(data.normal_z(f)),
        static_cast<Real>(data.distance(f)), static_cast<Real>(data.area(f)), time,
        data.has_weights ? static_cast<Real>(data.owner_weight(f)) : Real{-1}};
}

template<class Plan, class Coefficient> void validate_coefficient(const Plan&, const Coefficient&) {}
template<class Exec, class Real, class View>
void validate_coefficient(const DiffusionPlan<Exec, Real>& plan, FaceCoefficient<View> c)
{ scalar::detail::validate_view<Exec>(c.values, static_cast<std::size_t>(plan.mesh_view().sizes.n_faces)); }
template<class Exec, class Real, class View>
void validate_coefficient(const DiffusionPlan<Exec, Real>& plan, OwnerCellCoefficient<View> c)
{
    mpi::require(plan.mesh_view().sizes.n_inner_faces == 0,
        "Scalar diffusion: OwnerCellCoefficient is exterior-only; use InterpolatedCellCoefficient for interior faces");
    scalar::detail::validate_view<Exec>(c.values, static_cast<std::size_t>(plan.mesh_view().sizes.n_own));
}
template<class Exec, class Real, class View, class Interpolation>
void validate_coefficient(const DiffusionPlan<Exec, Real>& plan, InterpolatedCellCoefficient<View, Interpolation> c)
{
    scalar::detail::validate_view<Exec>(c.values, static_cast<std::size_t>(plan.mesh_view().sizes.n_cells));
    mpi::require(plan.mesh_view().sizes.n_inner_faces == 0 || plan.geometry_view().has_face_interp_weights(),
        "Scalar diffusion: cell coefficients require attached FaceInterpolationWeights");
}

template<class Exec, class Real, class Boundary>
void validate_boundary(const DiffusionPlan<Exec, Real>& plan, const Boundary& boundary)
{
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