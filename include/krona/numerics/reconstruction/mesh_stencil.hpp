#pragma once

#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/mesh/device/aux_view.hpp>

namespace krona::numerics::reconstruction {

/// @brief Optional allocation-free mesh adapter. Borrows already validated grouped mesh and CellFaces arrays.
/// Bounds sample real neighbours or extended boundary-face slots. No boundary ghost cells are introduced.
template<class Space, class Real = ComputeReal>
class MeshStencil {
    template<class T> using Array = Kokkos::View<const T*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Array<LocalIndex> owners_, neighbours_, faces_;
    Array<std::size_t> offsets_;
    Array<GeometryReal> cx_, cy_, cz_, fx_, fy_, fz_, nx_, ny_, nz_;
    LocalIndex owned_ = 0, cells_ = 0, inner_ = 0, face_count_ = 0;
    KOKKOS_INLINE_FUNCTION Vector<Real> centre(std::size_t i) const
    { return {static_cast<Real>(cx_(i)), static_cast<Real>(cy_(i)), static_cast<Real>(cz_(i))}; }
    KOKKOS_INLINE_FUNCTION Vector<Real> point(std::size_t f) const
    { return {static_cast<Real>(fx_(f)), static_cast<Real>(fy_(f)), static_cast<Real>(fz_(f))}; }
    KOKKOS_INLINE_FUNCTION Vector<Real> normal(std::size_t f) const
    { return {static_cast<Real>(nx_(f)), static_cast<Real>(ny_(f)), static_cast<Real>(nz_(f))}; }
public:
    using memory_space = Space;
    MeshStencil(mesh::device::MeshView<Space> m, mesh::device::AuxConnectivityView<Space> a)
        : owners_(m.connectivity.face_owner), neighbours_(m.connectivity.face_neigh), faces_(a.cell_faces), offsets_(a.cell_faces_offsets),
          cx_(m.geometry.cell_centroid_x), cy_(m.geometry.cell_centroid_y), cz_(m.geometry.cell_centroid_z),
          fx_(m.geometry.face_centroid_x), fy_(m.geometry.face_centroid_y), fz_(m.geometry.face_centroid_z),
          nx_(m.geometry.face_normal_x), ny_(m.geometry.face_normal_y), nz_(m.geometry.face_normal_z),
          owned_(m.sizes.n_own), cells_(m.sizes.n_cells), inner_(m.sizes.n_inner_faces), face_count_(m.sizes.n_faces)
    {
        using mesh::device::MeshData;

        mpi::require(
            m.sizes.faces_grouped &&
            m.has(MeshData::FaceCells | MeshData::CellGeometry | MeshData::FaceGeometry) &&
            a.has_cell_faces() &&
            owned_ >= 0 &&
            cells_ >= owned_ &&
            inner_ >= 0 &&
            face_count_ >= inner_ &&
            a.n_cells_own == owned_ &&
            a.n_faces == face_count_ &&
            offsets_.extent(0) == static_cast<std::size_t>(owned_) + 1,
            "Reconstruction: incompatible mesh stencil descriptors");

        const auto cells = static_cast<std::size_t>(cells_);
        const auto faces = static_cast<std::size_t>(face_count_);

        mpi::require(
            cx_.extent(0) >= cells &&
            cy_.extent(0) >= cells &&
            cz_.extent(0) >= cells &&
            owners_.extent(0) >= faces &&
            neighbours_.extent(0) >= faces &&
            fx_.extent(0) >= faces &&
            fy_.extent(0) >= faces &&
            fz_.extent(0) >= faces &&
            nx_.extent(0) >= faces &&
            ny_.extent(0) >= faces &&
            nz_.extent(0) >= faces,
            "Reconstruction: missing mesh stencil arrays");
    }
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return static_cast<std::size_t>(owned_); }
    KOKKOS_INLINE_FUNCTION std::size_t required_input_size() const
    { return static_cast<std::size_t>(cells_)+static_cast<std::size_t>(face_count_-inner_); }
    KOKKOS_INLINE_FUNCTION std::size_t size(std::size_t cell) const { return offsets_(cell+1)-offsets_(cell); }
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell, std::size_t k) const
    {
        const auto f = static_cast<std::size_t>(faces_(offsets_(cell)+k));
        const auto other = owners_(f) == static_cast<LocalIndex>(cell) ? neighbours_(f) : owners_(f);
        const auto offset = point(f)-centre(cell);
        const auto span = other < 0 ? offset : centre(static_cast<std::size_t>(other))-centre(cell);
        const auto sample = other < 0 ? static_cast<std::size_t>(cells_)+f-static_cast<std::size_t>(inner_) : static_cast<std::size_t>(other);
        const auto n = normal(f);
        const Real fraction = dot(offset, n)/dot(span, n);
        return StencilEntry<Real>{sample, {cell, sample, offset, span, fraction}};
    }
    /// @brief Inner faces only. FirstOrder specialization reads endpoints without loading geometry.
    template<class Scheme = SecondOrderUpwind>
    KOKKOS_INLINE_FUNCTION auto face(std::size_t f) const
    {
        KOKKOS_ASSERT(f < static_cast<std::size_t>(inner_));
        const auto p = static_cast<std::size_t>(owners_(f)), n = static_cast<std::size_t>(neighbours_(f));
        if constexpr (Scheme::requirements == Values) return FaceGeometry<Real>{{p, n}, {n, p}};
        else {
            const auto left = point(f)-centre(p), right = point(f)-centre(n);
            Real fraction = -1;
            if constexpr ((Scheme::requirements & OtherValues) != 0) fraction = dot(left, normal(f))/dot(left-right, normal(f));
            return face_geometry(p, n, left, right, fraction);
        }
    }
    /// @brief Reconstruct only the interior trace; the application supplies the physical boundary closure.
    KOKKOS_INLINE_FUNCTION auto boundary_side(std::size_t f) const
    {
        KOKKOS_ASSERT(f >= static_cast<std::size_t>(inner_) && f < static_cast<std::size_t>(face_count_));
        const auto p = static_cast<std::size_t>(owners_(f));
        const auto offset = point(f)-centre(p);
        return CellSide<Real>{p, static_cast<std::size_t>(cells_)+f-static_cast<std::size_t>(inner_), offset, offset, Real{1}};
    }
};

} // namespace krona::numerics::reconstruction