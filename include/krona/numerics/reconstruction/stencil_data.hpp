#pragma once

#include <krona/numerics/reconstruction/geometry.hpp>
#include <krona/mesh/device/aux_view.hpp>
#include <krona/mpi/error.hpp>

namespace krona::numerics::reconstruction {

/// @brief Borrows mesh topology and geometry for face evaluation and limiter preparation.
/// @tparam Space Memory space holding the mesh arrays.
/// @tparam Real Arithmetic type used to form the side geometry.
/// @details Build after mesh validation and keep the source arrays alive while kernels use this view.
template <class Space, class Real = ComputeReal>
class StencilData {
    template <class T>
    using Array = Kokkos::View<const T*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Array<LocalIndex> owners_, neighbours_, faces_;
    Array<std::size_t> offsets_;
    Array<GeometryReal> cx_, cy_, cz_, fx_, fy_, fz_, nx_, ny_, nz_;
    LocalIndex owned_ = 0, cells_ = 0, inner_ = 0, face_count_ = 0;

    KOKKOS_INLINE_FUNCTION Vector<Real> centre(std::size_t i) const {
        return {static_cast<Real>(cx_(i)), static_cast<Real>(cy_(i)), static_cast<Real>(cz_(i))};
    }

    KOKKOS_INLINE_FUNCTION Vector<Real> point(std::size_t f) const {
        return {static_cast<Real>(fx_(f)), static_cast<Real>(fy_(f)), static_cast<Real>(fz_(f))};
    }

    KOKKOS_INLINE_FUNCTION Vector<Real> normal(std::size_t f) const {
        return {static_cast<Real>(nx_(f)), static_cast<Real>(ny_(f)), static_cast<Real>(nz_(f))};
    }

    KOKKOS_INLINE_FUNCTION std::size_t boundary_sample(std::size_t f) const {
        return static_cast<std::size_t>(cells_) + (f - static_cast<std::size_t>(inner_));
    }

public:
    using memory_space = Space;

    /// @brief Checks the available array groups and borrows their memory.
    /// @param m Mesh with grouped faces, cell centroids, face centroids and normals.
    /// @param a CellFaces adjacency for the same owned cell ordering.
    StencilData(mesh::device::MeshView<Space> m, mesh::device::AuxConnectivityView<Space> a)
        : owners_(m.connectivity.face_owner), neighbours_(m.connectivity.face_neigh), faces_(a.cell_faces),
          offsets_(a.cell_faces_offsets), cx_(m.geometry.cell_centroid_x), cy_(m.geometry.cell_centroid_y),
          cz_(m.geometry.cell_centroid_z), fx_(m.geometry.face_centroid_x), fy_(m.geometry.face_centroid_y),
          fz_(m.geometry.face_centroid_z), nx_(m.geometry.face_normal_x), ny_(m.geometry.face_normal_y),
          nz_(m.geometry.face_normal_z), owned_(m.sizes.n_own), cells_(m.sizes.n_cells),
          inner_(m.sizes.n_inner_faces), face_count_(m.sizes.n_faces) {
        using mesh::device::MeshData;

        mpi::require(m.sizes.faces_grouped &&
                     m.has(MeshData::FaceCells | MeshData::CellGeometry | MeshData::FaceGeometry) &&
                     a.has_cell_faces() && owned_ >= 0 && cells_ >= owned_ && inner_ >= 0 &&
                     face_count_ >= inner_ && a.n_cells_own == owned_ && a.n_faces == face_count_ &&
                     offsets_.extent(0) == static_cast<std::size_t>(owned_) + 1,
                     "Reconstruction: incompatible mesh stencil descriptors");

        const std::size_t cells = static_cast<std::size_t>(cells_);
        const std::size_t faces = static_cast<std::size_t>(face_count_);

        mpi::require(cx_.extent(0) >= cells && cy_.extent(0) >= cells && cz_.extent(0) >= cells &&
                     owners_.extent(0) >= faces && neighbours_.extent(0) >= faces &&
                     fx_.extent(0) >= faces && fy_.extent(0) >= faces && fz_.extent(0) >= faces &&
                     nx_.extent(0) >= faces && ny_.extent(0) >= faces && nz_.extent(0) >= faces,
                     "Reconstruction: missing mesh stencil arrays");
    }

    /// @brief Number of owned cells that can receive a limiter factor.
    KOKKOS_INLINE_FUNCTION std::size_t size() const {
        return static_cast<std::size_t>(owned_);
    }

    /// @brief Minimum value extent covering ghosts and all physical-face samples.
    KOKKOS_INLINE_FUNCTION std::size_t required_input_size() const {
        return static_cast<std::size_t>(cells_) + static_cast<std::size_t>(face_count_ - inner_);
    }

    /// @brief Number of incident faces of an owned cell.
    KOKKOS_INLINE_FUNCTION std::size_t size(std::size_t cell) const {
        return offsets_(cell + 1) - offsets_(cell);
    }

    /// @brief Local face index for the kth face incident to the cell.
    KOKKOS_INLINE_FUNCTION std::size_t incident_face(std::size_t cell, std::size_t k) const {
        return static_cast<std::size_t>(faces_(offsets_(cell) + k));
    }

    /// @brief Reports whether the face is on the physical boundary.
    KOKKOS_INLINE_FUNCTION bool is_boundary(std::size_t f) const {
        return f >= static_cast<std::size_t>(inner_);
    }

    /// @brief Checks the orientation of an incident face relative to a cell.
    KOKKOS_INLINE_FUNCTION bool is_owner(std::size_t f, std::size_t cell) const {
        return cell == static_cast<std::size_t>(owners_(f));
    }

    /// @brief Returns a neighbour or physical-face value index for the bound search.
    KOKKOS_INLINE_FUNCTION std::size_t sample(std::size_t cell, std::size_t k) const {
        const auto f = incident_face(cell, k);
        if (is_boundary(f))
            return boundary_sample(f);
        return static_cast<std::size_t>(is_owner(f, cell) ? neighbours_(f) : owners_(f));
    }

    /// @brief Builds the two traces' geometry at the centroid of an inner face.
    /// @tparam Needed Geometry a formula asks this adapter to load.
    template <Geometry Needed = Geometry::Point>
    KOKKOS_INLINE_FUNCTION FaceGeometry<Real> face(std::size_t f) const {
        static_assert(Needed != Geometry::Explicit, "Supply a stencil that owns the explicit side data");
        KOKKOS_ASSERT(!is_boundary(f));
        const std::size_t owner = static_cast<std::size_t>(owners_(f));
        const std::size_t neighbour = static_cast<std::size_t>(neighbours_(f));
        if constexpr (Needed == Geometry::Cell) {
            return {{owner, neighbour}, {neighbour, owner}};
        } else {
            const auto target = point(f);
            const auto left = target - centre(owner);
            const auto right = target - centre(neighbour);
            if constexpr (Needed == Geometry::Line) {
                const auto n = normal(f);
                return face_geometry(owner, neighbour, left, right, dot(left, n) / dot(left - right, n));
            } else {
                return {{owner, neighbour, left}, {neighbour, owner, right}};
            }
        }
    }

    /// @brief Builds only the requested side of an inner face.
    /// @tparam Needed Geometry a formula asks this adapter to load.
    template <Geometry Needed>
    KOKKOS_INLINE_FUNCTION CellSide<Real> inner_side(std::size_t f, bool from_owner) const {
        if constexpr (Needed == Geometry::Line) {
            // Use the same canonical fraction for both sides and in limiter preparation.
            const auto both = face<Needed>(f);
            return from_owner ? both.left : both.right;
        } else {
            static_assert(Needed != Geometry::Explicit, "Supply a stencil that owns the explicit side data");
            KOKKOS_ASSERT(!is_boundary(f));
            const std::size_t cell = static_cast<std::size_t>(from_owner ? owners_(f) : neighbours_(f));
            if constexpr (Needed == Geometry::Point)
                return {cell, 0, point(f) - centre(cell)};
            else
                return {cell};
        }
    }

    /// @brief Builds the side of a physical face based in its adjacent owned cell.
    /// @tparam Needed Geometry a formula asks this adapter to load.
    template <Geometry Needed = Geometry::Point>
    KOKKOS_INLINE_FUNCTION CellSide<Real> boundary_side(std::size_t f) const {
        static_assert(Needed != Geometry::Explicit, "Supply a stencil that owns the explicit side data");
        KOKKOS_ASSERT(is_boundary(f) && f < static_cast<std::size_t>(face_count_));
        const auto cell = static_cast<std::size_t>(owners_(f));
        if constexpr (Needed == Geometry::Cell)
            return {cell};
        const auto offset = point(f) - centre(cell);
        return {cell, boundary_sample(f), offset, offset, Real{1}};
    }
};

} // namespace krona::numerics::reconstruction
