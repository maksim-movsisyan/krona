#pragma once

#include <krona/numerics/gradient/detail/data.hpp>

namespace krona::numerics::gradient::detail {

/// @brief Clears one selected cell so faces can atomically add to its gradient.
/// @tparam T Scalar type stored in the output fields.
/// @tparam N Number of input components processed together.
/// @tparam Space Kokkos memory space of the component descriptors.
template <class T, std::size_t N, class Space> 
struct ClearKernel {
    Batch<T, N, Space> fields;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        for (std::size_t c = 0; c < N; ++c)
            fields.x[c](i) = fields.y[c](i) = fields.z[c](i) = T{};
    }
};

/// @brief One iteration adds a single face's contribution to the selected cells it touches.
/// @tparam Space Kokkos memory space of the mesh arrays and component descriptors.
/// @tparam T Scalar type of the fields and atomic sums.
/// @tparam N Number of input components processed together.
/// @tparam Real Scalar type used to compute each contribution before converting it to T.
template <class Space, class T, std::size_t N, class Real> 
struct GreenGaussFaceKernel {
    StencilData<Space> stencil;
    Batch<T, N, Space> fields;
    CellRange range;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = stencil.mesh.connectivity.face_owner(f);
        const auto neigh = stencil.mesh.connectivity.face_neigh(f);
        const bool boundary = face >= stencil.mesh.sizes.n_inner_faces;

        // A ghost can supply a sample, but only cells in the requested owned range get a result.
        const bool write_owner = owner >= range.begin && owner < range.end;
        const bool write_neigh = !boundary && neigh >= range.begin && neigh < range.end;

        if (!write_owner && !write_neigh)
            return;

        const std::size_t i = static_cast<std::size_t>(owner);
        const auto j = stencil.sample(owner, face);

        // Interior phi_face uses the owner weight w. On a boundary, use the face sample directly.
        const Real w = boundary ? Real{0} : stencil.template owner_weight<Real>(f);
        const Real alpha = boundary ? Real{1} : Real{1} - w;

        // Read volumes only for owned cells whose gradients are updated.
        const Real owner_factor =
            write_owner ? alpha / static_cast<Real>(stencil.mesh.geometry.cell_volume(i)) : Real{0};
        const Real neigh_factor =
            write_neigh ? w / static_cast<Real>(stencil.mesh.geometry.cell_volume(j)) : Real{0};

        const Real area = static_cast<Real>(stencil.mesh.geometry.face_area(f));
        const Real sx = area * static_cast<Real>(stencil.mesh.geometry.face_normal_x(f));
        const Real sy = area * static_cast<Real>(stencil.mesh.geometry.face_normal_y(f));
        const Real sz = area * static_cast<Real>(stencil.mesh.geometry.face_normal_z(f));

        // Both sides use phi_sample - phi_owner. On the neighbour side, reversing the
        // normal and using phi_owner - phi_neighbour cancel the two sign changes.
        for (std::size_t c = 0; c < N; ++c) {
            const Real delta = static_cast<Real>(fields.input[c](j)) - static_cast<Real>(fields.input[c](i));

            // Other faces may update the same cell in parallel, so these writes must be atomic.
            if (write_owner) {
                const Real v = owner_factor * delta;
                Kokkos::atomic_add(&fields.x[c](i), static_cast<T>(sx * v));
                Kokkos::atomic_add(&fields.y[c](i), static_cast<T>(sy * v));
                Kokkos::atomic_add(&fields.z[c](i), static_cast<T>(sz * v));
            }

            if (write_neigh) {
                const Real v = neigh_factor * delta;
                Kokkos::atomic_add(&fields.x[c](j), static_cast<T>(sx * v));
                Kokkos::atomic_add(&fields.y[c](j), static_cast<T>(sy * v));
                Kokkos::atomic_add(&fields.z[c](j), static_cast<T>(sz * v));
            }
        }
    }
};

} // namespace krona::numerics::gradient::detail
