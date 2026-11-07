#pragma once

#include <krona/numerics/gradient/detail/data.hpp>

namespace krona::numerics::gradient::detail {

template<bool LeastSquares, class Space, class Coefficients, class T, std::size_t N, class Real>
struct CellKernel {
    Metrics<Space> metrics;
    Coefficients coefficients;
    Batch<T, N, Space> fields;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        Real cell_vol_inv = Real{1};
        if constexpr (!LeastSquares) {
            cell_vol_inv = Real{1} / static_cast<Real>(metrics.mesh.geometry.cell_volume(i));
        }
        Real gx[N]{}, gy[N]{}, gz[N]{}, self[N];
        for (std::size_t c = 0; c < N; ++c) self[c] = static_cast<Real>(fields.input[c](i));
        const auto& conn = metrics.connectivity;
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i + 1); ++k) {
            const auto face = conn.cell_faces(k);
            const std::size_t f = static_cast<std::size_t>(face);
            const auto sample = metrics.sample(cell, face);
            Real x, y, z;
            if constexpr (LeastSquares) {
                x = static_cast<Real>(coefficients(k, 0));
                y = static_cast<Real>(coefficients(k, 1));
                z = static_cast<Real>(coefficients(k, 2));
            } else {
                const bool owner = metrics.mesh.connectivity.face_owner(f) == cell;
                const Real alpha = face >= metrics.mesh.sizes.n_inner_faces
                    ? Real{1} : (owner ? Real{1} - metrics.template owner_weight<Real>(f) : metrics.template owner_weight<Real>(f));
                const Real factor = (owner ? alpha : -alpha) * static_cast<Real>(metrics.mesh.geometry.face_area(f)) * cell_vol_inv;
                x = factor * static_cast<Real>(metrics.mesh.geometry.face_normal_x(f));
                y = factor * static_cast<Real>(metrics.mesh.geometry.face_normal_y(f));
                z = factor * static_cast<Real>(metrics.mesh.geometry.face_normal_z(f));
            }
            for (std::size_t c = 0; c < N; ++c) {
                const Real delta = static_cast<Real>(fields.input[c](sample)) - self[c];
                gx[c] += x*delta; gy[c] += y*delta; gz[c] += z*delta;
            }
        }
        for (std::size_t c = 0; c < N; ++c) {
            fields.x[c](i) = static_cast<T>(gx[c]);
            fields.y[c](i) = static_cast<T>(gy[c]);
            fields.z[c](i) = static_cast<T>(gz[c]);
        }
    }
};

template<class T, std::size_t N, class Space>
struct ClearKernel {
    Batch<T, N, Space> fields;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        for (std::size_t c = 0; c < N; ++c) fields.x[c](i) = fields.y[c](i) = fields.z[c](i) = T{};
    }
};

template<class Space, class T, std::size_t N, class Real>
struct FaceKernel {
    Metrics<Space> metrics;
    Batch<T, N, Space> fields;
    CellRange range;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = metrics.mesh.connectivity.face_owner(f);
        const auto neigh = metrics.mesh.connectivity.face_neigh(f);
        const bool boundary = face >= metrics.mesh.sizes.n_inner_faces;

        const bool write_owner = owner >= range.begin && owner < range.end;
        const bool write_neigh = !boundary && neigh >= range.begin && neigh < range.end;

        if (!write_owner && !write_neigh) return;

        const std::size_t i = static_cast<std::size_t>(owner);
        const auto j = metrics.sample(owner, face);

        const Real w = boundary ? Real{0} : metrics.template owner_weight<Real>(f);
        const Real alpha = boundary ? Real{1} : Real{1} - w;

        // Read volumes only for owned cells whose gradients are updated.
        const Real owner_factor = write_owner ? alpha / static_cast<Real>(metrics.mesh.geometry.cell_volume(i)) : Real{0};
        const Real neigh_factor = write_neigh ? w / static_cast<Real>(metrics.mesh.geometry.cell_volume(j)) : Real{0};

        const Real area = static_cast<Real>(metrics.mesh.geometry.face_area(f));
        const Real sx = area * static_cast<Real>(metrics.mesh.geometry.face_normal_x(f));
        const Real sy = area * static_cast<Real>(metrics.mesh.geometry.face_normal_y(f));
        const Real sz = area * static_cast<Real>(metrics.mesh.geometry.face_normal_z(f));

        for (std::size_t c = 0; c < N; ++c) {
            const Real delta = static_cast<Real>(fields.input[c](j)) - static_cast<Real>(fields.input[c](i));

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
