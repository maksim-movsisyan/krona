#pragma once

#include <krona/math/inverse_spd.hpp>
#include <krona/numerics/gradient/detail/cell_accumulator.hpp>

namespace krona::numerics::gradient::detail {

/// @brief Precomputes three weights per entry of the owned-cell-to-face table.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Coefficients Writable Kokkos view indexed by CSR slot and axis.
/// @tparam Real Scalar type for solving the local 3D fit. The operator uses double.
template <class Space, class Coefficients, class Real> 
struct PrepareLeastSquares {
    StencilData<Space> stencil;
    Coefficients coefficients;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, std::size_t& bad) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        const auto& conn = stencil.connectivity;
        const auto begin = conn.cell_faces_offsets(i), end = conn.cell_faces_offsets(i + 1);

        // Find a length scale for this stencil so squaring a very large or small displacement is safer.
        Real scale = 0;
        for (auto k = begin; k < end; ++k) {
            Real d[3];
            stencil.displacement(cell, conn.cell_faces(k), d);
            for (int axis = 0; axis < 3; ++axis) {
                if (!Kokkos::isfinite(d[axis])) {
                    ++bad;
                    return;
                }
                const Real a = Kokkos::abs(d[axis]);
                if (a > scale)
                    scale = a;
            }
        }
        if (!(scale > 0)) {
            ++bad;
            return;
        }

        // Each normalized d contributes d*d^T / |d|^2 to the 3x3 normal matrix.
        Real a[9]{};
        for (auto k = begin; k < end; ++k) {
            Real d[3];
            stencil.displacement(cell, conn.cell_faces(k), d);
            for (auto& value : d)
                value /= scale;
            const Real r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (!(r2 > 0)) {
                ++bad;
                return;
            }
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    a[3 * row + col] += d[row] * d[col] / r2;
        }

        // A usable stencil must span all three directions. Cholesky rejects weak or missing pivots.
        Real inverse[9];
        if (!math::inverse_spd(a, inverse, static_cast<Real>(stencil.options.pivot_tolerance))) {
            ++bad;
            return;
        }

        // Precompute the vector that multiplies each sample difference during apply.
        // Divide by scale to recover weights in physical coordinates. Revisit the CSR row
        // so the kernel needs no temporary array sized by the number of faces.
        for (auto k = begin; k < end; ++k) {
            Real d[3];
            stencil.displacement(cell, conn.cell_faces(k), d);
            for (auto& value : d)
                value /= scale;
            const Real r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            for (int axis = 0; axis < 3; ++axis) {
                const Real value =
                    (inverse[3 * axis] * d[0] + inverse[3 * axis + 1] * d[1] + inverse[3 * axis + 2] * d[2]) /
                    r2 / scale;
                using C = typename Coefficients::non_const_value_type;
                const C stored = static_cast<C>(value);
                if (!Kokkos::isfinite(stored)) {
                    ++bad;
                    return;
                }
                coefficients(k, axis) = stored;
            }
        }
    }
};

/// @brief Applies the prepared least-squares fit to one owned cell.
/// @tparam Space Kokkos memory space of the mesh arrays and component descriptors.
/// @tparam Coefficients Kokkos view of prepared weights indexed by CSR slot and axis.
/// @tparam T Scalar type stored in the input and output fields.
/// @tparam N Number of input components processed together.
/// @tparam Real Scalar type used for contributions and accumulation.
template <class Space, class Coefficients, class T, std::size_t N, class Real> 
struct LeastSquaresFaceKernel {
    StencilData<Space> stencil;
    Coefficients coefficients;
    Batch<T, N, Space> fields;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        const auto& conn = stencil.connectivity;
        CellAccumulator<T, N, Space, Real> sum(fields, i);

        // Keep the same CSR order as preparation. Each slot already has the vector
        // that maps phi_sample - phi_cell to its contribution to the gradient.
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i + 1); ++k) {
            const auto sample = stencil.sample(cell, conn.cell_faces(k));
            const Real x = static_cast<Real>(coefficients(k, 0));
            const Real y = static_cast<Real>(coefficients(k, 1));
            const Real z = static_cast<Real>(coefficients(k, 2));
            sum.add(fields, sample, x, y, z);
        }

        sum.store(fields, i);
    }
};

} // namespace krona::numerics::gradient::detail
