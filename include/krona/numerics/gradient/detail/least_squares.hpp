#pragma once

#include <krona/numerics/gradient/detail/data.hpp>

namespace krona::numerics::gradient::detail {

// Relative Cholesky pivot checks reject singular 3-D stencils instead of returning zero gradients.
template<class Real>
KOKKOS_INLINE_FUNCTION bool inverse_spd(const Real a[9], Real inverse[9], Real tolerance)
{
    Real l[9]{};
    const Real threshold = tolerance * (a[0] + a[4] + a[8]);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col <= row; ++col) {
            Real v = a[3 * row + col];
            for (int k = 0; k < col; ++k) v -= l[3 * row + k] * l[3 * col + k];
            if (row == col) {
                if (!Kokkos::isfinite(v) || v <= threshold) return false;
                l[3 * row + col] = Kokkos::sqrt(v);
            } else l[3 * row + col] = v / l[3 * col + col];
        }
    }
    for (int col = 0; col < 3; ++col) {
        Real y[3]{}, x[3]{};
        for (int row = 0; row < 3; ++row) {
            Real v = row == col ? Real{1} : Real{0};
            for (int k = 0; k < row; ++k) v -= l[3 * row + k] * y[k];
            y[row] = v / l[3 * row + row];
        }
        for (int row = 2; row >= 0; --row) {
            Real v = y[row];
            for (int k = row + 1; k < 3; ++k) v -= l[3 * k + row] * x[k];
            x[row] = v / l[3 * row + row];
            inverse[3 * row + col] = x[row];
        }
    }
    return true;
}

template<class Space, class Coefficients, class Real>
struct PrepareLeastSquares {
    Metrics<Space> metrics;
    Coefficients coefficients;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, std::size_t& bad) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        const auto& conn = metrics.connectivity;
        const auto begin = conn.cell_faces_offsets(i), end = conn.cell_faces_offsets(i + 1);
        Real scale = 0;
        for (auto k = begin; k < end; ++k) {
            Real d[3]; metrics.displacement(cell, conn.cell_faces(k), d);
            for (int axis = 0; axis < 3; ++axis) {
                if (!Kokkos::isfinite(d[axis])) { ++bad; return; }
                const Real a = Kokkos::abs(d[axis]);
                if (a > scale) scale = a;
            }
        }
        if (!(scale > 0)) { ++bad; return; }
        Real a[9]{};
        for (auto k = begin; k < end; ++k) {
            Real d[3]; metrics.displacement(cell, conn.cell_faces(k), d);
            for (auto& value : d) value /= scale;
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            if (!(r2 > 0)) { ++bad; return; }
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) a[3*row+col] += d[row]*d[col]/r2;
        }
        Real inverse[9];
        if (!inverse_spd(a, inverse, static_cast<Real>(metrics.options.pivot_tolerance))) { ++bad; return; }
        // A second pass avoids fixed-size neighbour buffers and their degree limits.
        for (auto k = begin; k < end; ++k) {
            Real d[3]; metrics.displacement(cell, conn.cell_faces(k), d);
            for (auto& value : d) value /= scale;
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            for (int axis = 0; axis < 3; ++axis) {
                const Real value = (inverse[3*axis]*d[0] + inverse[3*axis+1]*d[1] + inverse[3*axis+2]*d[2]) / r2 / scale;
                using C = typename Coefficients::non_const_value_type;
                const C stored = static_cast<C>(value);
                if (!Kokkos::isfinite(stored)) { ++bad; return; }
                coefficients(k, axis) = stored;
            }
        }
    }
};

} // namespace krona::numerics::gradient::detail
