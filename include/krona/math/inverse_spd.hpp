#pragma once

#include <krona/math/symmetric_tensor3.hpp>
#include <Kokkos_NumericTraits.hpp>

#include <type_traits>

namespace krona::math {
namespace detail {

// Arithmetic body copied from the gradient module's 3-D Cholesky inverse.
template<class Real>
KOKKOS_INLINE_FUNCTION bool inverse_spd_cholesky(const Real a[9], Real inverse[9], Real tolerance) {
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

} // namespace detail

/// @brief Checked row-major 3x3 SPD inverse. On failure output is unchanged; in-place use is supported.
/// Requires exact symmetry and finite 0<tolerance<1; each Cholesky pivot must exceed tolerance*trace(a).
template<class Real>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool inverse_spd(const Real a[9], Real inverse[9],
    std::type_identity_t<Real> tolerance = Real{64}*Kokkos::epsilon_v<Real>) noexcept {
    static_assert(std::is_floating_point_v<Real>);
    if (!a || !inverse || !Kokkos::isfinite(tolerance) || !(tolerance > Real{0} && tolerance < Real{1})) return false;
    for (int i = 0; i < 9; ++i) if (!Kokkos::isfinite(a[i])) return false;
    if (a[1] != a[3] || a[2] != a[6] || a[5] != a[7]) return false;
    Real result[9];
    if (!detail::inverse_spd_cholesky(a, result, tolerance)) return false;
    for (const auto value : result) if (!Kokkos::isfinite(value)) return false;
    for (int i = 0; i < 9; ++i) inverse[i] = result[i];
    return true;
}

template<class Real>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool inverse_spd(const Matrix3<Real>& a, Matrix3<Real>& inverse,
    std::type_identity_t<Real> tolerance = Real{64}*Kokkos::epsilon_v<Real>) noexcept {
    Real input[9], result[9];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) input[3*i+j] = a.data[i][j];
    if (!inverse_spd(input, result, tolerance)) return false;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) inverse.data[i][j] = result[3*i+j];
    return true;
}

template<class Real>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool inverse_spd(SymmetricTensor3<Real> a, SymmetricTensor3<Real>& inverse,
    std::type_identity_t<Real> tolerance = Real{64}*Kokkos::epsilon_v<Real>) noexcept {
    Matrix3<Real> result;
    if (!inverse_spd(to_matrix(a), result, tolerance)) return false;
    inverse = {result.data[0][0], result.data[1][1], result.data[2][2],
               result.data[0][1], result.data[0][2], result.data[1][2]};
    return true;
}

} // namespace krona::math
