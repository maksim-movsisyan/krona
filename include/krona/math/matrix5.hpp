#pragma once

#include <Kokkos_Array.hpp>
#include <cstddef>
#include <type_traits>

namespace krona::math {

/// @brief Inline row-major 5 by 5 matrix with zero-initialized entries.
/// @tparam Real Scalar used to store each entry.
template<class Real>
struct Matrix5 {
    using value_type = Real;
    Kokkos::Array<Real, 25> values{};

    KOKKOS_INLINE_FUNCTION constexpr Real& operator()(std::size_t row, std::size_t column) noexcept
    { return values[5 * row + column]; }

    KOKKOS_INLINE_FUNCTION constexpr Real operator()(std::size_t row, std::size_t column) const noexcept
    { return values[5 * row + column]; }
};

/// @brief Multiplies two 5 by 5 matrices in the common precision of their scalars.
template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Matrix5<A>& a, const Matrix5<B>& b) noexcept {
    using R = std::common_type_t<A, B>;
    Matrix5<R> result;
    for (std::size_t row = 0; row < 5; ++row)
        for (std::size_t column = 0; column < 5; ++column)
            for (std::size_t k = 0; k < 5; ++k)
                result(row, column) += static_cast<R>(a(row, k)) * static_cast<R>(b(k, column));
    return result;
}

} // namespace krona::math
