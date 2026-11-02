#pragma once

#include <krona/math/vector3.hpp>

namespace krona::math {

/// @brief Inline row-major 3x3 matrix: data[row][column]. Default initialization gives zero.
template<class Real>
struct Matrix3 {
    static_assert(std::is_floating_point_v<Real>);
    using value_type = Real;
    Real data[3][3]{};

    KOKKOS_INLINE_FUNCTION constexpr Real& operator()(std::size_t row, std::size_t col) noexcept
    { KOKKOS_ASSERT(row < 3 && col < 3); return data[row][col]; }
    KOKKOS_INLINE_FUNCTION constexpr const Real& operator()(std::size_t row, std::size_t col) const noexcept
    { KOKKOS_ASSERT(row < 3 && col < 3); return data[row][col]; }

    KOKKOS_INLINE_FUNCTION static constexpr Matrix3 identity() noexcept
    {
        Matrix3 result;
        for (std::size_t i = 0; i < 3; ++i) result.data[i][i] = Real{1};
        return result;
    }

    KOKKOS_INLINE_FUNCTION constexpr Matrix3& operator+=(const Matrix3& b) noexcept
    { for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j) data[i][j] += b.data[i][j]; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr Matrix3& operator-=(const Matrix3& b) noexcept
    { for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j) data[i][j] -= b.data[i][j]; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr Matrix3& operator*=(Real s) noexcept
    { for (auto& row : data) for (auto& value : row) value *= s; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr Matrix3& operator/=(Real s) noexcept
    { for (auto& row : data) for (auto& value : row) value /= s; return *this; }
};

template<class To, class From>
KOKKOS_INLINE_FUNCTION constexpr Matrix3<To> cast(const Matrix3<From>& a) noexcept
{
    Matrix3<To> result;
    for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j)
        result.data[i][j] = static_cast<To>(a.data[i][j]);
    return result;
}

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(const Matrix3<A>& a, const Matrix3<B>& b) noexcept
{
    using R = std::common_type_t<A, B>;
    auto result = cast<R>(a); result += cast<R>(b); return result;
}
template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(const Matrix3<A>& a, const Matrix3<B>& b) noexcept
{
    using R = std::common_type_t<A, B>;
    auto result = cast<R>(a); result -= cast<R>(b); return result;
}
template<class R>
KOKKOS_INLINE_FUNCTION constexpr Matrix3<R> operator-(Matrix3<R> a) noexcept { a *= R{-1}; return a; }

template<class A, class S> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Matrix3<A>& a, S s) noexcept
{
    using R = std::common_type_t<A, S>;
    auto result = cast<R>(a); result *= static_cast<R>(s); return result;
}
template<class S, class A> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(S s, const Matrix3<A>& a) noexcept { return a*s; }
template<class A, class S> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(const Matrix3<A>& a, S s) noexcept
{
    using R = std::common_type_t<A, S>;
    auto result = cast<R>(a); result /= static_cast<R>(s); return result;
}

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Matrix3<A>& a, Vector3<B> b) noexcept
{
    using R = std::common_type_t<A, B>;
    Vector3<R> result;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            result[i] += static_cast<R>(a.data[i][k])*static_cast<R>(b[k]);
    return result;
}

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Matrix3<A>& a, const Matrix3<B>& b) noexcept
{
    using R = std::common_type_t<A, B>;
    Matrix3<R> result;
    for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t k = 0; k < 3; ++k)
            result.data[i][j] += static_cast<R>(a.data[i][k])*static_cast<R>(b.data[k][j]);
    return result;
}

template<class R>
KOKKOS_INLINE_FUNCTION constexpr Matrix3<R> transpose(const Matrix3<R>& a) noexcept
{
    Matrix3<R> result;
    for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j) result.data[i][j] = a.data[j][i];
    return result;
}
template<class R>
KOKKOS_INLINE_FUNCTION constexpr R trace(const Matrix3<R>& a) noexcept
{ return a.data[0][0]+a.data[1][1]+a.data[2][2]; }
template<class R>
KOKKOS_INLINE_FUNCTION constexpr R determinant(const Matrix3<R>& a) noexcept
{
    return a.data[0][0]*(a.data[1][1]*a.data[2][2]-a.data[1][2]*a.data[2][1]) -
           a.data[0][1]*(a.data[1][0]*a.data[2][2]-a.data[1][2]*a.data[2][0]) +
           a.data[0][2]*(a.data[1][0]*a.data[2][1]-a.data[1][1]*a.data[2][0]);
}

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto outer(Vector3<A> a, Vector3<B> b) noexcept
{
    using R = std::common_type_t<A, B>;
    Matrix3<R> result;
    for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j)
        result.data[i][j] = static_cast<R>(a[i])*static_cast<R>(b[j]);
    return result;
}
template<class R>
KOKKOS_INLINE_FUNCTION bool isfinite(const Matrix3<R>& a) noexcept
{ for (const auto& row : a.data) for (const auto value : row) if (!Kokkos::isfinite(value)) return false; return true; }

} // namespace krona::math
