#pragma once

#include <krona/math/matrix3.hpp>

namespace krona::math {

/// @brief Six independent entries: xx,yy,zz,xy,xz,yz. Off-diagonal entries are NOT doubled.
template<class Real>
struct SymmetricTensor3 {
    static_assert(std::is_floating_point_v<Real>);
    using value_type = Real;
    Real xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;

    KOKKOS_INLINE_FUNCTION constexpr Real& operator()(std::size_t row, std::size_t col) noexcept
    { return entry(*this, row, col); }
    KOKKOS_INLINE_FUNCTION constexpr const Real& operator()(std::size_t row, std::size_t col) const noexcept
    { return entry(*this, row, col); }

    KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3& operator+=(SymmetricTensor3 b) noexcept
    { xx += b.xx; yy += b.yy; zz += b.zz; xy += b.xy; xz += b.xz; yz += b.yz; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3& operator-=(SymmetricTensor3 b) noexcept
    { xx -= b.xx; yy -= b.yy; zz -= b.zz; xy -= b.xy; xz -= b.xz; yz -= b.yz; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3& operator*=(Real s) noexcept
    { xx *= s; yy *= s; zz *= s; xy *= s; xz *= s; yz *= s; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3& operator/=(Real s) noexcept
    { xx /= s; yy /= s; zz /= s; xy /= s; xz /= s; yz /= s; return *this; }

private:
    template<class Self>
    KOKKOS_INLINE_FUNCTION static constexpr decltype(auto) entry(Self& a, std::size_t row, std::size_t col) noexcept
    {
        KOKKOS_ASSERT(row < 3 && col < 3);
        if (row > col) { const auto tmp = row; row = col; col = tmp; }
        if (row == 0) return (col == 0 ? a.xx : col == 1 ? a.xy : a.xz);
        if (row == 1) return (col == 1 ? a.yy : a.yz);
        return (a.zz);
    }
};

template<class To, class From>
KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3<To> cast(SymmetricTensor3<From> a) noexcept
{ return {static_cast<To>(a.xx), static_cast<To>(a.yy), static_cast<To>(a.zz),
          static_cast<To>(a.xy), static_cast<To>(a.xz), static_cast<To>(a.yz)}; }

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(SymmetricTensor3<A> a, SymmetricTensor3<B> b) noexcept
{
    using R = std::common_type_t<A, B>;
    auto result = cast<R>(a); result += cast<R>(b); return result;
}
template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(SymmetricTensor3<A> a, SymmetricTensor3<B> b) noexcept
{
    using R = std::common_type_t<A, B>;
    auto result = cast<R>(a); result -= cast<R>(b); return result;
}
template<class R>
KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3<R> operator-(SymmetricTensor3<R> a) noexcept
{ a *= R{-1}; return a; }
template<class A, class S> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(SymmetricTensor3<A> a, S s) noexcept
{
    using R = std::common_type_t<A, S>;
    auto result = cast<R>(a); result *= static_cast<R>(s); return result;
}
template<class S, class A> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(S s, SymmetricTensor3<A> a) noexcept { return a*s; }
template<class A, class S> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(SymmetricTensor3<A> a, S s) noexcept
{
    using R = std::common_type_t<A, S>;
    auto result = cast<R>(a); result /= static_cast<R>(s); return result;
}

template<class R>
KOKKOS_INLINE_FUNCTION constexpr Matrix3<R> to_matrix(SymmetricTensor3<R> a) noexcept
{ return {{{a.xx, a.xy, a.xz}, {a.xy, a.yy, a.yz}, {a.xz, a.yz, a.zz}}}; }

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(SymmetricTensor3<A> a, Vector3<B> b) noexcept
{
    using R = std::common_type_t<A, B>;
    const auto s = cast<R>(a); const auto v = cast<R>(b);
    return Vector3<R>{s.xx*v.x+s.xy*v.y+s.xz*v.z,
                      s.xy*v.x+s.yy*v.y+s.yz*v.z,
                      s.xz*v.x+s.yz*v.y+s.zz*v.z};
}
template<class R>
KOKKOS_INLINE_FUNCTION constexpr R trace(SymmetricTensor3<R> a) noexcept { return a.xx+a.yy+a.zz; }
template<class R>
KOKKOS_INLINE_FUNCTION constexpr SymmetricTensor3<R> symmetric_outer(Vector3<R> v) noexcept
{ return {v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z}; }

/// @brief Full Frobenius contraction sum_ij(a_ij*b_ij); off-diagonal products occur twice.
template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto double_contract(SymmetricTensor3<A> a, SymmetricTensor3<B> b) noexcept
{
    using R = std::common_type_t<A, B>;
    const auto x = cast<R>(a), y = cast<R>(b);
    return x.xx*y.xx+x.yy*y.yy+x.zz*y.zz+R{2}*(x.xy*y.xy+x.xz*y.xz+x.yz*y.yz);
}
template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto quadratic_form(SymmetricTensor3<A> a, Vector3<B> v) noexcept
{ return dot(v, a*v); }
template<class R>
KOKKOS_INLINE_FUNCTION bool isfinite(SymmetricTensor3<R> a) noexcept
{
    return Kokkos::isfinite(a.xx) && Kokkos::isfinite(a.yy) && Kokkos::isfinite(a.zz) &&
           Kokkos::isfinite(a.xy) && Kokkos::isfinite(a.xz) && Kokkos::isfinite(a.yz);
}

} // namespace krona::math
