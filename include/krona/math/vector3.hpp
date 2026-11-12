#pragma once

#include <Kokkos_Assert.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <type_traits>

namespace krona::math {

/// @brief Three Cartesian values, stored inline. Scalar precision is explicit.
template<class Real>
struct Vector3 {
    static_assert(std::is_floating_point_v<Real>);
    using value_type = Real;
    Real x = 0, y = 0, z = 0;

    KOKKOS_INLINE_FUNCTION constexpr Real& operator[](std::size_t i) noexcept
    { KOKKOS_ASSERT(i < 3); return i == 0 ? x : i == 1 ? y : z; }
    KOKKOS_INLINE_FUNCTION constexpr const Real& operator[](std::size_t i) const noexcept
    { KOKKOS_ASSERT(i < 3); return i == 0 ? x : i == 1 ? y : z; }

    KOKKOS_INLINE_FUNCTION constexpr Vector3& operator+=(Vector3 b) noexcept
    { x += b.x; y += b.y; z += b.z; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr Vector3& operator-=(Vector3 b) noexcept
    { x -= b.x; y -= b.y; z -= b.z; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr Vector3& operator*=(Real s) noexcept
    { x *= s; y *= s; z *= s; return *this; }
    KOKKOS_INLINE_FUNCTION constexpr Vector3& operator/=(Real s) noexcept
    { x /= s; y /= s; z /= s; return *this; }

    template<class Other>
    KOKKOS_INLINE_FUNCTION constexpr auto dot(Vector3<Other> b) const noexcept {
        using R = std::common_type_t<Real, Other>;
        return static_cast<R>(x)*static_cast<R>(b.x) + static_cast<R>(y)*static_cast<R>(b.y) +
               static_cast<R>(z)*static_cast<R>(b.z);
    }

    template<class Other>
    KOKKOS_INLINE_FUNCTION constexpr auto cross(Vector3<Other> b) const noexcept {
        using R = std::common_type_t<Real, Other>;
        const R ax = static_cast<R>(x), ay = static_cast<R>(y), az = static_cast<R>(z);
        const R bx = static_cast<R>(b.x), by = static_cast<R>(b.y), bz = static_cast<R>(b.z);
        return Vector3<R>{ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx};
    }

    KOKKOS_INLINE_FUNCTION constexpr Real norm2() const noexcept { return dot(*this); }
    // Use two-argument hypot: some device backends implement the three-argument form as sqrt(dot).
    KOKKOS_INLINE_FUNCTION Real norm() const noexcept
    { return Kokkos::hypot(Kokkos::hypot(x, y), z); }
};

template<class To, class From>
KOKKOS_INLINE_FUNCTION constexpr Vector3<To> cast(Vector3<From> v) noexcept
{ return {static_cast<To>(v.x), static_cast<To>(v.y), static_cast<To>(v.z)}; }

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(Vector3<A> a, Vector3<B> b) noexcept {
    using R = std::common_type_t<A, B>;
    const auto x = cast<R>(a), y = cast<R>(b);
    return Vector3<R>{x.x+y.x, x.y+y.y, x.z+y.z};
}

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(Vector3<A> a, Vector3<B> b) noexcept {
    using R = std::common_type_t<A, B>;
    const auto x = cast<R>(a), y = cast<R>(b);
    return Vector3<R>{x.x-y.x, x.y-y.y, x.z-y.z};
}

template<class R>
KOKKOS_INLINE_FUNCTION constexpr Vector3<R> operator-(Vector3<R> v) noexcept
{ return {-v.x, -v.y, -v.z}; }

template<class A, class S> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(Vector3<A> v, S s) noexcept {
    using R = std::common_type_t<A, S>;
    const auto x = cast<R>(v);
    const R scale = static_cast<R>(s);
    return Vector3<R>{x.x*scale, x.y*scale, x.z*scale};
}

template<class S, class A> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(S s, Vector3<A> v) noexcept { return v*s; }

template<class A, class S> requires std::is_arithmetic_v<S>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(Vector3<A> v, S s) noexcept {
    using R = std::common_type_t<A, S>;
    const auto x = cast<R>(v);
    const R divisor = static_cast<R>(s);
    return Vector3<R>{x.x/divisor, x.y/divisor, x.z/divisor};
}

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto dot(Vector3<A> a, Vector3<B> b) noexcept { return a.dot(b); }

template<class R, class A, class B>
KOKKOS_INLINE_FUNCTION constexpr R dot_as(Vector3<A> a, Vector3<B> b) noexcept
{ return cast<R>(a).dot(cast<R>(b)); }

template<class A, class B>
KOKKOS_INLINE_FUNCTION constexpr auto cross(Vector3<A> a, Vector3<B> b) noexcept { return a.cross(b); }

template<class R>
KOKKOS_INLINE_FUNCTION constexpr R norm2(Vector3<R> v) noexcept { return v.norm2(); }
template<class R>
KOKKOS_INLINE_FUNCTION R norm(Vector3<R> v) noexcept { return v.norm(); }
template<class R>
KOKKOS_INLINE_FUNCTION bool isfinite(Vector3<R> v) noexcept
{ return Kokkos::isfinite(v.x) && Kokkos::isfinite(v.y) && Kokkos::isfinite(v.z); }

} // namespace krona::math
