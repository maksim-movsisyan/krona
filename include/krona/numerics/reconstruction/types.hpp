#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace krona::numerics::reconstruction {

template<class Real = ComputeReal>
struct Vector {
    Real x = 0, y = 0, z = 0;
    KOKKOS_INLINE_FUNCTION Vector operator-() const { return {-x, -y, -z}; }
    KOKKOS_INLINE_FUNCTION Vector operator-(Vector b) const { return {x-b.x, y-b.y, z-b.z}; }
    KOKKOS_INLINE_FUNCTION Vector operator*(Real a) const { return {a*x, a*y, a*z}; }
};
template<class A, class B>
KOKKOS_INLINE_FUNCTION auto dot(Vector<A> a, Vector<B> b)
{
    using R = std::common_type_t<A, B>;
    return static_cast<R>(a.x)*static_cast<R>(b.x) + static_cast<R>(a.y)*static_cast<R>(b.y) +
           static_cast<R>(a.z)*static_cast<R>(b.z);
}
template<class R, class A, class B>
KOKKOS_INLINE_FUNCTION R dot_as(Vector<A> a, Vector<B> b)
{
    return static_cast<R>(a.x)*static_cast<R>(b.x) + static_cast<R>(a.y)*static_cast<R>(b.y) +
           static_cast<R>(a.z)*static_cast<R>(b.z);
}

/// @brief Hessian or central cell moment tensor; off-diagonal entries are stored once.
template<class Real = ComputeReal>
struct SymmetricTensor { Real xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0; };

/// @brief Explicit geometry for one side; span points from cell to other, offset to the face point.
/// A negative fraction requests orthogonal projection onto span; an explicit fraction can use face-plane intersection.
template<class Real = ComputeReal>
struct CellSide {
    std::size_t cell = 0, other = 0;
    Vector<Real> offset{}, span{};
    Real fraction = -1;
};
template<class Real = ComputeReal>
struct FaceGeometry { CellSide<Real> left, right; };
template<class Left, class Right = Left>
struct FaceStencil { Left left; Right right; };

template<class Real>
KOKKOS_INLINE_FUNCTION auto face_geometry(std::size_t owner, std::size_t neighbour,
    Vector<Real> owner_to_face, Vector<Real> neighbour_to_face, Real owner_fraction = Real{-1})
{
    const auto span = owner_to_face-neighbour_to_face;
    return FaceGeometry<Real>{{owner, neighbour, owner_to_face, span, owner_fraction},
        {neighbour, owner, neighbour_to_face, -span, owner_fraction < Real{0} ? Real{-1} : Real{1}-owner_fraction}};
}

template<class Real, std::size_t N>
struct FaceValues { Kokkos::Array<Real, N> left{}, right{}; };

/// @brief All positions are immutable for the duration of submitted kernels.
struct CellRange { std::size_t begin = 0, end = std::numeric_limits<std::size_t>::max(); };
struct Componentwise {};
struct SharedMinimum {};

enum Input : unsigned { Values = 0, Gradients = 1, Factors = 2, Hessians = 4, Moments = 8, OtherValues = 16 };
struct NoGradients {};
struct NoHessians {};
struct NoMoments {};
struct UnitFactors {
    static constexpr std::size_t components = 1;
    KOKKOS_INLINE_FUNCTION int operator()(std::size_t, std::size_t) const { return 1; }
};

/// @brief Device-copyable borrowed inputs; gradients use (cell, component), moments use (cell).
template<class V, class G = NoGradients, class L = UnitFactors, class H = NoHessians, class M = NoMoments>
struct Data {
    static constexpr std::size_t components = V::components;
    V values; G gradients; L factors; H hessians; M moments;
    KOKKOS_INLINE_FUNCTION auto factor(std::size_t cell, std::size_t component) const
    {
        static_assert(L::components == 1 || L::components == components);
        return factors(cell, L::components == 1 ? 0 : component);
    }
};
template<class V, class G = NoGradients, class L = UnitFactors, class H = NoHessians, class M = NoMoments>
KOKKOS_INLINE_FUNCTION auto data(V values, G gradients = {}, L factors = {}, H hessians = {}, M moments = {})
{ return Data<V, G, L, H, M>{values, gradients, factors, hessians, moments}; }

template<class Real>
struct LimiterContext {
    std::size_t cell = 0, component = 0;
    Real value = 0, minimum = 0, maximum = 0, increment = 0;
};
template<class Real>
struct StencilEntry { std::size_t sample = 0; CellSide<Real> side; };

} // namespace krona::numerics::reconstruction