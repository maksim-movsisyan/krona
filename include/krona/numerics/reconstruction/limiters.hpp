#pragma once

#include <krona/numerics/reconstruction/types.hpp>

namespace krona::numerics::reconstruction {

namespace tvd {
struct Minmod {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R r) const { return Kokkos::max(R{0}, Kokkos::min(R{1}, r)); }
};
struct Superbee {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R r) const
    { return Kokkos::max(R{0}, Kokkos::max(Kokkos::min(R{2}*r, R{1}), Kokkos::min(r, R{2}))); }
};
struct MonotonizedCentral {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R r) const
    { return Kokkos::max(R{0}, Kokkos::min(R{2}, Kokkos::min(R{2}*r, (R{1}+r)/R{2}))); }
};
struct VanLeer {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R r) const
    {
        if (!(r > R{0})) return R{0};
        return r <= R{1} ? R{2}*r/(R{1}+r) : R{2}/(R{1}+R{1}/r);
    }
};
struct VanAlbada {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R r) const
    {
        if (!(r > R{0})) return R{0};
        if (r > R{1}) { const R inv = R{1}/r; return (R{1}+inv)/(R{1}+inv*inv); }
        return r*(r+R{1})/(r*r+R{1});
    }
};
struct Koren {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R r) const
    { return Kokkos::max(R{0}, Kokkos::min(R{2}, Kokkos::min(R{2}*r, (R{1}+R{2}*r)/R{3}))); }
};
} // namespace tvd

struct BarthJespersen {
    template<class R>
    KOKKOS_INLINE_FUNCTION R operator()(const LimiterContext<R>& c) const
    {
        if (c.increment == R{0}) return R{1};
        const R allowed = c.increment > R{0} ? c.maximum-c.value : c.minimum-c.value;
        return Kokkos::max(R{0}, Kokkos::min(R{1}, allowed/c.increment));
    }
};

struct ZeroRegularization {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(const LimiterContext<R>&) const { return R{0}; }
};
/// @brief Epsilon has the same units as the reconstructed variable; its square enters the limiter.
template<class Real = ComputeReal>
struct ConstantRegularization {
    Real epsilon = 0;
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(const LimiterContext<R>&) const { return static_cast<R>(epsilon); }
};

/// @brief Smoothed limiter. Nonzero regularization can relax local bounds; this is not a strict bound guarantee.
/// Even at epsilon=0 it may reduce an already bounded increment (allowed/increment=1 gives 3/4).
/// A custom regularization(context) can supply cell/component-dependent epsilon without new arrays.
template<class Regularization = ZeroRegularization>
struct Venkatakrishnan {
    Regularization regularization{};
    template<class R>
    KOKKOS_INLINE_FUNCTION R operator()(const LimiterContext<R>& c) const
    {
        if (c.increment == R{0}) return R{1};
        const R a = Kokkos::abs(c.increment);
        const R b = Kokkos::max(R{0}, c.increment > R{0} ? c.maximum-c.value : c.value-c.minimum);
        const R epsilon = static_cast<R>(regularization(c));
        KOKKOS_ASSERT(epsilon >= R{0} && Kokkos::isfinite(epsilon));
        const R scale = Kokkos::max(a, Kokkos::max(b, epsilon));
        const R x = a/scale, y = b/scale, e = epsilon/scale;
        return Kokkos::min(R{1}, (y*y+R{2}*x*y+e*e)/(y*y+x*y+R{2}*x*x+e*e));
    }
};
template<class R> Venkatakrishnan(R)->Venkatakrishnan<R>;

} // namespace krona::numerics::reconstruction
