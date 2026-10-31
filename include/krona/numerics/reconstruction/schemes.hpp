#pragma once

#include <krona/numerics/reconstruction/limiters.hpp>

#include <Kokkos_NumericTraits.hpp>

namespace krona::numerics::reconstruction {

struct FirstOrder {
    static constexpr unsigned requirements = Values;
    template<class R, class Side, class D>
    KOKKOS_INLINE_FUNCTION R increment(const Side&, const D&, std::size_t) const
    { return R{0}; }
};

struct SecondOrderUpwind {
    static constexpr unsigned requirements = Gradients;
    template<class R, class Side, class D>
    KOKKOS_INLINE_FUNCTION R increment(const Side& side, const D& d, std::size_t c) const
    { return dot_as<R>(d.gradients(side.cell, c), side.offset); }
};

/// @brief Mean-preserving quadratic Taylor polynomial, evaluated at one point.
/// moments(cell) = volume average of r*r^T about the volume centroid; hessians(cell,c) are true second derivatives.
struct Quadratic {
    static constexpr unsigned requirements = Gradients | Hessians | Moments;
    template<class R, class Side, class D>
    KOKKOS_INLINE_FUNCTION R increment(const Side& side, const D& d, std::size_t c) const
    {
        const auto h = d.hessians(side.cell, c);
        const auto m = d.moments(side.cell);
        const R x = static_cast<R>(side.offset.x), y = static_cast<R>(side.offset.y), z = static_cast<R>(side.offset.z);
        const R curvature = static_cast<R>(h.xx)*(x*x-static_cast<R>(m.xx)) +
            static_cast<R>(h.yy)*(y*y-static_cast<R>(m.yy)) + static_cast<R>(h.zz)*(z*z-static_cast<R>(m.zz)) +
            R{2}*(static_cast<R>(h.xy)*(x*y-static_cast<R>(m.xy)) +
                  static_cast<R>(h.xz)*(x*z-static_cast<R>(m.xz)) + static_cast<R>(h.yz)*(y*z-static_cast<R>(m.yz)));
        return SecondOrderUpwind{}.template increment<R>(side, d, c) + R{0.5}*curvature;
    }
};

/// @brief Applies a precomputed cell/component factor to the entire high-order increment.
template<class Scheme = SecondOrderUpwind>
struct Limited {
    static constexpr unsigned requirements = Scheme::requirements | Factors;
    Scheme scheme{};
    template<class R, class Side, class D>
    KOKKOS_INLINE_FUNCTION R increment(const Side& side, const D& d, std::size_t c) const
    {
        const R factor = static_cast<R>(d.factor(side.cell, c));
        KOKKOS_ASSERT(factor >= R{0} && factor <= R{1});
        if (factor == R{0}) return R{0};
        return factor*scheme.template increment<R>(side, d, c);
    }
};
using LimitedLinear = Limited<SecondOrderUpwind>;
using LimitedQuadratic = Limited<Quadratic>;

/// @brief Virtual upstream difference from the owner gradient; no distant cell lookup. Darwish & Moukalled.
struct DarwishMoukalled {
    template<class R> KOKKOS_INLINE_FUNCTION R operator()(R gradient_along_span, R jump) const
    { return R{2}*(gradient_along_span/jump)-R{1}; } // Caller handles an exactly zero jump.
};
struct NoSkewCorrection {
    template<class R, class G, class V>
    KOKKOS_INLINE_FUNCTION R operator()(G, V, V, R) const { return R{0}; }
};
/// @brief Adds the transverse Taylor increment. This preserves affine fields at skew face points.
/// A TVD ratio limiter alone does not bound this multidimensional correction.
struct LinearSkewCorrection {
    template<class R, class G, class V>
    KOKKOS_INLINE_FUNCTION R operator()(G gradient, V offset, V span, R fraction) const
    { return dot_as<R>(gradient, offset) - fraction*dot_as<R>(gradient, span); }
};

namespace detail {
template<class R, class S>
KOKKOS_INLINE_FUNCTION R face_fraction(const S& side)
{
    R fraction = static_cast<R>(side.fraction);
    if (fraction < R{0}) {
        const R length2 = dot_as<R>(side.span, side.span);
        KOKKOS_ASSERT(length2 > R{0});
        fraction = dot_as<R>(side.offset, side.span)/length2;
    }
    KOKKOS_ASSERT(fraction >= R{0} && fraction <= R{1});
    return fraction;
}
} // namespace detail

/// @brief Unlimited kappa-MUSCL profile using the same virtual-upstream construction as DarwishMoukalled.
/// Kappa=1/3 is the cell-average uniform-grid profile; kappa=1/2 recovers point-valued QUICK there.
/// General unstructured quadratic exactness is not implied; use explicit moments/stencils for that contract.
template<class Parameter = ComputeReal, class Skew = LinearSkewCorrection>
struct MusclKappa {
    static constexpr unsigned requirements = Gradients | OtherValues;
    Parameter kappa = Parameter{1}/Parameter{3};
    Skew skew{};
    template<class R, class S, class D>
    KOKKOS_INLINE_FUNCTION R increment(const S& side, const D& d, std::size_t c) const
    {
        const auto gradient = d.gradients(side.cell, c);
        const R jump = static_cast<R>(d.values(side.other, c))-static_cast<R>(d.values(side.cell, c));
        const R fraction = detail::face_fraction<R>(side), k = static_cast<R>(kappa);
        return fraction*((R{1}-k)*dot_as<R>(gradient, side.span)+k*jump) +
            skew.template operator()<R>(gradient, side.offset, side.span, fraction);
    }
};
template<class P> MusclKappa(P)->MusclKappa<P>;

/// @brief Line-based flux limiter with explicit face fraction and optional transverse correction.
/// Recovers the conventional TVD face interpolation on a uniform 1-D midpoint stencil.
/// On general meshes it is a reconstruction policy, not a proof of TVD/positivity of a complete solver.
template<class Limiter = tvd::VanLeer, class Ratio = DarwishMoukalled, class Skew = LinearSkewCorrection>
struct Tvd {
    static constexpr unsigned requirements = Gradients | OtherValues;
    Limiter limiter{}; Ratio ratio{}; Skew skew{};
    template<class R, class S, class D>
    KOKKOS_INLINE_FUNCTION R increment(const S& side, const D& d, std::size_t c) const
    {
        const R base = static_cast<R>(d.values(side.cell, c));
        const R jump = static_cast<R>(d.values(side.other, c))-base;
        const auto gradient = d.gradients(side.cell, c);
        const R fraction = detail::face_fraction<R>(side);
        const R correction = skew.template operator()<R>(gradient, side.offset, side.span, fraction);
        if (jump == R{0}) return correction;
        const R r = ratio(dot_as<R>(gradient, side.span), jump);
        return fraction*limiter(r)*jump+correction;
    }
};
template<class L> Tvd(L)->Tvd<L>;

/// @brief Three explicitly selected samples and geometry-dependent weights; no automatic stencil search.
template<class Real = ComputeReal>
struct ThreePointStencil {
    std::size_t cell = 0;
    Kokkos::Array<std::size_t, 3> samples{}; // Upstream, centre, downstream.
    Kokkos::Array<Real, 3> weights{};
    bool valid = false;
};

/// @brief Quadratic fit to projected cell averages. Variances are <(s-<s>)^2> in the same coordinate.
/// Exact for polynomials in this one coordinate; a full 3-D quadratic requires Quadratic and tensor moments.
/// Check valid before submission. Invalid/degenerate geometry returns NaN weights, never a silent fallback.
template<class R>
KOKKOS_INLINE_FUNCTION auto cell_average_quick_stencil(Kokkos::Array<std::size_t, 3> samples,
    Kokkos::Array<R, 3> centres, Kokkos::Array<R, 3> variances, R target,
    R tolerance = R{64}*Kokkos::epsilon_v<R>)
{
    const R nan = Kokkos::quiet_NaN_v<R>;
    ThreePointStencil<R> result{samples[1], samples, {nan, nan, nan}, false};
    if (!(tolerance >= R{0}) || !Kokkos::isfinite(tolerance) || !Kokkos::isfinite(target)) return result;
    for (std::size_t i = 0; i < 3; ++i)
        if (!Kokkos::isfinite(centres[i]) || !Kokkos::isfinite(variances[i]) || variances[i] < R{0}) return result;
    const R scale = Kokkos::max(Kokkos::abs(centres[0]-centres[1]), Kokkos::abs(centres[2]-centres[1]));
    if (!(scale > R{0}) || !Kokkos::isfinite(scale)) return result;
    const R u = (centres[0]-centres[1])/scale, v = (centres[2]-centres[1])/scale;
    const R f = (target-centres[1])/scale;
    const R mu = (variances[0]/scale)/scale, mc = (variances[1]/scale)/scale, md = (variances[2]/scale)/scale;
    const R a = u*u+mu-mc, b = v*v+md-mc, rhs = f*f-mc;
    const R determinant = u*b-v*a;
    if (!Kokkos::isfinite(determinant) || Kokkos::abs(determinant) <= tolerance*(Kokkos::abs(u*b)+Kokkos::abs(v*a))) return result;
    const R wu = (f*b-v*rhs)/determinant, wd = (u*rhs-f*a)/determinant;
    if (!Kokkos::isfinite(wu) || !Kokkos::isfinite(wd)) return result;
    result.weights = {wu, R{1}-wu-wd, wd}; result.valid = true;
    return result;
}

/// @brief Explicit point-value interpretation; on a uniform midpoint stencil gives (-1,6,3)/8.
template<class R>
KOKKOS_INLINE_FUNCTION auto point_quick_stencil(Kokkos::Array<std::size_t, 3> samples,
    Kokkos::Array<R, 3> centres, R target)
{ return cell_average_quick_stencil(samples, centres, Kokkos::Array<R, 3>{R{0}, R{0}, R{0}}, target); }

struct Quick {
    static constexpr unsigned requirements = OtherValues;
    template<class R, class S, class D>
    KOKKOS_INLINE_FUNCTION R increment(const S& side, const D& d, std::size_t c) const
    {
        KOKKOS_ASSERT(side.valid);
        const R base = static_cast<R>(d.values(side.cell, c));
        return static_cast<R>(side.weights[0])*(static_cast<R>(d.values(side.samples[0], c))-base) +
                      static_cast<R>(side.weights[2])*(static_cast<R>(d.values(side.samples[2], c))-base);
    }
};

} // namespace krona::numerics::reconstruction