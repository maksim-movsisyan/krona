#pragma once

#include <krona/numerics/reconstruction/geometry.hpp>
#include <krona/numerics/reconstruction/limiters.hpp>

namespace krona::numerics::reconstruction {

/// @brief Estimates the upstream slope ratio from the local derivative and the jump across the face.
struct DarwishMoukalled {
    template <class R>
    KOKKOS_INLINE_FUNCTION R operator()(R gradient_along_span, R jump) const {
        return R{2} * (gradient_along_span / jump) - R{1};
    } // Caller handles an exactly zero jump.
};

/// @brief Uses the interpolation along the centre-to-centre line as is.
struct NoSkewCorrection {
    template <class R, class G, class V>
    KOKKOS_INLINE_FUNCTION R operator()(G, V, V, R) const {
        return R{0};
    }
};

/// @brief Moves the line interpolation to the actual face point using the local gradient.
struct LinearSkewCorrection {
    template <class R, class G, class V>
    KOKKOS_INLINE_FUNCTION R operator()(G gradient, V offset, V span, R fraction) const {
        return dot_as<R>(gradient, offset) - fraction * dot_as<R>(gradient, span);
    }
};

namespace detail {
template <class R, class S>
KOKKOS_INLINE_FUNCTION R face_fraction(const S& side) {
    R fraction = static_cast<R>(side.fraction);
    if (fraction < R{0}) {
        const R length2 = dot_as<R>(side.span, side.span);
        KOKKOS_ASSERT(length2 > R{0});
        fraction = dot_as<R>(side.offset, side.span) / length2;
    }
    KOKKOS_ASSERT(fraction >= R{0} && fraction <= R{1});
    return fraction;
}
} // namespace detail

/// @brief Unlimited kappa-MUSCL profile using the same virtual-upstream construction as DarwishMoukalled.
/// Kappa=1/3 is the cell-average uniform-grid profile; kappa=1/2 recovers point-valued QUICK there.
/// @tparam Gradients Provider of a Cartesian derivative vector at (cell, component).
/// @tparam Parameter Scalar type storing kappa.
/// @tparam Skew Policy adding the transverse correction.
template <class Gradients, class Parameter = ComputeReal, class Skew = LinearSkewCorrection>
struct MusclKappa {
    static constexpr Geometry geometry = Geometry::Line;
    Gradients gradients;
    Parameter kappa = Parameter{1} / Parameter{3};
    Skew skew{};

    template <class Visitor>
    void visit_inputs(Visitor&& visit) const {
        visit("gradient", gradients);
    }

    template <class R, class S, class Values>
    KOKKOS_INLINE_FUNCTION R increment(const S& side, const Values& values, std::size_t c) const {
        const auto gradient = gradients(side.cell, c);
        const R jump = static_cast<R>(values(side.other, c)) - static_cast<R>(values(side.cell, c));
        const R fraction = detail::face_fraction<R>(side), k = static_cast<R>(kappa);
        // Blend the projected derivative and the jump, then move to the face centroid.
        return fraction * ((R{1} - k) * dot_as<R>(gradient, side.span) + k * jump) +
               skew.template operator()<R>(gradient, side.offset, side.span, fraction);
    }
};

/// @brief Line-based flux limiter with explicit face fraction and optional transverse correction.
/// Recovers the conventional TVD face interpolation on a uniform 1-D midpoint stencil.
/// @tparam Gradients Provider of a Cartesian derivative vector at (cell, component).
/// @tparam Limiter Function of the slope ratio that scales the line increment.
/// @tparam Ratio Function estimating the slope ratio from the local derivative and jump.
/// @tparam Skew Policy adding the transverse correction.
template <class Gradients, class Limiter = tvd::VanLeer, class Ratio = DarwishMoukalled,
          class Skew = LinearSkewCorrection>
struct Tvd {
    static constexpr Geometry geometry = Geometry::Line;
    Gradients gradients;
    Limiter limiter{};
    Ratio ratio{};
    Skew skew{};

    template <class Visitor>
    void visit_inputs(Visitor&& visit) const {
        visit("gradient", gradients);
    }

    template <class R, class S, class D>
    KOKKOS_INLINE_FUNCTION R increment(const S& side, const D& values, std::size_t c) const {
        const R base = static_cast<R>(values(side.cell, c));
        const R jump = static_cast<R>(values(side.other, c)) - base;
        const auto gradient = gradients(side.cell, c);
        const R fraction = detail::face_fraction<R>(side);
        const R correction = skew.template operator()<R>(gradient, side.offset, side.span, fraction);
        // An exactly zero jump eliminates the line term. The skew term can remain nonzero.
        if (jump == R{0})
            return correction;
        const R r = ratio(dot_as<R>(gradient, side.span), jump);
        return fraction * limiter(r) * jump + correction;
    }
};

} // namespace krona::numerics::reconstruction
