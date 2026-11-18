#pragma once

#include <krona/numerics/reconstruction/geometry.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::numerics::reconstruction {

/// @brief Holds three sample indices and prepared weights for one side of a face.
/// @tparam Real Scalar type in which the weights are stored.
template <class Real = ComputeReal>
struct ThreePointStencil {
    std::size_t cell = 0;
    Kokkos::Array<std::size_t, 3> samples{}; // Upstream, centre, downstream.
    Kokkos::Array<Real, 3> weights{};
    bool valid = false;
};

/// @brief Quadratic fit to projected cell averages. Variances are <(s-<s>)^2> in the same coordinate.
/// Exact for polynomials in this one coordinate; a full 3-D quadratic requires Quadratic and tensor moments.
/// @tparam R Arithmetic type of the projected coordinates and fit.
/// @param samples Neighbour, base cell, and third sample indices, in matching input order.
/// @param centres Centroids projected onto one common coordinate.
/// @param variances Central second moments along that coordinate, in units of length squared.
/// @param target Coordinate where the polynomial is evaluated.
/// @param tolerance Relative threshold for rejecting a nearly singular fit.
/// @details A failed fit returns valid=false and NaN weights.
template <class R>
KOKKOS_INLINE_FUNCTION auto cell_average_quick_stencil(Kokkos::Array<std::size_t, 3> samples,
                                                       Kokkos::Array<R, 3> centres,
                                                       Kokkos::Array<R, 3> variances, R target,
                                                       R tolerance = R{64} * Kokkos::epsilon_v<R>) {
    const R nan = Kokkos::quiet_NaN_v<R>;
    ThreePointStencil<R> result{samples[1], samples, {nan, nan, nan}, false};
    if (!(tolerance >= R{0}) || !Kokkos::isfinite(tolerance) || !Kokkos::isfinite(target))
        return result;
    for (std::size_t i = 0; i < 3; ++i)
        if (!Kokkos::isfinite(centres[i]) || !Kokkos::isfinite(variances[i]) || variances[i] < R{0})
            return result;
    // Translate the base centroid to zero and normalize by the sample spacing.
    const R scale = Kokkos::max(Kokkos::abs(centres[0] - centres[1]), Kokkos::abs(centres[2] - centres[1]));
    if (!(scale > R{0}) || !Kokkos::isfinite(scale))
        return result;
    const R u = (centres[0] - centres[1]) / scale, v = (centres[2] - centres[1]) / scale;
    const R f = (target - centres[1]) / scale;
    const R mu = (variances[0] / scale) / scale, mc = (variances[1] / scale) / scale,
            md = (variances[2] / scale) / scale;
    // Match the cell averages of the linear and quadratic basis functions.
    const R a = u * u + mu - mc, b = v * v + md - mc, rhs = f * f - mc;
    const R determinant = u * b - v * a;
    if (!Kokkos::isfinite(determinant) ||
        Kokkos::abs(determinant) <= tolerance * (Kokkos::abs(u * b) + Kokkos::abs(v * a)))
        return result;
    const R wu = (f * b - v * rhs) / determinant, wd = (u * rhs - f * a) / determinant;
    const R wc = R{1} - wu - wd;
    if (!Kokkos::isfinite(wu) || !Kokkos::isfinite(wc) || !Kokkos::isfinite(wd))
        return result;
    result.weights = {wu, wc, wd};
    result.valid = true;
    return result;
}

/// @brief Explicit point-value interpretation; on a uniform midpoint stencil gives (-1,6,3)/8.
template <class R>
KOKKOS_INLINE_FUNCTION auto point_quick_stencil(Kokkos::Array<std::size_t, 3> samples,
                                                Kokkos::Array<R, 3> centres, R target) {
    return cell_average_quick_stencil(samples, centres, Kokkos::Array<R, 3>{R{0}, R{0}, R{0}}, target);
}

/// @brief Uses the prepared QUICK weights to evaluate a cell-relative increment.
struct Quick {
    static constexpr Geometry geometry = Geometry::Explicit;

    template <class R, class S, class Values>
    KOKKOS_INLINE_FUNCTION R increment(const S& side, const Values& values, std::size_t c) const {
        KOKKOS_ASSERT(side.valid);
        const R base = static_cast<R>(values(side.cell, c));
        return static_cast<R>(side.weights[0]) * (static_cast<R>(values(side.samples[0], c)) - base) +
               static_cast<R>(side.weights[2]) * (static_cast<R>(values(side.samples[2], c)) - base);
    }
};

} // namespace krona::numerics::reconstruction
