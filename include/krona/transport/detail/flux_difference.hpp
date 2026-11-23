#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/flow/flux.hpp>
#include <krona/math/matrix5.hpp>

namespace krona::transport::detail {

template<class R>
KOKKOS_INLINE_FUNCTION auto primitive_array(boundary::physical::Primitive<R, true> q)
{ return Kokkos::Array<R, 5>{q.pressure, q.velocity.x, q.velocity.y, q.velocity.z, q.temperature}; }

template<class R>
KOKKOS_INLINE_FUNCTION auto flux_array(flow::Flux<R> q)
{ return Kokkos::Array<R, 5>{q.mass, q.momentum.x, q.momentum.y, q.momentum.z, q.energy}; }

/// @brief Differentiates all five flux components with respect to one primitive variable.
/// @tparam R Scalar type of the state and derivative.
/// @tparam Sample Callable that evaluates the boundary flux at a trial trace.
/// @param flux Bound flux function.
/// @param point Unperturbed primitive trace in p, u, v, w, T order.
/// @param base Flux at point.
/// @param column Index of the primitive variable to perturb.
/// @param step Initial perturbation size. It is reduced if needed to obtain valid samples.
/// @param derivative Matrix whose selected column is written on success.
template<class R, class Sample>
KOKKOS_INLINE_FUNCTION bool differentiate_flux_column(const Sample& flux, const Kokkos::Array<R, 5>& point,
                                                      const Kokkos::Array<R, 5>& base, std::size_t column,
                                                      R step, math::Matrix5<R>& derivative) {
    const R x = point[column];
    for (int attempt = 0; attempt < 6; ++attempt, step *= R{0.5}) {
        auto plus = point, minus = point;
        plus[column] = x + step;
        minus[column] = x - step;
        const R forward = plus[column] - x, backward = x - minus[column];
        const auto upper = flux(plus), lower = flux(minus);
        const bool up = forward > R{0} && upper.valid();
        const bool down = backward > R{0} && lower.valid();
        if (up && down) {
            const auto a = flux_array(upper.flux), b = flux_array(lower.flux);
            for (std::size_t row = 0; row < 5; ++row) {
                const R right = (a[row] - base[row]) / forward;
                const R left = (base[row] - b[row]) / backward;
                derivative(row, column) = left * (forward / (forward + backward)) +
                                           right * (backward / (forward + backward));
            }
            return true;
        }
        if (!up && !down) continue;

        // Use a quadratic one-sided stencil when a probe leaves the admissible domain.
        const auto a = flux_array(up ? upper.flux : lower.flux);
        const R first_step = up ? forward : -backward;
        auto second = point;
        second[column] = x + R{2} * first_step;
        const R second_step = second[column] - x;
        if (second_step == first_step) continue;
        const auto far = flux(second);
        if (!far.valid()) continue;
        const auto b = flux_array(far.flux);
        for (std::size_t row = 0; row < 5; ++row) {
            const R near_slope = (a[row] - base[row]) / first_step;
            const R far_slope = (b[row] - base[row]) / second_step;
            derivative(row, column) = near_slope + first_step * (near_slope - far_slope) / (second_step - first_step);
        }
        return true;
    }
    return false;
}
} // namespace krona::transport::detail
