#pragma once

#include <krona/transport/euler/boundary.hpp>
#include <krona/transport/euler/jacobian.hpp>
#include <krona/transport/euler/state.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace krona::transport::euler {

template<class R>
struct BoundaryLinearizationState {
    numerics::riemann::State<R> cell;
    boundary::physical::Primitive<R, true> trace, prepared_face;
};

template<class R>
struct BoundaryJacobian {
    numerics::riemann::Matrix5<R> block;
    bool valid = false;
};

namespace detail {
template<class R>
KOKKOS_INLINE_FUNCTION auto primitive_array(boundary::physical::Primitive<R, true> q)
{ return Kokkos::Array<R, 5>{q.pressure, q.velocity.x, q.velocity.y, q.velocity.z, q.temperature}; }

template<class R>
KOKKOS_INLINE_FUNCTION auto conservative_array(numerics::riemann::Conservative<R> q)
{ return Kokkos::Array<R, 5>{q.mass, q.momentum.x, q.momentum.y, q.momentum.z, q.energy}; }

template<class R, class Sample>
KOKKOS_INLINE_FUNCTION bool differentiate_flux_column(const Sample& flux, const Kokkos::Array<R, 5>& point,
    const Kokkos::Array<R, 5>& base, std::size_t column, R step, numerics::riemann::Matrix5<R>& derivative) {
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
            const auto a = conservative_array(upper.flux), b = conservative_array(lower.flux);
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
        const auto a = conservative_array(up ? upper.flux : lower.flux);
        const R first_step = up ? forward : -backward;
        auto second = point;
        second[column] = x + R{2} * first_step;
        const R second_step = second[column] - x;
        if (second_step == first_step) continue;
        const auto far = flux(second);
        if (!far.valid()) continue;
        const auto b = conservative_array(far.flux);
        for (std::size_t row = 0; row < 5; ++row) {
            const R near_slope = (a[row] - base[row]) / first_step;
            const R far_slope = (b[row] - base[row]) / second_step;
            derivative(row, column) = near_slope + first_step * (near_slope - far_slope) / (second_step - first_step);
        }
        return true;
    }
    return false;
}
} // namespace detail

/// @brief Differentiates the actual boundary flux in primitive variables, then applies analytic dW_cell/dU_cell.
/// Holds the reconstruction increment and face slots fixed. Profile providers receive the same face context.
struct FiniteDifferenceBoundary {
    template<class R, class Condition, class Model, class Closure, class StateLaw, class Solver>
    KOKKOS_INLINE_FUNCTION BoundaryJacobian<R> operator()(const boundary::FaceContext<R>& context,
        const Condition& condition, const Model& model, const Closure& closure,
        const BoundaryLinearizationState<R>& states, const StateLaw& state_law, const Solver& solver) const {
        namespace ri = numerics::riemann;
        const ri::Vector<R> normal{context.nx, context.ny, context.nz};
        const auto flux = [&](const Kokkos::Array<R, 5>& point) -> ri::Result<R> {
            const auto inside = detail::primitive(point);
            const auto left = detail::prepare_state(state_law, inside);
            if (!detail::state_valid(left)) return {{}, R{0}, ri::Status::InvalidState};
            const auto outside = closure(context, condition, inside, states.prepared_face, model);
            const auto right = detail::prepare_state(state_law, outside);
            if (!detail::state_valid(right)) return {{}, R{0}, ri::Status::InvalidState};
            const auto result = solver(left, right, normal);
            if (result.valid() && !ri::detail::finite(result.flux))
                return {{}, R{0}, ri::Status::NonfiniteFlux};
            return result;
        };

        const auto point = detail::primitive_array(states.trace);
        const auto base = flux(point);
        if (!base.valid()) return {};
        const auto reference = detail::conservative_array(base.flux);
        const R relative_step = Kokkos::cbrt(Kokkos::epsilon_v<R>);
        ri::Matrix5<R> trace_jacobian;
        for (std::size_t column = 0; column < 5; ++column) {
            R scale = Kokkos::abs(point[column]);
            if (column > 0 && column < 4) scale = Kokkos::max(scale, states.cell.sound_speed);
            if (!detail::differentiate_flux_column(flux, point, reference, column, relative_step * scale, trace_jacobian))
                return {};
        }

        const auto primitive_jacobian = ri::primitive_from_conservative_jacobian(states.cell, state_law.eos);
        BoundaryJacobian<R> result;
        for (std::size_t row = 0; row < 5; ++row)
            for (std::size_t column = 0; column < 5; ++column)
                for (std::size_t k = 0; k < 5; ++k)
                    result.block(row, column) += trace_jacobian(row, k) * primitive_jacobian(k, column);
        for (const auto value : result.block.values)
            if (!Kokkos::isfinite(value)) return {};
        result.valid = true;
        return result;
    }
};

} // namespace krona::transport::euler