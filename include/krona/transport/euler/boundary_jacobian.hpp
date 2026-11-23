#pragma once

#include <krona/flow/state_jacobian.hpp>
#include <krona/numerics/riemann/detail/validation.hpp>
#include <krona/transport/euler/boundary.hpp>
#include <krona/transport/euler/state.hpp>
#include <krona/transport/detail/flux_difference.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace krona::transport::euler {

/// @brief Input states for the derivative of one boundary face flux.
/// @tparam R Scalar type used for flux and state calculations.
template<class R>
struct BoundaryLinearizationState {
    numerics::riemann::State<R> cell;                       ///< Owner cell state for dW/dU.
    boundary::physical::Primitive<R, true> trace;           ///< Reconstructed interior trace, or cell value after fallback.
    boundary::physical::Primitive<R, true> prepared_face;   ///< Boundary value from the field's face slot, if needed.
};

/// @brief Derivative of the outward flux per unit area with respect to owner conservative variables.
/// @tparam R Scalar type of the matrix entries.
template<class R>
struct BoundaryJacobian {
    math::Matrix5<R> block;
    bool valid = false; ///< True when the block is valid and can be assembled.
};

/// @brief Differentiates the actual boundary flux in primitive variables, then applies analytic dW_cell/dU_cell.
/// Holds the reconstruction increment and face slots fixed. Profile providers receive the same face context.
struct FiniteDifferenceBoundary {
    /// @brief Linearizes the inviscid boundary flux for a single face.
    /// @tparam R Scalar used for perturbations and the derivative.
    /// @tparam Condition Typed boundary condition from the patch manager.
    /// @tparam Model Physical model active at this face.
    /// @tparam Closure Exterior trace policy used by the residual.
    /// @tparam StateLaw Converts primitives and provides the EOS for the chain rule.
    /// @tparam Solver Riemann solver used by the residual.
    /// @param context Face geometry and evaluation time, held fixed during all probes.
    /// @param condition Condition sampled at the face during flux evaluation.
    /// @param model Physical boundary model or its face-specific data.
    /// @param closure Policy used to produce the exterior state for each perturbed trace.
    /// @param states Owner state, interior trace and optional prepared boundary value.
    /// @param state_law Primitive-to-Riemann-state conversion.
    /// @param solver Numerical flux to differentiate.
    template<class R, class Condition, class Model, class Closure, class StateLaw, class Solver>
    KOKKOS_INLINE_FUNCTION BoundaryJacobian<R> operator()(const boundary::FaceContext<R>& context,
                                                          const Condition& condition, const Model& model, const Closure& closure,
                                                          const BoundaryLinearizationState<R>& states,
                                                          const StateLaw& state_law, const Solver& solver) const {
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

        // Perturb the trace while holding its reconstruction increment fixed.
        const auto point = transport::detail::primitive_array(states.trace);
        const auto base = flux(point);
        if (!base.valid()) return {};
        const auto reference = transport::detail::flux_array(base.flux);
        const R relative_step = Kokkos::cbrt(Kokkos::epsilon_v<R>);
        math::Matrix5<R> trace_jacobian;
        for (std::size_t column = 0; column < 5; ++column) {
            R scale = Kokkos::abs(point[column]);
            if (column > 0 && column < 4) scale = Kokkos::max(scale, states.cell.sound_speed);
            if (!transport::detail::differentiate_flux_column(flux, point, reference, column, relative_step * scale, trace_jacobian))
                return {};
        }

        // Chain rule: dF/dU = (dF/dW_trace) (dW_cell/dU_cell).
        const flow::InternalEnergyState<R> cell{states.cell.density, states.cell.velocity, states.cell.internal_energy};
        const auto primitive_jacobian = flow::primitive_from_conservative_jacobian(cell, state_law.eos);
        BoundaryJacobian<R> result{trace_jacobian * primitive_jacobian};
        for (const auto value : result.block.values)
            if (!Kokkos::isfinite(value)) return {};
        result.valid = true;
        return result;
    }
};

namespace detail {
// Preserve the prepared state for user-defined Jacobians, which can read it directly.
template<class Closure, class Linearization>
inline constexpr bool jacobian_uses_prepared_face =
    uses_prepared_face<Closure> || !std::is_same_v<Linearization, FiniteDifferenceBoundary>;
} // namespace detail

} // namespace krona::transport::euler
