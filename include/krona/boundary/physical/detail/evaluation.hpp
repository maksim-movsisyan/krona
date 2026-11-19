#pragma once

#include <krona/boundary/physical/models.hpp>
#include <krona/boundary/physical/detail/inlet_outlet_evaluation.hpp>
#include <krona/boundary/physical/detail/wall_evaluation.hpp>

namespace krona::boundary::physical::detail {

/// @brief Evaluates an application-defined condition using the same inputs as a built-in closure.
/// @tparam Condition Device callable returning Evaluation.
/// @tparam Real Floating-point type of the face state.
/// @tparam Model Flow model available to the condition.
/// @tparam Law Heat-flux conversion supplied by the operator.
/// @tparam Correction Tangential increments supplied by the operator.
/// @tparam Errors Error handler optionally accepted by the callable.
/// @param condition Numeric custom condition, after sampling if it is profiled.
/// @param c Current boundary face and time.
/// @param inside Interior state adjacent to the face.
/// @param model EOS or material and any characteristic policy for this face.
/// @param law Explicit flux law supplied by the caller.
/// @param correction Corrections supplied by the caller.
/// @param errors Optional checked-error interface for the callable.
template<class Condition, class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(Condition condition, const FaceContext<Real>& c,
                                             Primitive<Real, Model::thermal> inside, const Model& model,
                                             const Law& law, const Correction& correction,
                                             [[maybe_unused]] const Errors& errors) {
    if constexpr (requires { condition(c, inside, model, law, correction, errors); })
        return condition(c, inside, model, law, correction, errors);
    else
        return condition(c, inside, model, law, correction);
}

} // namespace krona::boundary::physical::detail
