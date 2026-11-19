#pragma once

#include <krona/boundary/physical/detail/evaluation.hpp>
#include <krona/boundary/physical/profiles.hpp>

namespace krona::boundary::physical {

/// @brief Evaluates a boundary condition at one face inside the caller's kernel.
/// @tparam Condition Built-in, profiled or device-callable custom condition.
/// @tparam Real Floating-point type of the face geometry and state.
/// @tparam Model Supplies thermodynamics or material data and any characteristic policy.
/// @tparam Law Supplies heat-flux conversion when the condition needs it.
/// @tparam Correction Supplies the explicit tangential increments of the primitive variables.
/// @tparam Errors Receives failure categories and lets the caller detect whether this face failed.
/// @param condition Condition assigned to the current patch.
/// @param c Face geometry and time at which to sample providers.
/// @param inside Interior state to which the condition is applied, optionally reconstructed.
/// @param model Flow model for this face.
/// @param law Flux-to-gradient conversion for thermal wall conditions.
/// @param correction Explicit corrections used by the numeric closure.
/// @param errors Error handler. The default aborts from the kernel. Manager launches pass a batch handler.
/// @details Pass unit directions for numeric conditions evaluated directly. Profiled directions are
/// normalized after sampling. EOS and material routines keep their own domain and error contracts.
template<class Condition, class Real, class Model, class Law, class Correction,
         class Errors = detail::AbortErrors>
KOKKOS_INLINE_FUNCTION auto evaluate(const Condition& condition, const FaceContext<Real>& c,
                                     Primitive<Real, Model::thermal> inside, const Model& model, const Law& law,
                                     const Correction& correction, const Errors& errors = {}) {

    // Both constant and profiled inputs reach the numeric closure in the same form.
    auto numeric = sample_condition(condition, c);
    const auto failed = Evaluation{inside, inside};
    if (!detail::prepare_parameters(numeric, model, errors, is_profiled<Condition>) ||
        !detail::valid_state(inside, model, errors, Error::Interior))
        return failed;
    if constexpr (std::is_same_v<decltype(numeric), HeatFluxWall<Real>> && !scalar::FluxLaw<Law, Real>) {
        errors.fail(Error::Unsupported, "Physical boundary: heat-flux wall requires an explicit flux law");
        return failed;
    }
    auto result = detail::evaluate_numeric(numeric, c, inside, model, law, correction, errors);
    if (!errors.failed()) {
        (void)detail::valid_state(result.face, model, errors, Error::State);
        (void)detail::valid_state(result.exterior, model, errors, Error::State);
    }
    return result;
}

/// @brief Evaluates a single face with zero tangential corrections and the default error handler.
/// @tparam Condition Built-in, profiled or custom condition that can run without a flux law.
/// @tparam Real Floating-point type of the face context and state.
/// @tparam Model Flow model for this face.
/// @param condition Condition to evaluate.
/// @param c Geometry and time of this face.
/// @param inside Adjacent interior state.
/// @param model EOS or material and any characteristic policy.
template<class Condition, class Real, class Model>
KOKKOS_INLINE_FUNCTION auto evaluate(const Condition& condition, const FaceContext<Real>& c,
                                     Primitive<Real, Model::thermal> inside, const Model& model) {
    return evaluate(condition, c, inside, model, scalar::NoFluxLaw{}, Corrections<>{});
}

} // namespace krona::boundary::physical
