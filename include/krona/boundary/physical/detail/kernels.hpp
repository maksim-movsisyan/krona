#pragma once

#include <krona/boundary/physical/evaluate.hpp>
#include <krona/boundary/physical/fields.hpp>

namespace krona::boundary::physical::detail {

/// @brief Reads the interior state and sends each successful face evaluation to a sink.
/// @tparam Model Default model or a provider of local models via at(context, condition).
/// @tparam Reader Supplies the interior primitive state.
/// @tparam Sink Consumes the face and exterior states and can report output errors.
/// @tparam Law Supplies the flux-to-gradient conversion for thermal walls.
/// @tparam Correction Groups the tangential increments used by the condition.
/// @tparam Flags Device view belonging to the current status batch.
template<class Model, class Reader, class Sink, class Law, class Correction, class Flags> 
struct Evaluate {
    Model model;
    Reader reader;
    Sink sink;
    Law law;
    Correction correction;
    Flags flags;

    /// @brief Evaluates the condition using the model selected for this face.
    /// @param c Face context used by model and parameter providers.
    /// @param condition Condition assigned to the face's patch.
    /// @param inside Interior trace supplied by the reader.
    /// @param errors Error handler for this face.
    template<class Real, class Condition, class State, class Errors>
    KOKKOS_INLINE_FUNCTION auto evaluate_local(const FaceContext<Real>& c, const Condition& condition,
                                               const State& inside, const Errors& errors) const {
        if constexpr (requires { model.at(c, condition); }) {
            const auto local = model.at(c, condition);
            static_assert(decltype(local)::thermal == Model::thermal &&
                              decltype(local)::compressible == Model::compressible,
                          "Physical boundary: local model capabilities must match the manager model");
            return physical::evaluate(condition, c, inside, local, law, correction, errors);
        } else
            return physical::evaluate(condition, c, inside, model, law, correction, errors);
    }

    /// @brief Runs the reader, closure and sink for one face, with local failure tracking.
    /// @param c Face geometry and time.
    /// @param condition Concrete condition selected during host dispatch.
    template<class Real, class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const {
        bool failed = false;
        const DeviceErrors<Flags> errors{flags, &failed};
        const auto inside = reader(c, condition);
        const auto result = evaluate_local(c, condition, inside, errors);
        if (!failed)
            sink(c, result, errors);
    }
};

/// @brief Adapts a user's two-argument callback to the internal sink interface.
/// @tparam Sink Device callable accepting (context, evaluation).
template<class Sink> 
struct CheckedSink {
    Sink sink;
    template<class Context, class Result, class Errors>
    KOKKOS_INLINE_FUNCTION void operator()(const Context& c, const Result& result, const Errors&) const {
        sink(c, result);
    }
};
/// @brief Stores the physical trace in the extended fields after checking every output component.
/// @tparam Fields Writable primitive-field binding.
template<class Fields> 
struct Fill {
    Fields fields;
    template<class Real, class State, class Errors>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Evaluation<State>& result,
                                           const Errors& errors) const {
        if (!fields.can_store(result.face)) {
            errors.fail(Error::Output, "Physical boundary: unrepresentable primitive output");
            return;
        }
        fields.write(c.extended_index, result.face);
    }
};

} // namespace krona::boundary::physical::detail
