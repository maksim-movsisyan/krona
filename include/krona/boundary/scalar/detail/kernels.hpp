#pragma once

#include <krona/boundary/scalar/conditions.hpp>
#include <krona/boundary/scalar/status.hpp>

namespace krona::boundary::scalar::detail {

/// @brief Evaluates a condition and writes the scalar value in the face's extended field slot.
/// @tparam Real Floating-point type used to compute the face value.
/// @tparam ComponentView Writable rank-one view containing cells followed by boundary face values.
/// @tparam Law Converts prescribed flux density to an outward normal derivative when needed.
/// @tparam Correction Returns the explicit tangential correction for a face.
/// @tparam Flags Device view that records invalid relations and results for the current status batch.
template<class Real, class ComponentView, class Law, class Correction, class Flags>
struct FillBoundary {
    ComponentView field;
    Law law;
    Correction correction;
    Flags flags;

    /// @brief Writes the value for one face after checking it in both arithmetic and storage precision.
    /// @tparam Condition Concrete type assigned to this patch.
    /// @param c Geometry, time and output index of this face.
    /// @param condition Condition copied into the patch kernel.
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const {
        using Value = typename ComponentView::non_const_value_type;
        Real next;
        if constexpr (is_fixed_value<Condition>) {
            // A prescribed face value needs only one sample of the constant or profile.
            next = static_cast<Real>(profile::sample(condition.value, c));
        } else {
            const auto increment = static_cast<Real>(correction(c));
            const auto relation = make_relation(condition, c, increment, law);
            if (!Kokkos::isfinite(increment) || !finite(relation)) { flag(flags, BadRelation); return; }
            // Read the owner only when its value contributes to the face value.
            const Real owner = relation.value_owner == Real{0} ? Real{0} : static_cast<Real>(field(static_cast<std::size_t>(c.owner)));
            next = relation.face_value(owner);
        }
        const Value stored = static_cast<Value>(next);
        if (!Kokkos::isfinite(next) || !Kokkos::isfinite(stored)) { flag(flags, BadOutput); return; }
        field(c.extended_index) = stored;
    }
};

/// @brief Gives the caller the two affine relations for assembling or evaluating a boundary term.
/// @tparam Real Floating-point type of the face context and relation.
/// @tparam Sink Device-callable consumer of (context, relation).
/// @tparam Law Converts prescribed flux density to an outward normal derivative when needed.
/// @tparam Correction Returns the explicit tangential correction for a face.
/// @tparam Flags Device view that records invalid relations for the current status batch.
template<class Real, class Sink, class Law, class Correction, class Flags>
struct EvaluateBoundary {
    Sink sink;
    Law law;
    Correction correction;
    Flags flags;

    /// @brief Passes this face to the sink when its correction and relation coefficients are finite.
    /// @tparam Condition Concrete type assigned to this patch.
    /// @param c Geometry and time of this face.
    /// @param condition Condition copied into the patch kernel.
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const {
        // Even a prescribed face value needs the correction when returning its derivative.
        const Real increment = static_cast<Real>(correction(c));
        const auto relation = make_relation(condition, c, increment, law);
        if (!Kokkos::isfinite(increment) || !finite(relation)) { flag(flags, BadRelation); return; }
        sink(c, relation);
    }
};

} // namespace krona::boundary::scalar::detail
