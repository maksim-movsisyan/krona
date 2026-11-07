#pragma once

#include <krona/boundary/scalar/conditions.hpp>
#include <krona/boundary/scalar/status.hpp>

namespace krona::boundary::scalar::detail {

template<class Real, class ComponentView, class Law, class Correction, class Flags>
struct FillBoundary {
    ComponentView field;
    Law law;
    Correction correction;
    Flags flags;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const
    {
        using Value = typename ComponentView::non_const_value_type;
        Real next;
        if constexpr (is_fixed_value<Condition>) {
            // Both constant and profiled Dirichlet data bypass owners, gradients and the flux law.
            next = static_cast<Real>(profile::sample(condition.value, c));
        } else {
            const auto increment = static_cast<Real>(correction(c));
            const auto relation = make_relation(condition, c, increment, law);
            if (!Kokkos::isfinite(increment) || !finite(relation)) { flag(flags, BadRelation); return; }
            const Real owner = relation.value_owner == Real{0} ? Real{0} : static_cast<Real>(field(static_cast<std::size_t>(c.owner)));
            next = relation.face_value(owner);
        }
        const Value stored = static_cast<Value>(next);
        if (!Kokkos::isfinite(next) || !Kokkos::isfinite(stored)) { flag(flags, BadOutput); return; }
        field(c.extended_index) = stored;
    }
};

template<class Real, class Sink, class Law, class Correction, class Flags>
struct EvaluateBoundary {
    Sink sink;
    Law law;
    Correction correction;
    Flags flags;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const
    {
        const Real increment = static_cast<Real>(correction(c));
        const auto relation = make_relation(condition, c, increment, law);
        if (!Kokkos::isfinite(increment) || !finite(relation)) { flag(flags, BadRelation); return; }
        sink(c, relation);
    }
};

} // namespace krona::boundary::scalar::detail
