#pragma once

#include <krona/boundary/correction.hpp>
#include <krona/boundary/scalar/conditions.hpp>

#include <type_traits>

namespace krona::boundary::scalar::detail {

template<class Real, class ComponentView, class Law, class Correction>
struct FillBoundary {
    ComponentView field;
    Law law;
    Correction correction;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const
    {
        using Value = typename ComponentView::non_const_value_type;
        if constexpr (std::is_same_v<Condition, FixedValue<Real>>) {
            // Prescribed values need neither owner values nor gradient reconstruction.
            field(c.extended_index) = static_cast<Value>(condition.value);
        } else {
            const auto relation = make_relation(condition, c, static_cast<Real>(correction(c)), law);
            const Real owner_value = relation.value_owner == Real{0} ? Real{0}
                : static_cast<Real>(field(static_cast<std::size_t>(c.owner)));
            field(c.extended_index) = static_cast<Value>(relation.face_value(owner_value));
        }
    }
};

template<class Real, class Sink, class Law, class Correction>
struct EvaluateBoundary {
    Sink sink;
    Law law;
    Correction correction;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c, const Condition& condition) const
    { sink(c, make_relation(condition, c, static_cast<Real>(correction(c)), law)); }
};

} // namespace krona::boundary::scalar::detail
