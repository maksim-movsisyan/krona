#pragma once

#include <krona/boundary/physical/geometry_view.hpp>

namespace krona::boundary::physical::detail {

/// @brief Per-face host-dispatched condition; the callback controls all device-side work.
template<class Space, class Real, class Condition, class Function, class Transform>
struct ForEachFace {
    GeometryView<Space> geometry;
    Condition condition; Function function;
    Transform transform;
    Real time = 0;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    { function(transform(geometry.template context<Real>(face, time)), condition); }
};

template<class Reader, class Closure, class Sink>
struct Evaluate {
    Reader reader; Closure closure; Sink sink;
    template<class Real, class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const Context<Real>& c, const Condition& condition) const
    {
        const auto owner = reader(static_cast<std::size_t>(c.owner));
        sink(c, owner, condition(c, owner, closure), condition);
    }
};

KOKKOS_INLINE_FUNCTION void abort_evaluation(Status status)
{
    switch (status) {
        case Status::InvalidOwner: Kokkos::abort("Physical boundary: invalid owner state"); break;
        case Status::InvalidParameters: Kokkos::abort("Physical boundary: invalid parameters/direction"); break;
        case Status::InvalidThermodynamics: Kokkos::abort("Physical boundary: invalid boundary thermodynamics"); break;
        case Status::UnsupportedCondition: Kokkos::abort("Physical boundary: condition unsupported by model"); break;
        case Status::RegimeMismatch: Kokkos::abort("Physical boundary: normal Mach regime mismatch"); break;
        case Status::Backflow: Kokkos::abort("Physical boundary: backflow needs explicit inflow data"); break;
        case Status::NoSubsonicSolution: Kokkos::abort("Physical boundary: no subsonic stagnation inlet solution"); break;
        case Status::Success: break;
    }
}

template<class Space, class Fields>
struct Store {
    GeometryView<Space> geometry; Fields fields;
    template<class Real, class S, class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const Context<Real>& c, const S&,
                                           const Evaluation<S>& result, const Condition&) const
    {
        if (!result.valid()) { abort_evaluation(result.status); return; }
        fields.store(geometry.extended_index(c.face), result.state);
    }
};

} // namespace krona::boundary::physical::detail
