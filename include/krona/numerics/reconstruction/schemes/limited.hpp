#pragma once

#include <krona/numerics/reconstruction/geometry.hpp>

namespace krona::numerics::reconstruction {

/// @brief Scales a formula's complete increment by the prepared factor of its base cell.
/// @tparam Formula Unlimited formula evaluated when the factors were calculated.
/// @tparam Factors Field accessor with one shared factor or one factor per component.
template <class Formula, class Factors>
struct Limited {
    static constexpr Geometry geometry = Formula::geometry;
    Formula formula;
    Factors factors;

    template <class Visitor>
    void visit_inputs(Visitor&& visit) const {
        if constexpr (requires { formula.visit_inputs(visit); })
            formula.visit_inputs(visit);
        visit("factor", factors, true);
    }

    template <class Real, class Side, class Values>
    KOKKOS_INLINE_FUNCTION Real increment(const Side& side, const Values& values, std::size_t component) const {
        static_assert(Factors::components == 1 || Factors::components == Values::components);
        const Real factor = static_cast<Real>(factors(side.cell, Factors::components == 1 ? 0 : component));
        KOKKOS_ASSERT(factor >= Real{0} && factor <= Real{1});
        if (factor == Real{0})
            return Real{0};
        return factor * formula.template increment<Real>(side, values, component);
    }
};

} // namespace krona::numerics::reconstruction
