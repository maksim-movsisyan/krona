#pragma once

#include <krona/boundary/physical/conditions.hpp>

namespace krona::boundary::physical::detail {

/// @brief Copies a parameter-free condition into the manager's numeric condition set.
/// @tparam Real Floating-point type used by the manager.
/// @param c Condition parsed from TOML.
template<class Real> Symmetry condition_cast(Symmetry c) {
    return c;
}
template<class Real> SupersonicOutlet condition_cast(SupersonicOutlet c) {
    return c;
}

// Convert each numeric condition from the TOML precision to the manager's arithmetic type.
// The manager prepares the converted condition before storing it on a patch.
template<class Real, class O> SlipWall<Real> condition_cast(SlipWall<O> c) {
    return {math::cast<Real>(c.velocity)};
}
template<class Real, class O> AdiabaticWall<Real> condition_cast(AdiabaticWall<O> c) {
    return {math::cast<Real>(c.velocity)};
}
template<class Real, class O> IsothermalWall<Real> condition_cast(IsothermalWall<O> c) {
    return {static_cast<Real>(c.temperature), math::cast<Real>(c.velocity)};
}
template<class Real, class O> HeatFluxWall<Real> condition_cast(HeatFluxWall<O> c) {
    return {static_cast<Real>(c.flux), math::cast<Real>(c.velocity)};
}
template<class Real, class O> TotalConditionsInlet<Real> condition_cast(TotalConditionsInlet<O> c) {
    return {static_cast<Real>(c.total_pressure), static_cast<Real>(c.total_temperature),
            math::cast<Real>(c.direction), c.allow_choking};
}
template<class Real, class O> SupersonicInlet<Real> condition_cast(SupersonicInlet<O> c) {
    return {static_cast<Real>(c.pressure), static_cast<Real>(c.temperature), math::cast<Real>(c.velocity),
            static_cast<Real>(c.mach), math::cast<Real>(c.direction)};
}
template<class Real, class O> VelocityInlet<Real> condition_cast(VelocityInlet<O> c) {
    return {math::cast<Real>(c.velocity), static_cast<Real>(c.temperature)};
}
template<class Real, class O> PressureOutlet<Real> condition_cast(PressureOutlet<O> c) {
    Backflow<Real> back{c.backflow.mode, condition_cast<Real>(c.backflow.reservoir),
                        math::cast<Real>(c.backflow.direction), static_cast<Real>(c.backflow.temperature)};
    return {static_cast<Real>(c.pressure), back};
}

} // namespace krona::boundary::physical::detail
