#pragma once

#include <krona/boundary/physical/state.hpp>

namespace krona::boundary::physical {

struct Symmetry {};
template<class Real = ComputeReal> struct SlipWall { Vector<Real> velocity{}; };
template<class Real = ComputeReal> struct AdiabaticWall { Vector<Real> velocity{}; };
template<class Real = ComputeReal> struct IsothermalWall { Real temperature = 0; Vector<Real> velocity{}; };
template<class Real = ComputeReal> struct HeatFluxWall { Real flux = 0; Vector<Real> velocity{}; };

template<class Real = ComputeReal>
struct TotalConditionsInlet {
    Real total_pressure = 0, total_temperature = 0;
    Vector<Real> direction{};
    bool allow_choking = false;
};

template<class Real = ComputeReal>
struct SupersonicInlet {
    Real pressure = 0, temperature = 0;
    Vector<Real> velocity{};
    Real mach = 0;                 // Zero selects explicit velocity; positive selects mach*sound_speed*direction.
    Vector<Real> direction{};
};
struct SupersonicOutlet {};

/// Velocity inlet for incompressible flow; pressure is extrapolated.
template<class Real = ComputeReal>
struct VelocityInlet { Vector<Real> velocity{}; Real temperature = 0; };

enum class BackflowMode { Reject, TotalConditions, DirectionTemperature };
template<class Real = ComputeReal>
struct Backflow {
    BackflowMode mode = BackflowMode::Reject;
    TotalConditionsInlet<Real> reservoir;
    Vector<Real> direction{};
    Real temperature = 0;
};

template<class Real = ComputeReal>
struct PressureOutlet { Real pressure = 0; Backflow<Real> backflow; };

} // namespace krona::boundary::physical
