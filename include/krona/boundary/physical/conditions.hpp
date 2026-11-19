#pragma once

#include <krona/boundary/physical/state.hpp>

#include <variant>

namespace krona::boundary::physical {

/// @brief Sets the normal face velocity to zero and reflects it in the exterior state.
struct Symmetry {};

/// @brief Matches the wall's normal velocity and leaves the tangential flow free to slip.
/// @tparam Real Floating-point type of the wall velocity.
template<class Real = ComputeReal>
struct SlipWall {
    Vector<Real> velocity{}; ///< Wall velocity in Cartesian coordinates. The default is stationary.
};

/// @brief Prescribes the no-slip velocity and zero outward heat flux at the wall.
/// @tparam Real Floating-point type of the wall velocity.
template<class Real = ComputeReal>
struct AdiabaticWall {
    Vector<Real> velocity{}; ///< Wall velocity in Cartesian coordinates.
};

/// @brief Prescribes the no-slip velocity and temperature at the wall.
/// @tparam Real Floating-point type of the wall data.
template<class Real = ComputeReal>
struct IsothermalWall {
    Real temperature = 0;    ///< Positive absolute temperature of the physical face.
    Vector<Real> velocity{}; ///< Wall velocity in Cartesian coordinates.
};

/// @brief Prescribes the no-slip velocity and outward heat flux per unit area.
/// @tparam Real Floating-point type of the wall data.
template<class Real = ComputeReal>
struct HeatFluxWall {
    Real flux = 0;           ///< Heat-flux density, positive when heat leaves the domain.
    Vector<Real> velocity{}; ///< Wall velocity in Cartesian coordinates.
};

/// @brief Specifies reservoir conditions for a characteristic compressible inlet.
/// @tparam Real Floating-point type of the inlet data.
template<class Real = ComputeReal>
struct TotalConditionsInlet {
    Real total_pressure = 0;    ///< Positive stagnation pressure.
    Real total_temperature = 0; ///< Positive stagnation temperature.
    Vector<Real> direction{};   ///< Inward flow direction. The manager normalizes it when binding.
    bool allow_choking = false; ///< Allows the sonic fallback when the subsonic branch has no solution.
};

/// @brief Prescribes the full state entering a supersonic inlet.
/// @tparam Real Floating-point type of the inlet data.
template<class Real = ComputeReal>
struct SupersonicInlet {
    Real pressure = 0;        ///< Positive static pressure.
    Real temperature = 0;     ///< Positive static temperature.
    Vector<Real> velocity{};  ///< Prescribed Cartesian velocity when mach is zero.
    Real mach = 0;            ///< Positive Mach number for deriving speed from the local sound speed.
    Vector<Real> direction{}; ///< Inward flow direction used with a prescribed Mach number.
};

/// @brief Extrapolates the interior state when all normal characteristics travel out of the domain.
struct SupersonicOutlet {};

/// @brief Prescribes velocity for incompressible flow and extrapolates pressure.
/// @tparam Real Floating-point type of the inlet data.
template<class Real = ComputeReal>
struct VelocityInlet {
    Vector<Real> velocity{}; ///< Prescribed Cartesian velocity.
    Real temperature = 0;    ///< Prescribed absolute temperature when the model has a thermal equation.
};

/// @brief Selects what to do if flow enters through a pressure outlet.
enum class BackflowMode {
    Reject,               ///< Report reverse flow as an error.
    TotalConditions,      ///< Use a compressible reservoir and characteristic inlet.
    DirectionTemperature  ///< Use the given direction and optional temperature for incompressible flow.
};

/// @brief Data for the chosen pressure-outlet backflow treatment.
/// @tparam Real Floating-point type of the prescribed data.
template<class Real = ComputeReal>
struct Backflow {
    BackflowMode mode = BackflowMode::Reject; ///< Treatment applied when the normal velocity points inward.
    TotalConditionsInlet<Real> reservoir;    ///< Compressible reservoir for TotalConditions.
    Vector<Real> direction{};                ///< Inward direction for DirectionTemperature.
    Real temperature = 0;                    ///< Temperature for thermal incompressible backflow.
};

/// @brief Prescribes outlet pressure with an explicit policy for reverse flow.
/// @tparam Real Floating-point type of the outlet data.
template<class Real = ComputeReal>
struct PressureOutlet {
    Real pressure = 0;      ///< Target static pressure, which may be gauge pressure in incompressible flow.
    Backflow<Real> backflow; ///< Treatment and data used if flow reverses at the outlet.
};

namespace detail {

/// @brief Built-in condition types shared by the TOML decoder and the patch manager.
/// @tparam Real Floating-point type of the numeric parameters.
/// @tparam Additional Alternative types needed by the caller, such as custom conditions or CSV specs.
template<class Real, class... Additional>
using ConditionVariant = std::variant<Symmetry, SlipWall<Real>, AdiabaticWall<Real>,
    IsothermalWall<Real>, HeatFluxWall<Real>, PressureOutlet<Real>, TotalConditionsInlet<Real>,
    VelocityInlet<Real>, SupersonicInlet<Real>, SupersonicOutlet, Additional...>;

} // namespace detail

} // namespace krona::boundary::physical
