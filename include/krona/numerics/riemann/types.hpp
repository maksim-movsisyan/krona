#pragma once

#include <krona/execution/types.hpp>
#include <krona/flow/conservative.hpp>
#include <krona/flow/flux.hpp>
#include <krona/math/vector3.hpp>

#include <type_traits>

namespace krona::numerics::riemann {

/// @brief Cartesian vector in the same coordinate system as the mesh.
/// @tparam Real Scalar type of the components.
template <typename Real = ComputeReal>
using Vector = math::Vector3<Real>;


/// @brief One side of a face, ready to pass to a Riemann solver.
/// The caller supplies the velocity and the thermodynamic quantities from the EOS.
/// @tparam Real Scalar type of the prepared values.
template<class Real = ComputeReal>
struct State {
    static_assert(std::is_floating_point_v<Real>);
    Real density=0;          ///< Mass per unit volume.
    Vector<Real> velocity{}; ///< Cartesian velocity in mesh coordinates.
    Real pressure=0,         ///< Absolute static pressure.
         internal_energy=0,  ///< Internal energy per unit mass, with the EOS energy reference.
         sound_speed=0;      ///< Local speed of sound.
};

/// @brief Explains how the solver obtained its flux or why it could not compute one.
enum class Status {
    Success,           ///< Flux from the requested solver.
    HllFallback,       ///< Valid flux from HLL after the requested solver rejected its wave fan.
    InvalidState,      ///< Invalid side data or unrepresentable conservative densities.
    InvalidNormal,     ///< Invalid unit normal or face area.
    IncompatibleModel, ///< State pressure or sound speed is inconsistent with Roe's ideal gas.
    InvalidFan,        ///< Invalid wave bounds or an unusable intermediate wave fan.
    NonfiniteFlux      ///< Nonfinite flux or unusable signal speed.
};

/// @brief Flux, signal speed and outcome for a single Riemann problem.
/// A failure returned by a solver carries NaNs in the flux and signal speed.
/// @tparam Real Scalar type of the computed flux and speed.
template<class Real = ComputeReal>
struct Result {
    flow::Flux<Real> flux{};       ///< Per unit face area from a solver, or area-integrated from integrated_flux().
    Real max_signal_speed=0;       ///< Positive wave speed estimate for CFL, in velocity units.
    Status status=Status::Success; ///< Whether and how the flux was computed.

    /// @brief True whenever the result contains a usable flux, including HLL fallback.
    KOKKOS_INLINE_FUNCTION bool valid() const
    { return status==Status::Success || status==Status::HllFallback; }

    /// @brief True when HLL computed the flux in place of the requested solver.
    KOKKOS_INLINE_FUNCTION bool used_fallback() const { return status==Status::HllFallback; }
};

/// @brief Signed bounds of the wave fan along the normal. Both can have the same sign.
/// @tparam Real Scalar type of the wave speeds.
template<class Real>
struct WaveSpeeds {
    Real left=0,  ///< Lower wave speed bound.
         right=0; ///< Upper wave speed bound.
};

/// @brief Builds the conservative densities for a prepared state.
/// @tparam R Scalar type of the state and result.
/// @param s State with specific internal energy. The total energy also includes kinetic energy.
template<class R>
KOKKOS_INLINE_FUNCTION flow::Conservative<R> conservative(State<R> s)
{ return flow::to_conservative(s.density, s.velocity, s.internal_energy); }

/// @brief Converts a jump in conservative densities to a flux contribution per unit area.
/// @tparam R Scalar type of the jump, speed and resulting flux.
/// @param jump Change in density, momentum density and total energy density across the wave.
/// @param speed Signed propagation speed along the face normal.
template<class R>
KOKKOS_INLINE_FUNCTION flow::Flux<R> wave_flux(flow::Conservative<R> jump, R speed)
{ return {speed * jump.density, jump.momentum * speed, speed * jump.total_energy_density}; }

/// @brief Computes the physical Euler flux per unit area for a single state.
/// This helper assumes the state and geometry are already valid.
/// @tparam R Scalar type used to evaluate the flux.
/// @param s Prepared Euler state.
/// @param normal Unit normal pointing from the left side to the right side.
template<class R>
KOKKOS_INLINE_FUNCTION flow::Flux<R> physical_flux(State<R> s, Vector<R> normal) {
    const R un=s.velocity.dot(normal), mass=s.density*un;
    const R energy=s.density*(s.internal_energy+R{0.5}*s.velocity.norm2());
    return {mass,s.velocity*mass+normal*s.pressure,(energy+s.pressure)*un};
}

} // namespace krona::numerics::riemann