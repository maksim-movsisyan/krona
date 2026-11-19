#pragma once
 
#include <krona/execution/types.hpp>
#include <krona/flow/conservative.hpp>
#include <krona/math/vector3.hpp>
#include <Kokkos_NumericTraits.hpp>

#include <type_traits>

namespace krona::numerics::riemann {

/// @brief Cartesian vector in the same coordinate system as the mesh.
/// @tparam Real Scalar type of the components.
template <typename Real = ComputeReal>
using Vector = math::Vector3<Real>;

/// @brief Mass, momentum and total energy flux through a face.
/// Solvers return flux per unit area. integrated_flux() multiplies it by the face area.
/// @tparam Real Scalar type of all five flux components.
template<class Real = ComputeReal>
struct Flux {
    Real mass=0;             ///< Mass flux.
    Vector<Real> momentum{}; ///< Cartesian momentum flux.
    Real energy=0;           ///< Total energy flux.

    /// @brief Adds corresponding entries.
    KOKKOS_INLINE_FUNCTION Flux operator+(Flux b) const
    { return {mass+b.mass,momentum+b.momentum,energy+b.energy}; }

    /// @brief Subtracts corresponding entries.
    KOKKOS_INLINE_FUNCTION Flux operator-(Flux b) const
    { return {mass-b.mass,momentum-b.momentum,energy-b.energy}; }

    /// @brief Scales all five entries by the same factor.
    KOKKOS_INLINE_FUNCTION Flux operator*(Real a) const
    { return {a*mass,momentum*a,a*energy}; }
};

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
    Flux<Real> flux{};             ///< Per unit face area from a solver, or area-integrated from integrated_flux().
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
KOKKOS_INLINE_FUNCTION Flux<R> wave_flux(flow::Conservative<R> jump, R speed)
{ return {speed * jump.density, jump.momentum * speed, speed * jump.total_energy_density}; }

/// @brief Computes the physical Euler flux per unit area for a single state.
/// This helper assumes the state and geometry are already valid.
/// @tparam R Scalar type used to evaluate the flux.
/// @param s Prepared Euler state.
/// @param normal Unit normal pointing from the left side to the right side.
template<class R>
KOKKOS_INLINE_FUNCTION Flux<R> physical_flux(State<R> s, Vector<R> normal) {
    const R un=s.velocity.dot(normal), mass=s.density*un;
    const R energy=s.density*(s.internal_energy+R{0.5}*s.velocity.norm2());
    return {mass,s.velocity*mass+normal*s.pressure,(energy+s.pressure)*un};
}

namespace detail {
template<class R> KOKKOS_INLINE_FUNCTION bool finite(Vector<R> v)
{ return Kokkos::isfinite(v.x)&&Kokkos::isfinite(v.y)&&Kokkos::isfinite(v.z); }
template<class R> KOKKOS_INLINE_FUNCTION bool finite(Flux<R> u)
{ return Kokkos::isfinite(u.mass)&&finite(u.momentum)&&Kokkos::isfinite(u.energy); }
template<class R> KOKKOS_INLINE_FUNCTION bool finite(flow::Conservative<R> u)
{ return Kokkos::isfinite(u.density)&&finite(u.momentum)&&Kokkos::isfinite(u.total_energy_density); }
/// @brief Checks the side data shared by all solvers. The sign of e depends on its reference.
template<class R> KOKKOS_INLINE_FUNCTION bool valid(State<R> s) {
    return Kokkos::isfinite(s.density)&&s.density>R{0}&&finite(s.velocity)&&
        Kokkos::isfinite(s.pressure)&&s.pressure>R{0}&&Kokkos::isfinite(s.internal_energy)&&
        Kokkos::isfinite(s.sound_speed)&&s.sound_speed>R{0};
}
template<class R> KOKKOS_INLINE_FUNCTION Result<R> failure(Status status) {
    const R nan=Kokkos::quiet_NaN_v<R>;
    return {{nan,{nan,nan,nan},nan},nan,status};
}
template<class R> KOKKOS_INLINE_FUNCTION bool unit(Vector<R> n) {
    const R norm=n.norm2();
    // A normal can come from mesh geometry stored in a different precision.
    const R tolerance=R{128}*(Kokkos::epsilon_v<R>+static_cast<R>(Kokkos::epsilon_v<GeometryReal>));
    return finite(n)&&Kokkos::isfinite(norm)&&Kokkos::abs(norm-R{1})<=tolerance;
}
/// @brief Checks the face geometry and both sides, including the conversion to densities.
template<class R> KOKKOS_INLINE_FUNCTION Status input_status(State<R> l,State<R> r,Vector<R> n) {
    if(!unit(n)) return Status::InvalidNormal;
    if(!valid(l) || !valid(r) || !finite(conservative(l)) || !finite(conservative(r))) return Status::InvalidState;
    return Status::Success;
}
/// @brief Packages the flux only when its entries and the CFL speed are usable.
template<class R> KOKKOS_INLINE_FUNCTION Result<R> finish(Flux<R> flux,R speed) {
    if(!finite(flux)||!Kokkos::isfinite(speed)||!(speed>R{0}))return failure<R>(Status::NonfiniteFlux);
    return {flux,speed,Status::Success};
}
/// @brief Takes the larger of |u dot n| + a from the two states.
template<class R> KOKKOS_INLINE_FUNCTION R endpoint_speed(State<R> l,State<R> r,Vector<R> n)
{ return Kokkos::max(Kokkos::abs(l.velocity.dot(n))+l.sound_speed,Kokkos::abs(r.velocity.dot(n))+r.sound_speed); }
} // namespace detail

/// @brief Computes a face-integrated flux from an oriented area vector.
/// The vector length gives the area. Only the flux is scaled by that area.
/// @tparam Solver Callable such as Hllc<> that returns a flux per unit area.
/// @tparam R Scalar type of the states, geometry and result.
/// @param solver Solver to apply to the normalized face direction.
/// @param left State on the side the area vector points away from.
/// @param right State on the side the area vector points towards.
/// @param area_vector Face area times the unit normal from left to right.
template<class Solver,class R>
KOKKOS_INLINE_FUNCTION Result<R> integrated_flux(const Solver& solver,State<R> left,State<R> right,Vector<R> area_vector) {
    const R area=Kokkos::sqrt(area_vector.norm2());
    if(!detail::finite(area_vector)||!Kokkos::isfinite(area)||!(area>R{0}))
        return detail::failure<R>(Status::InvalidNormal);
    auto result=solver(left,right,area_vector*(R{1}/area));
    if(result.valid()){
        result.flux=result.flux*area;
        if(!detail::finite(result.flux))return detail::failure<R>(Status::NonfiniteFlux);
    }
    return result;
}

/// @brief Computes a face-integrated flux when the unit normal and area are already available.
/// The flux is multiplied by the area. The signal speed remains in velocity units.
/// @tparam Solver Callable such as Hllc<> that returns a flux per unit area.
/// @tparam R Scalar type of the states, geometry and result.
/// @param solver Solver to apply to the supplied unit normal.
/// @param left State on the side the normal points away from.
/// @param right State on the side the normal points towards.
/// @param normal Unit normal pointing from left to right.
/// @param area Positive face area.
template<class Solver, class R>
KOKKOS_INLINE_FUNCTION Result<R> integrated_flux(
    const Solver& solver, State<R> left, State<R> right, Vector<R> normal, R area) {
    if (!Kokkos::isfinite(area) || !(area > R{0}))
        return detail::failure<R>(Status::InvalidNormal);

    auto result = solver(left, right, normal);
    if (result.valid()) {
        result.flux = result.flux * area;
        if (!detail::finite(result.flux))
            return detail::failure<R>(Status::NonfiniteFlux);
    }
    return result;
}


} // namespace krona::numerics::riemann
