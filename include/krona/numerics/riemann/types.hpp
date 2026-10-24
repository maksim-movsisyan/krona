#pragma once
 
#include <krona/execution/types.hpp>
#include <Kokkos_NumericTraits.hpp>

#include <type_traits>

namespace krona::numerics::riemann {

template<class Real = ComputeReal>
struct Vector {
    Real x=0, y=0, z=0;
    KOKKOS_INLINE_FUNCTION Vector operator+(Vector b) const { return {x+b.x,y+b.y,z+b.z}; }
    KOKKOS_INLINE_FUNCTION Vector operator-(Vector b) const { return {x-b.x,y-b.y,z-b.z}; }
    KOKKOS_INLINE_FUNCTION Vector operator-() const { return {-x,-y,-z}; }
    KOKKOS_INLINE_FUNCTION Vector operator*(Real a) const { return {a*x,a*y,a*z}; }
    KOKKOS_INLINE_FUNCTION Real dot(Vector b) const { return x*b.x+y*b.y+z*b.z; }
    KOKKOS_INLINE_FUNCTION Real norm2() const { return dot(*this); }
};

/// @brief Densities for U; mass/momentum/energy flux densities for a flux result.
template<class Real = ComputeReal>
struct Conservative {
    Real mass=0;
    Vector<Real> momentum{};
    Real energy=0;
    KOKKOS_INLINE_FUNCTION Conservative operator+(Conservative b) const
    { return {mass+b.mass,momentum+b.momentum,energy+b.energy}; }
    KOKKOS_INLINE_FUNCTION Conservative operator-(Conservative b) const
    { return {mass-b.mass,momentum-b.momentum,energy-b.energy}; }
    KOKKOS_INLINE_FUNCTION Conservative operator*(Real a) const
    { return {a*mass,momentum*a,a*energy}; }
};

/// @brief Prepared Euler state: specific internal energy, acoustic speed, global velocity.
/// EOS evaluation and field gathering are outside the numerical flux algorithms.
template<class Real = ComputeReal>
struct State {
    static_assert(std::is_floating_point_v<Real>);
    Real density=0;
    Vector<Real> velocity{};
    Real pressure=0, internal_energy=0, sound_speed=0;
};

enum class Status {
    Success, HllFallback, InvalidState, InvalidNormal, IncompatibleModel, InvalidFan, NonfiniteFlux
};

template<class Real = ComputeReal>
struct Result {
    Conservative<Real> flux{};
    Real max_signal_speed=0;
    Status status=Status::Success;
    KOKKOS_INLINE_FUNCTION bool valid() const
    { return status==Status::Success || status==Status::HllFallback; }
    KOKKOS_INLINE_FUNCTION bool used_fallback() const { return status==Status::HllFallback; }
};

template<class Real>
struct WaveSpeeds { Real left=0, right=0; };

template<class R>
KOKKOS_INLINE_FUNCTION Conservative<R> conservative(State<R> s)
{ return {s.density,s.velocity*s.density,s.density*(s.internal_energy+R{0.5}*s.velocity.norm2())}; }

/// @brief Unit-normal flux density. Orientation is left to right; no face area is included.
template<class R>
KOKKOS_INLINE_FUNCTION Conservative<R> physical_flux(State<R> s,Vector<R> normal)
{
    const R un=s.velocity.dot(normal), mass=s.density*un;
    const R energy=s.density*(s.internal_energy+R{0.5}*s.velocity.norm2());
    return {mass,s.velocity*mass+normal*s.pressure,(energy+s.pressure)*un};
}

namespace detail {
template<class R> KOKKOS_INLINE_FUNCTION bool finite(Vector<R> v)
{ return Kokkos::isfinite(v.x)&&Kokkos::isfinite(v.y)&&Kokkos::isfinite(v.z); }
template<class R> KOKKOS_INLINE_FUNCTION bool finite(Conservative<R> u)
{ return Kokkos::isfinite(u.mass)&&finite(u.momentum)&&Kokkos::isfinite(u.energy); }
template<class R> KOKKOS_INLINE_FUNCTION bool valid(State<R> s)
{
    return Kokkos::isfinite(s.density)&&s.density>R{0}&&finite(s.velocity)&&
        Kokkos::isfinite(s.pressure)&&s.pressure>R{0}&&Kokkos::isfinite(s.internal_energy)&&
        Kokkos::isfinite(s.sound_speed)&&s.sound_speed>R{0};
}
template<class R> KOKKOS_INLINE_FUNCTION Result<R> failure(Status status)
{
    const R nan=Kokkos::quiet_NaN_v<R>;
    return {{nan,{nan,nan,nan},nan},nan,status};
}
template<class R> KOKKOS_INLINE_FUNCTION bool unit(Vector<R> n)
{
    const R norm=n.norm2();
    const R tolerance=R{128}*(Kokkos::epsilon_v<R>+static_cast<R>(Kokkos::epsilon_v<GeometryReal>));
    return finite(n)&&Kokkos::isfinite(norm)&&Kokkos::abs(norm-R{1})<=tolerance;
}
template<class R> KOKKOS_INLINE_FUNCTION Status input_status(State<R> l,State<R> r,Vector<R> n)
{
    if(!unit(n))return Status::InvalidNormal;
    if(!valid(l)||!valid(r)||!finite(conservative(l))||!finite(conservative(r)))return Status::InvalidState;
    return Status::Success;
}
template<class R> KOKKOS_INLINE_FUNCTION Result<R> finish(Conservative<R> flux,R speed)
{
    if(!finite(flux)||!Kokkos::isfinite(speed)||!(speed>R{0}))return failure<R>(Status::NonfiniteFlux);
    return {flux,speed,Status::Success};
}
template<class R> KOKKOS_INLINE_FUNCTION R endpoint_speed(State<R> l,State<R> r,Vector<R> n)
{ return Kokkos::max(Kokkos::abs(l.velocity.dot(n))+l.sound_speed,Kokkos::abs(r.velocity.dot(n))+r.sound_speed); }
} // namespace detail

/// @brief Integrated face flux convenience. Signal speed remains in velocity units, not area-weighted.
template<class Solver,class R>
KOKKOS_INLINE_FUNCTION Result<R> integrated_flux(const Solver& solver,State<R> left,State<R> right,Vector<R> area_vector)
{
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

/// @brief Integrated face flux convenience. Signal speed remains in velocity units. Normal are unit, area are given independenlty
template<class Solver, class R>
KOKKOS_INLINE_FUNCTION Result<R> integrated_flux(
    const Solver& solver, State<R> left, State<R> right,
    Vector<R> normal, R area)
{
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
