#pragma once

#include <krona/boundary/context.hpp>

namespace krona::boundary::physical {

template<class Real>
struct Vector {
    Real x = 0, y = 0, z = 0;
    KOKKOS_INLINE_FUNCTION Vector operator+(Vector b) const { return {x+b.x, y+b.y, z+b.z}; }
    KOKKOS_INLINE_FUNCTION Vector operator-(Vector b) const { return {x-b.x, y-b.y, z-b.z}; }
    KOKKOS_INLINE_FUNCTION Vector operator*(Real a) const { return {x*a, y*a, z*a}; }
    KOKKOS_INLINE_FUNCTION Real dot(Vector b) const { return x*b.x+y*b.y+z*b.z; }
    KOKKOS_INLINE_FUNCTION Real norm2() const { return dot(*this); }
};

template<class Real>
KOKKOS_INLINE_FUNCTION Vector<Real> normal(const FaceContext<Real>& c) { return {c.nx,c.ny,c.nz}; }

template<class Real, bool Thermal>
struct Primitive;
template<class Real>
struct Primitive<Real, false> { Real pressure = 0; Vector<Real> velocity; };
template<class Real>
struct Primitive<Real, true> { Real pressure = 0; Vector<Real> velocity; Real temperature = 0; };

template<class State>
struct Evaluation {
    State face;      // Physical face trace, including the viscous wall temperature.
    State exterior;  // Exterior inviscid state; NOT a stored ghost-cell value.
};

template<class Real>
struct Conservative { Real density = 0; Vector<Real> momentum; Real total_energy = 0; };

} // namespace krona::boundary::physical
