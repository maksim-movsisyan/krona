#pragma once

#include <krona/boundary/context.hpp>
#include <krona/math/vector3.hpp>

namespace krona::boundary::physical {

/// @brief Cartesian vector used for velocity, normals and directions.
/// @tparam Real Floating-point type of the components.
template<class Real>
using Vector = math::Vector3<Real>;

/// @brief Reads the outward unit normal from a face context.
/// @tparam Real Floating-point type of the face geometry.
/// @param c Context of the boundary face.
template<class Real>
KOKKOS_INLINE_FUNCTION Vector<Real> normal(const FaceContext<Real>& c) { return {c.nx, c.ny, c.nz}; }

/// @brief Primitive variables for one interior, face or exterior state.
/// @tparam Real Floating-point type of the state.
/// @tparam Thermal True when the flow model includes temperature.
template<class Real, bool Thermal>
struct Primitive;

/// @brief Mechanical state with pressure and Cartesian velocity.
/// @tparam Real Floating-point type of the state.
template<class Real>
struct Primitive<Real, false> {
    Real pressure = 0;
    Vector<Real> velocity;
};

/// @brief Thermal state with pressure, Cartesian velocity and absolute temperature.
/// @tparam Real Floating-point type of the state.
template<class Real>
struct Primitive<Real, true> {
    Real pressure = 0;
    Vector<Real> velocity;
    Real temperature = 0;
};

/// @brief The two states supplied by a physical boundary condition.
/// @tparam State Primitive variables of the flow model.
template<class State>
struct Evaluation {
    State face;     ///< Physical face trace, for example the prescribed temperature of a viscous wall.
    State exterior; ///< Exterior inviscid state used with the interior trace in the numerical flux.
};

} // namespace krona::boundary::physical
