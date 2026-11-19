#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/boundary/scalar/conditions.hpp>
#include <krona/numerics/gradient/correction.hpp>
#include <krona/math/matrix3.hpp>

namespace krona::boundary::physical {

/// @brief 3 by 3 matrix acting on Cartesian velocity components.
/// @tparam Real Floating-point type of a matrix entry.
template<class Real>
using Matrix = math::Matrix3<Real>;

/// @brief Expresses face velocity and its outward normal derivative as A * owner_velocity + b.
/// @tparam Real Floating-point type of the matrices and vectors.
/// @details Slip can couple Cartesian components through the face normal. A coupled solver can use
/// the matrix coefficients directly.
template<class Real>
struct VelocityRelation {
    Matrix<Real> value_owner, gradient_owner;
    Vector<Real> value_constant, gradient_constant;

    /// @brief Evaluates the relation for the velocity at the face.
    /// @param owner Velocity at the adjacent cell centre.
    KOKKOS_INLINE_FUNCTION Vector<Real> face_value(Vector<Real> owner) const
    { return value_owner*owner + value_constant; }

    /// @brief Evaluates the outward normal derivative of velocity.
    /// @param owner Velocity at the adjacent cell centre.
    KOKKOS_INLINE_FUNCTION Vector<Real> normal_gradient(Vector<Real> owner) const
    { return gradient_owner*owner + gradient_constant; }
};

/// @brief Builds the affine relations for a prescribed no-slip wall velocity.
/// @tparam Real Floating-point type of the geometry and velocity.
/// @param c Face context with a positive projected owner-to-face distance.
/// @param wall Prescribed velocity at the face.
/// @param increment Tangential velocity correction to the owner value.
template<class Real>
KOKKOS_INLINE_FUNCTION VelocityRelation<Real> no_slip_relation(
    const FaceContext<Real>& c,Vector<Real> wall,Vector<Real> increment = {}) {
    VelocityRelation<Real> r;
    r.value_constant = wall;
    r.gradient_constant = (wall-increment)*(Real{1}/c.distance);
    for (int i=0;i<3;++i) r.gradient_owner.data[i][i] = -Real{1}/c.distance;
    return r;
}

/// @brief Builds the affine relations for a wall that constrains only the normal velocity.
/// @tparam Real Floating-point type of the geometry and velocity.
/// @param c Face context with the outward unit normal and positive projected distance.
/// @param wall Wall velocity supplying its normal component.
/// @param increment Tangential reconstruction increment to the owner value.
template<class Real>
KOKKOS_INLINE_FUNCTION VelocityRelation<Real> slip_relation(
    const FaceContext<Real>& c,Vector<Real> wall = {},Vector<Real> increment = {}) {
    VelocityRelation<Real> r;
    const auto n = normal(c);
    const Real entries[]{n.x,n.y,n.z};
    // I - n*n^T projects a velocity onto the tangent plane.
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        const Real nn = entries[i]*entries[j];
        r.value_owner.data[i][j] = (i==j?Real{1}:Real{0})-nn;
        r.gradient_owner.data[i][j] = -nn/c.distance;
    }
    r.value_constant = increment+n*(n.dot(wall)-n.dot(increment));
    r.gradient_constant = n*((n.dot(wall)-n.dot(increment))/c.distance);
    return r;
}

/// @brief Supplies the default zero vector correction for velocity.
struct ZeroVelocityIncrement {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Vector<Real> operator()(const FaceContext<Real>&) const { return {}; }
};
/// @brief Combines three scalar correction providers into a Cartesian velocity increment.
/// @tparam X Provider for the x velocity component.
/// @tparam Y Provider for the y velocity component.
/// @tparam Z Provider for the z velocity component.
template<class X,class Y,class Z>
struct VectorCorrection {
    X x; Y y; Z z;
    /// @brief Evaluates the three component corrections on the same face.
    /// @tparam Real Floating-point type of the face context and returned vector.
    /// @param c Current face geometry and time.
    template<class Real>
    KOKKOS_INLINE_FUNCTION Vector<Real> operator()(const FaceContext<Real>& c) const
    { return {static_cast<Real>(x(c)),static_cast<Real>(y(c)),static_cast<Real>(z(c))}; }
};
/// @brief Groups the corrections that extrapolate primitive variables toward the face.
/// @tparam P Scalar pressure correction.
/// @tparam U Cartesian velocity correction.
/// @tparam T Scalar temperature correction.
template<class P = numerics::gradient::NoCorrection,class U = ZeroVelocityIncrement,class T = numerics::gradient::NoCorrection>
struct Corrections { P pressure; U velocity; T temperature; };

/// @brief Builds the temperature relation for zero outward heat flux.
/// @tparam Real Floating-point type of the face relation.
/// @tparam Law Optional conversion that supplies the derivative corresponding to zero flux.
/// @param c Face context used by the relation and the flux law.
/// @param increment Tangential temperature correction for this face.
/// @param law Heat-flux conversion. The default gives a zero normal derivative.
template<class Real,class Law>
KOKKOS_INLINE_FUNCTION scalar::Relation<Real> wall_temperature_relation(
    AdiabaticWall<Real>,const FaceContext<Real>& c,Real increment,const Law& law) {
    if constexpr (scalar::FluxLaw<Law,Real>)
        return scalar::make_relation(scalar::FixedFlux<Real>{0},c,increment,law);
    else return scalar::NormalGradient<Real>{0}(c,increment);
}

/// @brief Uses the prescribed wall temperature to form the face and derivative relations.
/// @param wall Condition supplying the temperature at the face.
/// @param c Face context containing the projected distance.
/// @param increment Tangential temperature correction for this face.
template<class Real,class Law>
KOKKOS_INLINE_FUNCTION scalar::Relation<Real> wall_temperature_relation(
    IsothermalWall<Real> wall,const FaceContext<Real>& c,Real increment,const Law&)
{ return scalar::FixedValue<Real>{wall.temperature}(c,increment); }

/// @brief Converts prescribed outward heat flux to a normal derivative and builds the temperature relation.
/// @param wall Condition supplying the outward heat-flux density.
/// @param c Face context passed to the relation and heat-flux law.
/// @param increment Tangential temperature correction for this face.
/// @param law Converts flux density to an outward normal temperature derivative.
template<class Real,class Law>
KOKKOS_INLINE_FUNCTION scalar::Relation<Real> wall_temperature_relation(
    HeatFluxWall<Real> wall,const FaceContext<Real>& c,Real increment,const Law& law)
{ return scalar::make_relation(scalar::FixedFlux<Real>{wall.flux},c,increment,law); }

} // namespace krona::boundary::physical
