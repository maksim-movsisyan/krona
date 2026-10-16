#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/boundary/scalar/conditions.hpp>
#include <krona/boundary/correction.hpp>

namespace krona::boundary::physical {

template<class Real>
struct Matrix {
    Real data[3][3]{};
    KOKKOS_INLINE_FUNCTION Vector<Real> operator*(Vector<Real> u) const
    {
        return {data[0][0]*u.x+data[0][1]*u.y+data[0][2]*u.z,
                data[1][0]*u.x+data[1][1]*u.y+data[1][2]*u.z,
                data[2][0]*u.x+data[2][1]*u.y+data[2][2]*u.z};
    }
};

/// Affine velocity value/normal derivative for mechanical walls; includes component coupling.
template<class Real>
struct VelocityRelation {
    Matrix<Real> value_owner, gradient_owner;
    Vector<Real> value_constant, gradient_constant;
    KOKKOS_INLINE_FUNCTION Vector<Real> face_value(Vector<Real> owner) const
    { return value_owner*owner+value_constant; }
    KOKKOS_INLINE_FUNCTION Vector<Real> normal_gradient(Vector<Real> owner) const
    { return gradient_owner*owner+gradient_constant; }
};

template<class Real>
KOKKOS_INLINE_FUNCTION VelocityRelation<Real> no_slip_relation(
    const FaceContext<Real>& c,Vector<Real> wall,Vector<Real> increment = {})
{
    VelocityRelation<Real> r;
    r.value_constant = wall;
    r.gradient_constant = (wall-increment)*(Real{1}/c.distance);
    for (int i=0;i<3;++i) r.gradient_owner.data[i][i] = -Real{1}/c.distance;
    return r;
}

template<class Real>
KOKKOS_INLINE_FUNCTION VelocityRelation<Real> slip_relation(
    const FaceContext<Real>& c,Vector<Real> wall = {},Vector<Real> increment = {})
{
    VelocityRelation<Real> r;
    const auto n = normal(c);
    const Real entries[]{n.x,n.y,n.z};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        const Real nn = entries[i]*entries[j];
        r.value_owner.data[i][j] = (i==j?Real{1}:Real{0})-nn;
        r.gradient_owner.data[i][j] = -nn/c.distance;
    }
    r.value_constant = increment+n*(n.dot(wall)-n.dot(increment));
    r.gradient_constant = n*((n.dot(wall)-n.dot(increment))/c.distance);
    return r;
}

struct ZeroVelocityIncrement {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Vector<Real> operator()(const FaceContext<Real>&) const { return {}; }
};
template<class X,class Y,class Z>
struct VectorCorrection {
    X x; Y y; Z z;
    template<class Real>
    KOKKOS_INLINE_FUNCTION Vector<Real> operator()(const FaceContext<Real>& c) const
    { return {static_cast<Real>(x(c)),static_cast<Real>(y(c)),static_cast<Real>(z(c))}; }
};
template<class P = boundary::NoCorrection,class U = ZeroVelocityIncrement,class T = boundary::NoCorrection>
struct Corrections { P pressure; U velocity; T temperature; };

/// Adiabatic defaults to isotropic conduction; an explicit law overrides this shortcut.
template<class Real,class Law>
KOKKOS_INLINE_FUNCTION scalar::Relation<Real> wall_temperature_relation(
    AdiabaticWall<Real>,const FaceContext<Real>& c,Real increment,const Law& law)
{
    if constexpr (scalar::FluxLaw<Law,Real>)
        return scalar::make_relation(scalar::FixedFlux<Real>{0},c,increment,law);
    else return scalar::NormalGradient<Real>{0}(c,increment);
}
template<class Real,class Law>
KOKKOS_INLINE_FUNCTION scalar::Relation<Real> wall_temperature_relation(
    IsothermalWall<Real> wall,const FaceContext<Real>& c,Real increment,const Law&)
{ return scalar::FixedValue<Real>{wall.temperature}(c,increment); }
template<class Real,class Law>
KOKKOS_INLINE_FUNCTION scalar::Relation<Real> wall_temperature_relation(
    HeatFluxWall<Real> wall,const FaceContext<Real>& c,Real increment,const Law& law)
{ return scalar::make_relation(scalar::FixedFlux<Real>{wall.flux},c,increment,law); }

} // namespace krona::boundary::physical
