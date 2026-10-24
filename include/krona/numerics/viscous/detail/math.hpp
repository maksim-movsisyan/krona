#pragma once

#include <krona/numerics/viscous/types.hpp>

namespace krona::numerics::viscous::detail {

template<class R,class V>
KOKKOS_INLINE_FUNCTION Traction<R> components(const V& v)
{ return {static_cast<R>(v.x),static_cast<R>(v.y),static_cast<R>(v.z)}; }

template<class R> KOKKOS_INLINE_FUNCTION bool finite(Traction<R> v)
{ return Kokkos::isfinite(v.x)&&Kokkos::isfinite(v.y)&&Kokkos::isfinite(v.z); }

template<class R> KOKKOS_INLINE_FUNCTION bool finite(SymmetricStress<R> s)
{
    return Kokkos::isfinite(s.xx)&&Kokkos::isfinite(s.yy)&&Kokkos::isfinite(s.zz)&&
           Kokkos::isfinite(s.xy)&&Kokkos::isfinite(s.xz)&&Kokkos::isfinite(s.yz);
}

template<class R> KOKKOS_INLINE_FUNCTION bool finite(VelocityGradient<R> g)
{
    return Kokkos::isfinite(g.du_dx)&&Kokkos::isfinite(g.du_dy)&&Kokkos::isfinite(g.du_dz)&&
           Kokkos::isfinite(g.dv_dx)&&Kokkos::isfinite(g.dv_dy)&&Kokkos::isfinite(g.dv_dz)&&
           Kokkos::isfinite(g.dw_dx)&&Kokkos::isfinite(g.dw_dy)&&Kokkos::isfinite(g.dw_dz);
}

template<class R> KOKKOS_INLINE_FUNCTION R dot(Traction<R> a,Traction<R> b)
{ return a.x*b.x+a.y*b.y+a.z*b.z; }

template<class R,class N>
KOKKOS_INLINE_FUNCTION bool unit(const N& normal)
{
    using Stored=std::remove_cvref_t<decltype(normal.x)>;
    static_assert(std::is_floating_point_v<Stored>);
    const auto n=components<R>(normal);
    const R norm=dot(n,n);
    const R tolerance=R{128}*(Kokkos::epsilon_v<R>+static_cast<R>(Kokkos::epsilon_v<Stored>));
    return finite(n)&&Kokkos::isfinite(norm)&&Kokkos::abs(norm-R{1})<=tolerance;
}

template<class R> KOKKOS_INLINE_FUNCTION Status input_status(VelocityGradient<R> g,R mu,R bulk)
{
    if(!Kokkos::isfinite(mu)||mu<R{0}||!Kokkos::isfinite(bulk)||bulk<R{0})return Status::InvalidCoefficient;
    if(!finite(g))return Status::InvalidGradient;
    return Status::Success;
}

template<class R> KOKKOS_INLINE_FUNCTION R divergence(VelocityGradient<R> g)
{ return g.du_dx+g.dv_dy+g.dw_dz; }

template<class R>
KOKKOS_INLINE_FUNCTION SymmetricStress<R> stress(VelocityGradient<R> g,R mu,R bulk)
{
    const R theta=divergence(g),third=theta/R{3},isotropic=bulk*theta;
    return {R{2}*mu*(g.du_dx-third)+isotropic,
            R{2}*mu*(g.dv_dy-third)+isotropic,
            R{2}*mu*(g.dw_dz-third)+isotropic,
            mu*(g.du_dy+g.dv_dx),mu*(g.du_dz+g.dw_dx),mu*(g.dv_dz+g.dw_dy)};
}

template<class R>
KOKKOS_INLINE_FUNCTION Traction<R> project(SymmetricStress<R> s,Traction<R> n)
{ return {s.xx*n.x+s.xy*n.y+s.xz*n.z,s.xy*n.x+s.yy*n.y+s.yz*n.z,s.xz*n.x+s.yz*n.y+s.zz*n.z}; }

template<class R> KOKKOS_INLINE_FUNCTION Result<R> fail_scalar(Status status)
{ return {Kokkos::quiet_NaN_v<R>,status}; }

template<class R> KOKKOS_INLINE_FUNCTION Result<Traction<R>> fail_traction(Status status)
{ const R n=Kokkos::quiet_NaN_v<R>;return {{n,n,n},status}; }

template<class R> KOKKOS_INLINE_FUNCTION Result<SymmetricStress<R>> fail_stress(Status status)
{ const R n=Kokkos::quiet_NaN_v<R>;return {{n,n,n,n,n,n},status}; }

template<class R> KOKKOS_INLINE_FUNCTION Result<SurfaceTransport<R>> fail_surface(Status status)
{ const R n=Kokkos::quiet_NaN_v<R>;return {{{n,n,n},n},status}; }

} // namespace krona::numerics::viscous::detail
