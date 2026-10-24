#pragma once

#include <krona/numerics/viscous/detail/math.hpp>

namespace krona::numerics::viscous {

/// @brief Isotropic Newtonian law; bulk is zeta, not the second Lame viscosity zeta-2*mu/3.
struct NewtonianStress {
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<SymmetricStress<R>> operator()(VelocityGradient<R> gradient,
        std::type_identity_t<R> mu,std::type_identity_t<R> bulk=R{0}) const
    {
        const auto status=detail::input_status(gradient,mu,bulk);
        if(status!=Status::Success)return detail::fail_stress<R>(status);
        const auto value=detail::stress(gradient,mu,bulk);
        if(!detail::finite(value))return detail::fail_stress<R>(Status::NonfiniteResult);
        return {value};
    }

    /// @brief Direct projection, sharing the constitutive formula without a second stress validation.
    template<class R,class N>
    KOKKOS_INLINE_FUNCTION Result<Traction<R>> traction(VelocityGradient<R> gradient,const N& normal,
        std::type_identity_t<R> mu,std::type_identity_t<R> bulk=R{0}) const
    {
        const auto status=detail::input_status(gradient,mu,bulk);
        if(status!=Status::Success)return detail::fail_traction<R>(status);
        if(!detail::unit<R>(normal))return detail::fail_traction<R>(Status::InvalidNormal);
        const auto value=detail::project(detail::stress(gradient,mu,bulk),detail::components<R>(normal));
        if(!detail::finite(value))return detail::fail_traction<R>(Status::NonfiniteResult);
        return {value};
    }

    /// @brief Phi=tau:grad(u), a source density, not an integrated face flux.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> dissipation(VelocityGradient<R> g,
        std::type_identity_t<R> mu,std::type_identity_t<R> bulk=R{0}) const
    {
        const auto status=detail::input_status(g,mu,bulk);
        if(status!=Status::Success)return detail::fail_scalar<R>(status);
        const R theta=detail::divergence(g),third=theta/R{3};
        const R x=g.du_dx-third,y=g.dv_dy-third,z=g.dw_dz-third;
        const R xy=(g.du_dy+g.dv_dx)/R{2},xz=(g.du_dz+g.dw_dx)/R{2},yz=(g.dv_dz+g.dw_dy)/R{2};
        const R value=R{2}*mu*(x*x+y*y+z*z+R{2}*(xy*xy+xz*xz+yz*yz))+bulk*theta*theta;
        if(!Kokkos::isfinite(value))return detail::fail_scalar<R>(Status::NonfiniteResult);
        return {value};
    }
};

/// @brief Project a supplied symmetric stress. Unit normal, no face area.
template<class R,class N>
KOKKOS_INLINE_FUNCTION Result<Traction<R>> project_traction(SymmetricStress<R> stress,const N& normal)
{
    if(!detail::finite(stress))return detail::fail_traction<R>(Status::InvalidStress);
    if(!detail::unit<R>(normal))return detail::fail_traction<R>(Status::InvalidNormal);
    const auto value=detail::project(stress,detail::components<R>(normal));
    if(!detail::finite(value))return detail::fail_traction<R>(Status::NonfiniteResult);
    return {value};
}

template<class R,class N>
KOKKOS_INLINE_FUNCTION Result<Traction<R>> project_traction(Result<SymmetricStress<R>> stress,const N& normal)
{
    if(!stress.valid())return detail::fail_traction<R>(stress.status);
    return project_traction(stress.value,normal);
}

/// @brief q*n=-lambda*grad(T)*n. Arithmetic defaults to ComputeReal, independently of input storage.
struct FourierHeatFlux {
    template<class R=ComputeReal,class G,class N>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(const G& gradient,const N& normal,std::type_identity_t<R> conductivity) const
    {
        static_assert(std::is_floating_point_v<R>);
        if(!Kokkos::isfinite(conductivity)||conductivity<R{0})return detail::fail_scalar<R>(Status::InvalidCoefficient);
        const auto g=detail::components<R>(gradient);
        if(!detail::finite(g))return detail::fail_scalar<R>(Status::InvalidGradient);
        if(!detail::unit<R>(normal))return detail::fail_scalar<R>(Status::InvalidNormal);
        const R value=-conductivity*detail::dot(g,detail::components<R>(normal));
        if(!Kokkos::isfinite(value))return detail::fail_scalar<R>(Status::NonfiniteResult);
        return {value};
    }
};

/// @brief No thermal data are needed when heat transfer is omitted.
template<class R>
KOKKOS_INLINE_FUNCTION Result<SurfaceTransport<R>> surface_transport(Result<Traction<R>> traction,Result<R> heat={})
{
    if(!traction.valid())return detail::fail_surface<R>(traction.status);
    if(!heat.valid())return detail::fail_surface<R>(heat.status);
    if(!detail::finite(traction.value)||!Kokkos::isfinite(heat.value))return detail::fail_surface<R>(Status::NonfiniteResult);
    return {{traction.value,heat.value}};
}

} // namespace krona::numerics::viscous
