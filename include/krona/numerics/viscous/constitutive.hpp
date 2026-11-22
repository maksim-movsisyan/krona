#pragma once

#include <krona/numerics/viscous/detail/math.hpp>

namespace krona::numerics::viscous {

/// @brief Isotropic 3-D Newtonian stress with dynamic viscosity mu and bulk viscosity zeta.
/// The default zeta=0 is Stokes' hypothesis.
struct NewtonianStress {
    /// @brief Evaluates tau = 2*mu*dev(sym(grad(u))) + zeta*div(u)*I.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<SymmetricStress<R>> operator()(VelocityGradient<R> gradient,
                                                                 std::type_identity_t<R> mu, 
                                                                 std::type_identity_t<R> bulk = R{0}) const {
        const auto status = detail::input_status(gradient, mu, bulk);
        if (status != Status::Success) return detail::fail_stress<R>(status);
        const auto value = detail::stress(gradient, mu, bulk);
        if (!math::isfinite(value)) return detail::fail_stress<R>(Status::NonfiniteResult);
        return {value};
    }

    /// @brief Evaluates tau*n for a supplied unit normal without applying face area.
    template<class R, class N>
    KOKKOS_INLINE_FUNCTION Result<Traction<R>> traction(VelocityGradient<R> gradient, const N& normal,
                                                        std::type_identity_t<R> mu,
                                                        std::type_identity_t<R> bulk = R{0}) const {
        const auto status = detail::input_status(gradient, mu, bulk);
        if (status != Status::Success) return detail::fail_traction<R>(status);
        if (!detail::unit<R>(normal)) return detail::fail_traction<R>(Status::InvalidNormal);
        const auto value = detail::stress(gradient, mu, bulk) * detail::components<R>(normal);
        if (!math::isfinite(value)) return detail::fail_traction<R>(Status::NonfiniteResult);
        return {value};
    }

    /// @brief Evaluates the viscous dissipation density tau:grad(u).
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> dissipation(VelocityGradient<R> g,
                                                 std::type_identity_t<R> mu, 
                                                 std::type_identity_t<R> bulk = R{0}) const {
        const auto status = detail::input_status(g, mu, bulk);
        if (status != Status::Success) return detail::fail_scalar<R>(status);

        const R theta = detail::divergence(g);
        const R third = theta / R{3};
        const SymmetricStress<R> deviator{g.du_dx - third, g.dv_dy - third, g.dw_dz - third,
            (g.du_dy + g.dv_dx) / R{2}, (g.du_dz + g.dw_dx) / R{2}, (g.dv_dz + g.dw_dy) / R{2}};
        const R value = R{2} * mu * math::double_contract(deviator, deviator) + bulk * theta * theta;
        if (!Kokkos::isfinite(value)) return detail::fail_scalar<R>(Status::NonfiniteResult);
        return {value};
    }
};

/// @brief Projects a supplied symmetric stress onto a unit normal.
template<class R, class N>
KOKKOS_INLINE_FUNCTION Result<Traction<R>> project_traction(SymmetricStress<R> stress, const N& normal) {
    if (!math::isfinite(stress)) return detail::fail_traction<R>(Status::InvalidStress);
    if (!detail::unit<R>(normal)) return detail::fail_traction<R>(Status::InvalidNormal);
    const auto value = stress * detail::components<R>(normal);
    if (!math::isfinite(value)) return detail::fail_traction<R>(Status::NonfiniteResult);
    return {value};
}

template<class R, class N>
KOKKOS_INLINE_FUNCTION Result<Traction<R>> project_traction(Result<SymmetricStress<R>> stress, const N& normal) {
    if (!stress.valid()) return detail::fail_traction<R>(stress.status);
    return project_traction(stress.value, normal);
}

/// @brief Computes q*n = -lambda*grad(T)*n.
struct FourierHeatFlux {
    template<class R = ComputeReal, class G, class N>
    KOKKOS_INLINE_FUNCTION Result<R> operator()(const G& gradient, const N& normal,
                                                std::type_identity_t<R> conductivity) const {
        if (!detail::coefficient_valid(conductivity))
            return detail::fail_scalar<R>(Status::InvalidCoefficient);
        const auto g = detail::components<R>(gradient);
        if (!math::isfinite(g)) return detail::fail_scalar<R>(Status::InvalidGradient);
        if (!detail::unit<R>(normal)) return detail::fail_scalar<R>(Status::InvalidNormal);
        return evaluate(g.dot(detail::components<R>(normal)), conductivity);
    }

    /// @brief Evaluates Fourier flux from an already projected outward derivative.
    template<class R>
    KOKKOS_INLINE_FUNCTION Result<R> from_normal_gradient(R derivative, R conductivity) const {
        if (!detail::coefficient_valid(conductivity))
            return detail::fail_scalar<R>(Status::InvalidCoefficient);
        if (!Kokkos::isfinite(derivative)) return detail::fail_scalar<R>(Status::InvalidGradient);
        return evaluate(derivative, conductivity);
    }

private:
    template<class R>
    KOKKOS_INLINE_FUNCTION static Result<R> evaluate(R derivative, R conductivity) {
        const R value = -conductivity * derivative;
        if (!Kokkos::isfinite(value)) return detail::fail_scalar<R>(Status::NonfiniteResult);
        return {value};
    }
};

/// @brief Combines traction and heat flux and propagates errors from their constitutive laws.
template<class R>
KOKKOS_INLINE_FUNCTION Result<SurfaceTransport<R>> surface_transport(Result<Traction<R>> traction,
                                                                     Result<R> heat = {}) {
    if (!traction.valid()) return detail::fail_surface<R>(traction.status);
    if (!heat.valid()) return detail::fail_surface<R>(heat.status);
    if (!math::isfinite(traction.value) || !Kokkos::isfinite(heat.value))
        return detail::fail_surface<R>(Status::NonfiniteResult);
    return {{traction.value, heat.value}};
}

} // namespace krona::numerics::viscous
