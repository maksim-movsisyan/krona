#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>

namespace krona::boundary::scalar {

/// @brief Geometry and explicit data for one boundary face; normal points out of owner.
template<class Real>
struct Context {
    LocalIndex face = 0, owner = 0;
    std::size_t boundary_index = 0;
    Real x = 0, y = 0, z = 0;
    Real nx = 0, ny = 0, nz = 0;
    Real distance = 0, area = 0, diffusivity = 0, time = 0;
    Real tangential_increment = 0;
};

/// @brief Affine face value and outward diffusive flux density, without face area.
/// phi_f = value_owner * phi_P + value_constant; q_n = flux_owner * phi_P + flux_constant.
template<class Real>
struct Relation {
    Real value_owner = 0, value_constant = 0;
    Real flux_owner = 0, flux_constant = 0;

    KOKKOS_INLINE_FUNCTION Real face_value(Real owner_value) const
    { return value_owner * owner_value + value_constant; }

    KOKKOS_INLINE_FUNCTION Real outward_flux(Real owner_value) const
    { return flux_owner * owner_value + flux_constant; }
};

/// @brief Prescribes phi_f; tangential correction enters the diffusive flux only.
template<class Real = ComputeReal>
struct FixedValue {
    Real value = 0;
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const Context<Real>& c) const
    {
        const Real factor = c.diffusivity / c.distance;
        return {Real{0}, value, factor, factor * (c.tangential_increment - value)};
    }
};

/// @brief Prescribes outward normal derivative d(phi)/dn, including its sign.
template<class Real = ComputeReal>
struct NormalGradient {
    Real gradient = 0;
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const Context<Real>& c) const
    {
        return {Real{1}, c.distance * gradient + c.tangential_increment,
                Real{0}, -c.diffusivity * gradient};
    }
};

/// @brief Prescribes outward flux density q_n = -D*d(phi)/dn; requires D > 0.
template<class Real = ComputeReal>
struct FixedFlux {
    Real flux = 0;
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const Context<Real>& c) const
    {
        KOKKOS_ASSERT(c.diffusivity > Real{0});
        return {Real{1}, c.tangential_increment - c.distance * flux / c.diffusivity,
                Real{0}, flux};
    }
};

/// @brief Constant face diffusivity; no allocation or device upload.
template<class Real = ComputeReal>
struct UniformDiffusivity {
    Real value = 1;
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};

/// @brief Face-indexed diffusivity; View includes all mesh faces and may be strided.
template<class View>
struct FaceDiffusivity {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t face) const { return values(face); }
};

/// @brief Orthogonal approximation: no explicit tangential gradient contribution.
struct NoCorrection {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t, Real, Real, Real) const { return Real{0}; }
};

/// @brief Uses owned cell gradients; MPI ghost gradients are not needed at exterior faces.
template<class X, class Y, class Z>
struct GradientCorrection {
    X x;
    Y y;
    Z z;
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t owner, Real tx, Real ty, Real tz) const
    {
        return tx * static_cast<Real>(x(owner)) + ty * static_cast<Real>(y(owner)) +
               tz * static_cast<Real>(z(owner));
    }
};

} // namespace krona::boundary::scalar
