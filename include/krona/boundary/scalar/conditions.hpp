#pragma once

#include <krona/boundary/context.hpp>

#include <concepts>

namespace krona::boundary::scalar {

/// @brief Affine face value and outward normal derivative; no constitutive coefficient or area.
template<class Real>
struct Relation {
    Real value_owner = 0, value_constant = 0;
    Real gradient_owner = 0, gradient_constant = 0;

    KOKKOS_INLINE_FUNCTION Real face_value(Real owner_value) const
    { return value_owner == Real{0} ? value_constant : value_owner*owner_value + value_constant; }

    KOKKOS_INLINE_FUNCTION Real normal_gradient(Real owner_value) const
    { return gradient_owner == Real{0} ? gradient_constant : gradient_owner*owner_value + gradient_constant; }
};

/// @brief Prescribes phi_f; the normal derivative includes the supplied explicit increment.
template<class Real = ComputeReal>
struct FixedValue {
    Real value = 0;
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const FaceContext<Real>& c, Real increment = 0) const
    {
        const Real inverse = Real{1}/c.distance;
        return {Real{0}, value, -inverse, (value - increment)*inverse};
    }
};

/// @brief Prescribes d(phi)/dn with the outward-normal sign convention.
template<class Real = ComputeReal>
struct NormalGradient {
    Real gradient = 0;
    KOKKOS_INLINE_FUNCTION Relation<Real> operator()(const FaceContext<Real>& c, Real increment = 0) const
    { return {Real{1}, c.distance*gradient + increment, Real{0}, gradient}; }
};

/// @brief Declares an outward physical flux density; conversion belongs to a supplied flux law.
template<class Real = ComputeReal>
struct FixedFlux { Real flux = 0; };

/// @brief Explicit absence of a flux-to-gradient law; sufficient for value/gradient conditions.
struct NoFluxLaw {};

template<class Law, class Real>
concept FluxLaw = requires(const Law& law, const FaceContext<Real>& c, Real flux) {
    { law.normal_gradient(c, flux) } -> std::convertible_to<Real>;
};

/// @brief Custom conditions return Relation<Real> from (face geometry, explicit increment).
template<class Condition, class Real, class Law = NoFluxLaw>
KOKKOS_INLINE_FUNCTION Relation<Real> make_relation(const Condition& condition,
    const FaceContext<Real>& c, Real increment = 0, const Law& = {})
{ return condition(c, increment); }

/// @brief Converts only this flux condition; other conditions never access the supplied law.
template<class Real, class Law = NoFluxLaw>
KOKKOS_INLINE_FUNCTION Relation<Real> make_relation(const FixedFlux<Real>& condition,
    const FaceContext<Real>& c, Real increment = 0, const Law& law = {})
{
    if constexpr (FluxLaw<Law, Real>) {
        return NormalGradient<Real>{static_cast<Real>(law.normal_gradient(c, condition.flux))}(c, increment);
    } else {
        Kokkos::abort("Scalar boundary: FIXED_FLUX requires a flux law with normal_gradient(context, flux)");
        return {};
    }
}

} // namespace krona::boundary::scalar
