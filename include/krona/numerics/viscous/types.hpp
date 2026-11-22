#pragma once

#include <krona/execution/types.hpp>
#include <krona/math/symmetric_tensor3.hpp>
#include <type_traits>

namespace krona::numerics::viscous {

/// @brief Three Cartesian velocity gradients in row order: grad(u), grad(v), grad(w).
/// @tparam Real Scalar used for the derivatives and constitutive arithmetic.
template<class Real = ComputeReal>
struct VelocityGradient {
    static_assert(std::is_floating_point_v<Real>);
    Real du_dx = 0, du_dy = 0, du_dz = 0;
    Real dv_dx = 0, dv_dy = 0, dv_dz = 0;
    Real dw_dx = 0, dw_dy = 0, dw_dz = 0;
};

/// @brief Forms the rows of the velocity gradient from three Cartesian vectors.
template<class Real = ComputeReal, class U, class V, class W>
KOKKOS_INLINE_FUNCTION VelocityGradient<Real> velocity_gradient(const U& u, const V& v, const W& w) {
    return {static_cast<Real>(u.x), static_cast<Real>(u.y), static_cast<Real>(u.z),
            static_cast<Real>(v.x), static_cast<Real>(v.y), static_cast<Real>(v.z),
            static_cast<Real>(w.x), static_cast<Real>(w.y), static_cast<Real>(w.z)};
}

/// @brief Symmetric viscous stress with the pressure term omitted.
template<class Real = ComputeReal>
using SymmetricStress = math::SymmetricTensor3<Real>;

/// @brief Traction tau*n using the supplied normal.
template<class Real = ComputeReal>
using Traction = math::Vector3<Real>;

/// @brief Momentum traction and outward Fourier heat flow per unit area.
template<class Real = ComputeReal>
struct SurfaceTransport {
    Traction<Real> traction{};
    Real outward_heat_flux = 0;
};

enum class Status {
    Success, InvalidGradient, InvalidCoefficient, InvalidNormal, InvalidStress,
    InvalidVelocity, InvalidArea, NonfiniteResult
};

/// @brief A constitutive value or an error for the calling kernel to handle.
template<class Value>
struct Result {
    Value value{};
    Status status = Status::Success;
    KOKKOS_INLINE_FUNCTION bool valid() const { return status == Status::Success; }
};

} // namespace krona::numerics::viscous
