#pragma once

#include <krona/execution/types.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <type_traits>

namespace krona::numerics::viscous {

/// @brief Rows are grad(u), grad(v), grad(w); columns are x,y,z derivatives. Always 3-D.
template<class Real = ComputeReal>
struct VelocityGradient {
    static_assert(std::is_floating_point_v<Real>);
    Real du_dx=0, du_dy=0, du_dz=0;
    Real dv_dx=0, dv_dy=0, dv_dz=0;
    Real dw_dx=0, dw_dy=0, dw_dz=0;
};

template<class Real = ComputeReal, class U, class V, class W>
KOKKOS_INLINE_FUNCTION VelocityGradient<Real> velocity_gradient(const U& u,const V& v,const W& w)
{
    return {static_cast<Real>(u.x),static_cast<Real>(u.y),static_cast<Real>(u.z),
            static_cast<Real>(v.x),static_cast<Real>(v.y),static_cast<Real>(v.z),
            static_cast<Real>(w.x),static_cast<Real>(w.y),static_cast<Real>(w.z)};
}

/// @brief Viscous stress only; pressure is excluded. The tensor is symmetric.
template<class Real = ComputeReal>
struct SymmetricStress { Real xx=0, yy=0, zz=0, xy=0, xz=0, yz=0; };

/// @brief Cartesian components of tau*n, not the outward momentum flux -tau*n.
template<class Real = ComputeReal>
struct Traction { Real x=0, y=0, z=0; };

template<class Real = ComputeReal>
struct SurfaceTransport {
    Traction<Real> traction{};
    Real outward_heat_flux=0; // q*n = -lambda*grad(T)*n, per unit area.
};

enum class Status {
    Success, InvalidGradient, InvalidCoefficient, InvalidNormal, InvalidStress,
    InvalidVelocity, InvalidArea, NonfiniteResult
};

template<class Value>
struct Result {
    Value value{};
    Status status=Status::Success;
    KOKKOS_INLINE_FUNCTION bool valid() const { return status==Status::Success; }
};

} // namespace krona::numerics::viscous
