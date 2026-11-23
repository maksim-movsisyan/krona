#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/flow/flux.hpp>
#include <krona/numerics/viscous/flux.hpp>
#include <krona/transport/properties/types.hpp>

namespace krona::transport::viscous::detail {

namespace law = numerics::viscous;
namespace physical = boundary::physical;

template<class R> using Vector = math::Vector3<R>;
template<class R> using State = physical::Primitive<R, true>;

/// @brief Cartesian gradients at an evaluation point. Velocity component k is row k.
template<class R>
struct Gradients {
    Vector<R> u, v, w, temperature;

    KOKKOS_INLINE_FUNCTION Vector<R> velocity_increment(Vector<R> offset) const {
        return {u.dot(offset), v.dot(offset), w.dot(offset)};
    }
};

/// @brief Molecular coefficients held fixed during compact linearization.
template<class R>
struct Material {
    properties::Coefficients<R> molecular;
    R bulk = 0;
};

/// @brief Prepared face state and gradients used by the local constitutive laws.
template<class R>
struct FaceData {
    Vector<R> normal;             ///< Unit normal pointing out of the owner cell.
    Vector<R> velocity;           ///< Face velocity used to compute viscous work.
    Gradients<R> gradients;       ///< Velocity and temperature gradients at the face.
    Material<R> material;         ///< Dynamic viscosity, bulk viscosity and thermal conductivity.
    R prescribed_heat = 0;        ///< Outward heat flux per unit area, used when heat_prescribed is true.
    bool heat_prescribed = false; ///< Uses prescribed_heat directly in the energy flux.
    bool free_slip = false;       ///< Removes tangential viscous traction at slip and symmetry faces.
};

template<class R>
KOKKOS_INLINE_FUNCTION bool positive(R value) {
    return value > R{0} && Kokkos::isfinite(value);
}

template<class R>
KOKKOS_INLINE_FUNCTION bool valid(State<R> q) {
    return positive(q.pressure) && positive(q.temperature) && math::isfinite(q.velocity);
}

template<class R>
KOKKOS_INLINE_FUNCTION bool valid(Gradients<R> g) {
    return math::isfinite(g.u) && math::isfinite(g.v) && math::isfinite(g.w) && math::isfinite(g.temperature);
}

/// @brief Applies the same slip projection to a traction and to its derivative.
template<class R>
KOKKOS_INLINE_FUNCTION Vector<R> wall_traction(Vector<R> value, Vector<R> n, bool free_slip) {
    return free_slip ? n * (value.dot(n) / n.norm2()) : value;
}

template<class R>
KOKKOS_INLINE_FUNCTION auto surface(const FaceData<R>& face) {
    const auto g = face.gradients;
    const auto coefficients = face.material.molecular;
    const auto stress = law::NewtonianStress{}(
        law::velocity_gradient<R>(g.u, g.v, g.w), coefficients.dynamic_viscosity, face.material.bulk);
    if (!stress.valid())
        return law::Result<law::SurfaceTransport<R>>{{}, stress.status};

    // The face plan validates stored normals once in their geometry precision.
    const auto traction = wall_traction(stress.value * face.normal, face.normal, face.free_slip);
    const auto heat = face.heat_prescribed ? law::Result<R>{face.prescribed_heat}
        : law::FourierHeatFlux{}.from_normal_gradient(g.temperature.dot(face.normal), coefficients.thermal_conductivity);
    return law::surface_transport(law::Result<Vector<R>>{traction}, heat);
}

template<class R>
KOKKOS_INLINE_FUNCTION auto outward_flux(const FaceData<R>& face) {
    return law::outward_flux<flow::Flux<R>>(surface(face), face.velocity);
}

} // namespace krona::transport::viscous::detail
