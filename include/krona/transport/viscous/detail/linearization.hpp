#pragma once

#include <krona/flow/state_jacobian.hpp>
#include <krona/thermodynamics/state.hpp>
#include <krona/transport/viscous/detail/face_data.hpp>

namespace krona::transport::viscous::detail {

/// @brief Maps an increment of velocity normal derivative to an increment of traction.
template<class R>
KOKKOS_INLINE_FUNCTION Vector<R> traction_increment(const FaceData<R>& face, Vector<R> du_dn) {
    const auto n = face.normal;
    const R mu = face.material.molecular.dynamic_viscosity;
    const R normal_slope = face.material.bulk + mu / R{3};
    return wall_traction(du_dn * (mu * n.norm2()) + n * (normal_slope * du_dn.dot(n)), n, face.free_slip);
}

/// @brief Writes one primitive derivative column of the outward flux density.
template<class R>
KOKKOS_INLINE_FUNCTION void flux_column(math::Matrix5<R>& block, std::size_t column, const FaceData<R>& face,
                                        Vector<R> traction, Vector<R> d_traction, Vector<R> d_velocity, R d_heat) {
    block(1, column) = -d_traction.x;
    block(2, column) = -d_traction.y;
    block(3, column) = -d_traction.z;
    block(4, column) = d_heat - face.velocity.dot(d_traction) - d_velocity.dot(traction);
}

template<class R>
KOKKOS_INLINE_FUNCTION Vector<R> axis(std::size_t index) {
    return {R(index == 0), R(index == 1), R(index == 2)};
}

/// @brief Differentiates one side of the two-point term, with explicit gradients and coefficients fixed.
template<class R>
KOKKOS_INLINE_FUNCTION math::Matrix5<R> inner_primitive_block(const FaceData<R>& face, Vector<R> traction,
                                                              R normal_slope, R value_weight) {
    math::Matrix5<R> block;
    for (std::size_t k = 0; k < 3; ++k) {
        const auto e = axis<R>(k);
        flux_column(block, k + 1, face, traction, traction_increment(face, e * normal_slope),
                    e * value_weight, R{0});
    }
    block(4, 4) = -face.material.molecular.thermal_conductivity * normal_slope * face.normal.norm2();
    return block;
}

template<class R, class EOS>
KOKKOS_INLINE_FUNCTION bool conservative_block(const math::Matrix5<R>& primitive, State<R> state,
                                                const EOS& eos, math::Matrix5<R>& block) {
    const auto thermo = [&] {
        if constexpr (requires(thermodynamics::AcousticState<R>& q) {
            eos.try_acoustic_from_pT(state.pressure, state.temperature, q);
        }) {
            thermodynamics::AcousticState<R> q;
            if (!eos.try_acoustic_from_pT(state.pressure, state.temperature, q))
                return thermodynamics::AcousticState<R>{};
            return q;
        } else {
            const auto q = eos.from_pT(state.pressure, state.temperature);
            return thermodynamics::AcousticState<R>{q.pressure, q.temperature, q.density, q.internal_energy, q.sound_speed};
        }
    }();
    if (!positive(thermo.density) || !Kokkos::isfinite(thermo.internal_energy)) return false;
    const auto jacobian = flow::primitive_from_conservative_jacobian(
        flow::InternalEnergyState<R>{thermo.density, state.velocity, thermo.internal_energy}, eos);
    block = primitive * jacobian;
    for (const auto x : block.values)
        if (!Kokkos::isfinite(x)) return false;
    return true;
}

} // namespace krona::transport::viscous::detail
