#pragma once

#include <krona/transport/detail/component_pack.hpp>
#include <krona/transport/viscous/detail/face_data.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::transport::viscous {

namespace detail {

/// @brief Converts a prescribed outward heat flux using conductivity at the owner state.
template<class R, class Properties>
KOKKOS_INLINE_FUNCTION R wall_temperature_gradient(R heat, State<R> owner, const Properties& properties) {
    if (heat == R{0}) return R{0};
    if (!valid(owner)) return Kokkos::quiet_NaN_v<R>;
    const auto coefficients = properties.template evaluate<R>(owner);
    if (!coefficients.valid() || !positive(coefficients.thermal_conductivity))
        return Kokkos::quiet_NaN_v<R>;
    return -heat / coefficients.thermal_conductivity;
}

} // namespace detail

/// @brief Adapts the property model to physical boundary-slot preparation.
/// @tparam Values Readable five-component field pack ordered as p,u,v,w,T.
/// @tparam Properties Molecular viscosity and conductivity model used by the viscous operator.
/// @details The thermal trace for a prescribed flux uses conductivity at the owner cell.
/// The energy equation uses the prescribed heat flux directly.
template<class Values, class Properties>
struct WallHeatFlux {
    Values values;
    Properties properties;

    template<class R>
    KOKKOS_INLINE_FUNCTION R normal_gradient(const boundary::FaceContext<R>& context, R heat) const {
        const std::size_t cell = static_cast<std::size_t>(context.owner);
        const auto q = transport::detail::read_primitive<R>(values, cell);
        return detail::wall_temperature_gradient(heat, q, properties);
    }
};

} // namespace krona::transport::viscous
