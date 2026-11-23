#pragma once

#include <krona/transport/viscous/detail/stencil_data.hpp>

namespace krona::transport::viscous::detail {

template<class Value, class R>
KOKKOS_INLINE_FUNCTION Value interpolate(Value left, Value right, R weight) {
    return left * weight + right * (R{1} - weight);
}

/// @brief Keeps the interpolated tangential gradient and enforces the cell-centre difference (gradFi_f * d = delta(Fi)).
template<class R>
KOKKOS_INLINE_FUNCTION Vector<R> corrected_gradient(Vector<R> gradient, R jump, const InnerGeometry<R>& g) {
    return gradient + g.normal * ((jump - gradient.dot(g.connection)) / g.height);
}

/// @brief Builds an owner-oriented face with centrally interpolated primitive properties.
/// @param result Receives the corrected gradients and the face values used in viscous work.
template<class R, class Properties>
KOKKOS_INLINE_FUNCTION bool prepare_inner(const InnerGeometry<R>& g, State<R> left, State<R> right,
                                          Gradients<R> gl, Gradients<R> gr, 
                                          const Properties& properties, R bulk, FaceData<R>& result) {
    if (!positive(g.height) || !valid(left) || !valid(right) || !valid(gl) || !valid(gr)) return false;

    const R w = g.owner_weight;
    const Gradients<R> gf{interpolate(gl.u, gr.u, w), 
                          interpolate(gl.v, gr.v, w),
                          interpolate(gl.w, gr.w, w), 
                          interpolate(gl.temperature, gr.temperature, w)};
    State<R> face{interpolate(left.pressure, right.pressure, w),
                  interpolate(left.velocity, right.velocity, w) + gf.velocity_increment(g.skew),
                  interpolate(left.temperature, right.temperature, w)};
    const R corrected_temperature = face.temperature + gf.temperature.dot(g.skew);
    
    // Keep a positive face temperature if the explicit skew extrapolation overreaches.
    if (positive(corrected_temperature)) face.temperature = corrected_temperature;
    if (!valid(face)) return false;

    const auto du = right.velocity - left.velocity;
    const Gradients<R> gradients{corrected_gradient(gf.u, du.x, g), 
                                 corrected_gradient(gf.v, du.y, g),
                                 corrected_gradient(gf.w, du.z, g), 
                                 corrected_gradient(gf.temperature, right.temperature - left.temperature, g)};
    result = {g.normal, face.velocity, gradients, {properties.template evaluate<R>(face), bulk}};
    return result.material.molecular.valid() && valid(gradients);
}

} // namespace krona::transport::viscous::detail
