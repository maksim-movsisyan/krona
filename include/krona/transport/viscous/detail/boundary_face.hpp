#pragma once

#include <krona/boundary/physical/evaluate.hpp>
#include <krona/math/matrix3.hpp>
#include <krona/transport/viscous/heat_flux.hpp>

namespace krona::transport::viscous::detail {

template<class Condition, class R>
inline constexpr bool no_slip_wall = std::is_same_v<Condition, physical::IsothermalWall<R>> ||
    std::is_same_v<Condition, physical::AdiabaticWall<R>> || std::is_same_v<Condition, physical::HeatFluxWall<R>>;

template<class Condition, class R>
inline constexpr bool slip_wall = std::is_same_v<Condition, physical::SlipWall<R>> ||
    std::is_same_v<Condition, physical::Symmetry>;

/// @brief Prepared boundary face data and the slopes used in wall linearization.
template<class R>
struct BoundaryFace {
    FaceData<R> data;                       ///< Face data used to evaluate the viscous flux.
    math::Matrix3<R> velocity_owner;        ///< Derivative of face velocity with respect to owner-cell velocity.
    math::Matrix3<R> normal_gradient_owner; ///< Derivative of the outward normal velocity gradient with respect to owner-cell velocity.
    R thermal_gradient_owner = 0;           ///< Derivative of the outward normal temperature gradient with respect to owner-cell temperature.
};

struct TrialErrors {
    bool* bad;
    KOKKOS_INLINE_FUNCTION void fail(physical::Error, const char*) const { *bad = true; }
    KOKKOS_INLINE_FUNCTION bool failed() const { return *bad; }
};

/// @brief Replaces just the outward normal derivative of a cell gradient.
template<class R>
KOKKOS_INLINE_FUNCTION Vector<R> wall_gradient(Vector<R> gradient, Vector<R> n, R derivative) {
    return gradient + n * ((derivative - gradient.dot(n)) / n.norm2());
}

/// @brief Resolves prescribed wall data, characteristic inlet states and extrapolated outlets.
/// @param condition Typed patch condition. Profile data are sampled at the face and current time
/// @param frozen Holds the base molecular coefficients during a finite-difference probe.
template<class R, class Condition, class Model, class Properties>
KOKKOS_INLINE_FUNCTION bool prepare_boundary(const boundary::FaceContext<R>& context, const Condition& condition,
                                             const Model& model, State<R> owner,
                                             Gradients<R> gradients, const Properties& properties, R bulk,
                                             BoundaryFace<R>& result, const Material<R>* frozen = nullptr) {
    if (!positive(context.distance) || !valid(owner) || !valid(gradients)) return false;

    const auto n = physical::normal(context);
    const Vector<R> tangent{context.tx, context.ty, context.tz};
    const auto du = gradients.velocity_increment(tangent);
    const R dt = gradients.temperature.dot(tangent);
    
    const auto numeric = physical::sample_condition(condition, context);
    using Numeric = std::remove_cvref_t<decltype(numeric)>;

    State<R> face = owner;
    Vector<R> velocity_normal{}; // dV/dn_f = (du/dn, dv/dn, dw/dn)_f
    R temperature_normal = 0;    // dT/dn_f
    R heat = 0;
    bool heat_prescribed = false;
    bool bad = false;

    if constexpr (no_slip_wall<Numeric, R>) {
        const auto velocity = physical::no_slip_relation(context, numeric.velocity, du);
        result.normal_gradient_owner = velocity.gradient_owner;
        face.velocity = velocity.face_value(owner.velocity);
        velocity_normal = velocity.normal_gradient(owner.velocity);

        if constexpr (std::is_same_v<Numeric, physical::IsothermalWall<R>>) {
            const auto thermal = physical::wall_temperature_relation(numeric, context, dt, boundary::scalar::NoFluxLaw{});
            face.temperature = thermal.face_value(owner.temperature);
            temperature_normal = thermal.normal_gradient(owner.temperature);
            result.thermal_gradient_owner = thermal.gradient_owner;
        } else {
            if constexpr (std::is_same_v<Numeric, physical::HeatFluxWall<R>>) heat = numeric.flux;
            heat_prescribed = true;
            // Reconstruct the face temperature only when evaluating material coefficients.
            if (!frozen)
                face.temperature += dt + context.distance * wall_temperature_gradient(heat, owner, properties);
        }
    } else if constexpr (slip_wall<Numeric, R>) {
        Vector<R> wall{};
        if constexpr (std::is_same_v<Numeric, physical::SlipWall<R>>) wall = numeric.velocity;
        const auto velocity = physical::slip_relation(context, wall, du);
        result.normal_gradient_owner = velocity.gradient_owner;
        result.velocity_owner = velocity.value_owner;
        face.velocity = velocity.face_value(owner.velocity);
        face.temperature += dt;
        velocity_normal = velocity.normal_gradient(owner.velocity);
        heat_prescribed = true;
    } else {
        face = physical::evaluate(condition, context, owner, model, boundary::scalar::NoFluxLaw{},
                                  physical::Corrections<>{}, TrialErrors{&bad}).face;
        if (bad) return false;

        // Outflow extrapolates velocity and temperature in the normal direction.
        // A pressure-outlet backflow uses its characteristic inlet trace.
        bool outflow = std::is_same_v<Numeric, physical::SupersonicOutlet>;
        if constexpr (std::is_same_v<Numeric, physical::PressureOutlet<R>>)
            outflow = face.velocity.dot(n) >= R{0};
        if (outflow) {
            face.velocity = owner.velocity + du;
            face.temperature = owner.temperature + dt;
            heat_prescribed = true;
        } else {
            velocity_normal = (face.velocity - owner.velocity - du) / context.distance;
            temperature_normal = (face.temperature - owner.temperature - dt) / context.distance;
        }
    }

    if (!valid(face) || !math::isfinite(velocity_normal) || !Kokkos::isfinite(heat)) return false;
    const Material<R> material = frozen ? *frozen : Material<R>{properties.template evaluate<R>(face), bulk};
    result.data = {n, face.velocity,
        {wall_gradient(gradients.u, n, velocity_normal.x), 
         wall_gradient(gradients.v, n, velocity_normal.y),
         wall_gradient(gradients.w, n, velocity_normal.z), 
         wall_gradient(gradients.temperature, n, temperature_normal)},
        material, heat, heat_prescribed, slip_wall<Numeric, R>};
    return material.molecular.valid() && valid(result.data.gradients);
}

} // namespace krona::transport::viscous::detail
