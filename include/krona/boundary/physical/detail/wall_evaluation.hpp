#pragma once

#include <krona/boundary/physical/detail/parameters.hpp>
#include <krona/boundary/physical/relations.hpp>

namespace krona::boundary::physical::detail {

/// @brief Reflects the velocity component normal to a moving wall for the inviscid exterior state.
/// @tparam Real Floating-point type of the velocity.
/// @param inside Interior velocity approaching the face.
/// @param wall Wall velocity.
/// @param n Outward unit normal.
template<class Real>
KOKKOS_INLINE_FUNCTION Vector<Real> mirror_normal_velocity(Vector<Real> inside, Vector<Real> wall, Vector<Real> n) {
    return inside - n * (Real{2} * (inside - wall).dot(n));
}

/// @brief Applies a no-slip wall to the physical trace and builds its inviscid exterior state.
/// @tparam Wall AdiabaticWall, IsothermalWall or HeatFluxWall.
/// @tparam Real Floating-point type of the face state.
/// @tparam Model Flow model determining whether temperature is present.
/// @tparam Law Optional conversion from heat-flux density to normal temperature derivative.
/// @tparam Correction Supplies tangential corrections for pressure and temperature.
/// @tparam Errors Error handler for an invalid temperature relation.
/// @param wall Numeric wall condition after sampling and validation.
/// @param c Geometry of this boundary face.
/// @param inside Interior state adjacent to the face.
/// @param law Heat-flux law used by the selected thermal condition.
/// @param correction Explicit increments for the physical trace.
/// @param errors Receives a failure and lets the caller skip this face.
template<class Wall, class Real, class Model, class Law, class Correction, class Errors>
    requires (std::is_same_v<Wall, AdiabaticWall<Real>> || std::is_same_v<Wall, IsothermalWall<Real>> ||
              std::is_same_v<Wall, HeatFluxWall<Real>>)
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(Wall wall, const FaceContext<Real>& c,
    Primitive<Real, Model::thermal> inside, const Model&, const Law& law,
    const Correction& correction, const Errors& errors) {
    auto face = inside;
    face.pressure += static_cast<Real>(correction.pressure(c));
    face.velocity = wall.velocity;

    if constexpr (Model::thermal) {
        if constexpr (std::is_same_v<Wall, IsothermalWall<Real>>) {
            // The prescribed face temperature is already known without a gradient.
            face.temperature = wall.temperature;
        } else {
            const Real increment = static_cast<Real>(correction.temperature(c));
            if (!require_device(Kokkos::isfinite(increment), errors, Error::Relation,
                                "Physical boundary: invalid temperature correction"))
                return Evaluation{inside, inside};
            const auto relation = wall_temperature_relation(wall, c, increment, law);
            if (!require_device(scalar::finite(relation), errors, Error::Relation,
                                "Physical boundary: invalid wall temperature relation"))
                return Evaluation{inside, inside};
            face.temperature = relation.face_value(inside.temperature);
        }
    }

    // The inviscid state mirrors only the normal velocity relative to the moving wall.
    auto exterior = inside;
    exterior.velocity = mirror_normal_velocity(inside.velocity, wall.velocity, normal(c));
    return Evaluation{face, exterior};
}

/// @brief Keeps tangential flow free and sets the physical trace's normal speed to the wall speed.
/// @tparam Real Floating-point type of the face state.
/// @tparam Model Flow model determining whether temperature is present.
/// @tparam Law Flux law from the common evaluation interface.
/// @tparam Correction Supplies increments of the interior variables toward the face.
/// @tparam Errors Error handler from the common evaluation interface.
/// @param wall Slip-wall condition with the wall velocity.
/// @param c Geometry of the current face.
/// @param inside Interior state adjacent to the face.
/// @param correction Tangential increments for the physical trace.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(SlipWall<Real> wall, const FaceContext<Real>& c,
    Primitive<Real, Model::thermal> inside, const Model&, const Law&,
    const Correction& correction, const Errors&) {
    const auto n = normal(c);
    auto face = inside;
    const auto reconstructed = inside.velocity + correction.velocity(c);
    // Keep the interior tangential projection and replace its normal component.
    face.velocity = reconstructed - n * reconstructed.dot(n) + n * wall.velocity.dot(n);
    face.pressure += static_cast<Real>(correction.pressure(c));
    if constexpr (Model::thermal)
        face.temperature += static_cast<Real>(correction.temperature(c));

    auto exterior = inside;
    exterior.velocity = mirror_normal_velocity(inside.velocity, wall.velocity, n);
    return Evaluation{face, exterior};
}

/// @brief Uses the slip-wall closure with zero wall velocity for a symmetry plane.
/// @param c Geometry of the current face.
/// @param inside Interior state adjacent to the face.
/// @param model Flow model that determines the state shape.
/// @param law Heat-flux law passed through the common interface.
/// @param correction Tangential increments for the physical trace.
/// @param errors Error handler from the common evaluation interface.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(Symmetry, const FaceContext<Real>& c,
    Primitive<Real, Model::thermal> inside, const Model& model, const Law& law,
    const Correction& correction, const Errors& errors) {
    return evaluate_numeric(SlipWall<Real>{}, c, inside, model, law, correction, errors);
}

} // namespace krona::boundary::physical::detail
