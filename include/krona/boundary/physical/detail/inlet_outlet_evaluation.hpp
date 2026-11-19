#pragma once

#include <krona/boundary/physical/detail/parameters.hpp>

namespace krona::boundary::physical::detail {

/// @brief Asks the model's characteristic policy to find a state entering from a reservoir.
/// @tparam Real Floating-point type of the face state.
/// @tparam Model Compressible model with an EOS and characteristic policy.
/// @tparam Errors Error handler for an unsupported or incompatible inlet.
/// @param specified Reservoir pressure, temperature and direction.
/// @param c Face context with the outward normal.
/// @param inside Adjacent state supplying the outgoing characteristic.
/// @param model Supplies the EOS and characteristic policy.
/// @param errors Receives inlet failures when the policy supports checked evaluation.
template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION Primitive<Real, Model::thermal>
inlet(TotalConditionsInlet<Real> specified, const FaceContext<Real>& c, Primitive<Real, Model::thermal> inside,
      const Model& model, [[maybe_unused]] const Errors& errors) {

    if constexpr (Model::compressible && Model::has_characteristics) {
        if constexpr (requires { model.characteristics.inlet(model.eos, c, inside, specified, errors); })
            return model.characteristics.inlet(model.eos, c, inside, specified, errors);
        else
            return model.characteristics.inlet(model.eos, c, inside, specified);
    } else {
        errors.fail(Error::Unsupported,
                    "Physical boundary: total inlet requires a compatible characteristic policy");
        return inside;
    }
}

/// @brief Handles flow entering through a pressure outlet according to its backflow policy.
/// @tparam Real Floating-point type of the face state.
/// @tparam Model Flow model determining which backflow treatment applies.
/// @tparam Errors Error handler for invalid reverse flow.
/// @param specified Outlet pressure and backflow data.
/// @param c Face context with the outward normal.
/// @param inside Adjacent state with inward normal velocity or the state that triggered backflow.
/// @param model Flow model that supplies the compressible inlet policy when needed.
/// @param errors Receives invalid direction and characteristic failures.
template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION Primitive<Real, Model::thermal>
backflow(PressureOutlet<Real> specified, const FaceContext<Real>& c, Primitive<Real, Model::thermal> inside,
         const Model& model, [[maybe_unused]] const Errors& errors) {
    if (specified.backflow.mode == BackflowMode::Reject) {
        errors.fail(Error::Characteristics, "Physical boundary: reverse flow requires explicit backflow data");
        return inside;
    }
    if constexpr (Model::compressible) {
        return inlet(specified.backflow.reservoir, c, inside, model, errors);
    } else {
        const auto n = normal(c);
        const Real mu = specified.backflow.direction.dot(n);
        if (!(mu < Real{0})) {
            errors.fail(Error::Characteristics,
                        "Physical boundary: backflow direction must point into the domain");
            return inside;
        }
        // Change the flow direction while retaining the original normal speed.
        inside.pressure = specified.pressure;
        inside.velocity = specified.backflow.direction * (inside.velocity.dot(n) / mu);
        if constexpr (Model::thermal)
            inside.temperature = specified.backflow.temperature;
        return inside;
    }
}

/// @brief Uses the reservoir inlet state as both the physical and inviscid trace.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(TotalConditionsInlet<Real> specified, const FaceContext<Real>& c,
                                             Primitive<Real, Model::thermal> inside, const Model& model,
                                             const Law&, const Correction&,
                                             [[maybe_unused]] const Errors& errors) {

    const auto face = inlet(specified, c, inside, model, errors);
    return Evaluation{face, face};
}

/// @brief Applies an outlet pressure, including extrapolation or backflow when indicated by the normal
/// velocity.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(PressureOutlet<Real> specified, const FaceContext<Real>& c,
                                             Primitive<Real, Model::thermal> inside, const Model& model,
                                             const Law&, const Correction& correction,
                                             [[maybe_unused]] const Errors& errors) {
    auto face = inside;
    const Real un = inside.velocity.dot(normal(c));
    if constexpr (Model::compressible) {
        if constexpr (Model::has_characteristics) {
            const auto qi = model.thermodynamic(inside);
            // An outward supersonic state carries all characteristics out of the domain.
            if (un >= qi.sound_speed)
                return Evaluation{inside, inside};
            if (un < Real{0})
                face = backflow(specified, c, inside, model, errors);
            else {
                if constexpr (requires {
                                  model.characteristics.outlet(model.eos, c, inside, specified.pressure,
                                                               errors);
                              })
                    face = model.characteristics.outlet(model.eos, c, inside, specified.pressure, errors);
                else
                    face = model.characteristics.outlet(model.eos, c, inside, specified.pressure);
                // A subsonic pressure closure can itself reverse the normal velocity.
                if (face.velocity.dot(normal(c)) < Real{0})
                    face = backflow(specified, c, inside, model, errors);
            }
        } else
            errors.fail(Error::Unsupported,
                        "Physical boundary: pressure outlet requires a characteristic policy");
    } else {
        if (un < Real{0})
            face = backflow(specified, c, inside, model, errors);
        else {
            face.pressure = specified.pressure;
            face.velocity = inside.velocity + correction.velocity(c);
            if constexpr (Model::thermal)
                face.temperature += static_cast<Real>(correction.temperature(c));
        }
    }

    return Evaluation{face, face};
}

/// @brief Prescribes incompressible inlet velocity and extrapolates pressure from the interior.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(VelocityInlet<Real> specified, const FaceContext<Real>& c,
                                             Primitive<Real, Model::thermal> inside, const Model&, const Law&,
                                             const Correction& correction,
                                             [[maybe_unused]] const Errors& errors) {
    if constexpr (Model::compressible)
        errors.fail(Error::Unsupported,
                    "Physical boundary: VelocityInlet currently targets incompressible flow");
    auto face = inside;
    face.pressure += static_cast<Real>(correction.pressure(c));
    face.velocity = specified.velocity;
    if constexpr (Model::thermal)
        face.temperature = specified.temperature;

    return Evaluation{face, face};
}

/// @brief Builds the prescribed supersonic state and checks that its normal velocity is supersonic inward.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(SupersonicInlet<Real> specified, const FaceContext<Real>& c,
                                             Primitive<Real, Model::thermal> inside, const Model& model,
                                             const Law&, const Correction&,
                                             [[maybe_unused]] const Errors& errors) {
    auto face = inside;
    if constexpr (Model::compressible) {
        face = {specified.pressure, specified.velocity, specified.temperature};
        if (specified.mach > Real{0})
            face.velocity = specified.direction * (specified.mach * model.thermodynamic(face).sound_speed);
        // Check the normal Mach number. An oblique total speed can be supersonic
        // even when a characteristic still leaves the domain through this face.
        if (!(face.velocity.dot(normal(c)) < -model.thermodynamic(face).sound_speed))
            errors.fail(Error::Characteristics, "Physical boundary: prescribed supersonic inlet is not "
                                                "supersonic in the inward normal direction");
    } else
        errors.fail(Error::Unsupported, "Physical boundary: supersonic inlet requires compressible flow");
    return Evaluation{face, face};
}

/// @brief Extrapolates the interior state and checks that all normal characteristics travel outward.
template<class Real, class Model, class Law, class Correction, class Errors>
KOKKOS_INLINE_FUNCTION auto
evaluate_numeric(SupersonicOutlet, const FaceContext<Real>& c, Primitive<Real, Model::thermal> inside,
                 const Model& model, const Law&, const Correction&, [[maybe_unused]] const Errors& errors) {
    if constexpr (Model::compressible) {
        if (!(inside.velocity.dot(normal(c)) >= model.thermodynamic(inside).sound_speed))
            errors.fail(Error::Characteristics,
                        "Physical boundary: supersonic outlet has an incoming characteristic");
    } else
        errors.fail(Error::Unsupported, "Physical boundary: supersonic outlet requires compressible flow");
    return Evaluation{inside, inside};
}

} // namespace krona::boundary::physical::detail
