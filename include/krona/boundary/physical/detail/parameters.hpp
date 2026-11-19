#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/boundary/physical/status.hpp>

#include <type_traits>

namespace krona::boundary::physical::detail {

/// @brief Reports a failed check to the caller's error handler and returns whether it passed.
/// @tparam Errors Object providing fail(category, message).
/// @param ok Result of the check.
/// @param errors Error handler for the current face or host binding.
/// @param code Category used by batch diagnostics.
/// @param message Specific reason for callers that report the error immediately.
template<class Errors>
KOKKOS_INLINE_FUNCTION bool require_device(bool ok, const Errors& errors, Error code, const char* message) {
    if (!ok)
        errors.fail(code, message);
    return ok;
}

// These checks depend only on the condition type and model. The manager can run
// them for a profile at binding time, before any face values are available.
template<class C, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<C>, const Model&) {
    return nullptr;
}

template<class Real, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<IsothermalWall<Real>>, const Model&) {
    return Model::thermal ? nullptr : "Physical boundary: isothermal wall requires a temperature field";
}

template<class Real, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<HeatFluxWall<Real>>, const Model&) {
    return Model::thermal ? nullptr : "Physical boundary: heat-flux wall requires a temperature field";
}

template<class Real, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<TotalConditionsInlet<Real>>, const Model&) {
    return Model::compressible && Model::has_characteristics ? nullptr :
        "Physical boundary: total inlet requires a compatible compressible characteristic model";
}

template<class Real, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<VelocityInlet<Real>>, const Model&) {
    return Model::compressible ? "Physical boundary: VelocityInlet currently targets incompressible flow" : nullptr;
}

template<class Real, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<SupersonicInlet<Real>>, const Model&) {
    return Model::compressible ? nullptr : "Physical boundary: supersonic inlet requires compressible flow";
}

template<class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<SupersonicOutlet>, const Model&) {
    return Model::compressible ? nullptr : "Physical boundary: supersonic outlet requires compressible flow";
}

template<class Real, class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_model(std::type_identity<PressureOutlet<Real>>, const Model&) {
    if constexpr (Model::compressible)
        return Model::has_characteristics ? nullptr : "Physical boundary: pressure outlet requires a characteristic policy";
    return nullptr;
}

/// @brief Reports which configured backflow type is unavailable for the flow model.
/// @tparam Model Model defining the compressibility of the flow.
/// @param mode Configured treatment for reverse flow at a pressure outlet.
template<class Model>
KOKKOS_INLINE_FUNCTION constexpr const char* incompatible_backflow(BackflowMode mode) {
    if constexpr (Model::compressible)
        return mode == BackflowMode::DirectionTemperature ?
            "Physical boundary: compressible backflow requires total reservoir conditions" : nullptr;
    else
        return mode == BackflowMode::TotalConditions ?
            "Physical boundary: incompressible backflow uses DIRECTION_TEMPERATURE" : nullptr;
}

/// @brief Checks a finite and strictly positive pressure or temperature.
template<class Real, class Errors>
KOKKOS_INLINE_FUNCTION bool positive_parameter(Real value, const Errors& errors) {
    return require_device(Kokkos::isfinite(value) && value > Real{0}, errors, Error::Parameters,
                          "Physical boundary: require a finite positive parameter");
}

/// @brief Checks every component of a velocity or direction.
template<class Real, class Errors>
KOKKOS_INLINE_FUNCTION bool finite_vector(Vector<Real> value, const Errors& errors) {
    return require_device(math::isfinite(value), errors, Error::Parameters,
                          "Physical boundary: vector must be finite");
}

/// @brief Checks a nonzero direction and optionally turns it into a unit vector.
/// @tparam Real Arithmetic type of the direction.
/// @tparam Errors Error handler for an invalid vector.
/// @param direction Direction supplied by the condition.
/// @param errors Receives the failed parameter check.
/// @param normalize True for a new constant binding or a freshly sampled direction.
template<class Real, class Errors>
KOKKOS_INLINE_FUNCTION bool direction_parameter(Vector<Real>& direction, const Errors& errors, bool normalize) {
    if (!finite_vector(direction, errors))
        return false;

    const Real length2 = direction.norm2();
    if (!require_device(Kokkos::isfinite(length2) && length2 > Real{0}, errors, Error::Parameters,
                        "Physical boundary: direction must be nonzero"))
        return false;
    if (normalize)
        direction = direction * (Real{1} / Kokkos::sqrt(length2));
    return true;
}

// Numeric rules are shared by constant bindings and per-face evaluation. A
// custom condition can check its own data, including values read in its closure.
template<class C, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(C&, const Model&, const Errors&, bool) {
    return true;
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(SlipWall<Real>& c, const Model&, const Errors& errors, bool) {
    return finite_vector(c.velocity, errors);
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(AdiabaticWall<Real>& c, const Model&, const Errors& errors, bool) {
    return finite_vector(c.velocity, errors);
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(IsothermalWall<Real>& c, const Model&, const Errors& errors, bool) {
    return finite_vector(c.velocity, errors) && positive_parameter(c.temperature, errors);
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(HeatFluxWall<Real>& c, const Model&, const Errors& errors, bool) {
    return finite_vector(c.velocity, errors) &&
        require_device(Kokkos::isfinite(c.flux), errors, Error::Parameters,
                       "Physical boundary: flux must be finite");
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(TotalConditionsInlet<Real>& c, const Model&, const Errors& errors,
                                             bool normalize) {
    return positive_parameter(c.total_pressure, errors) &&
        positive_parameter(c.total_temperature, errors) &&
        direction_parameter(c.direction, errors, normalize);
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(VelocityInlet<Real>& c, const Model&, const Errors& errors, bool) {
    if (!finite_vector(c.velocity, errors))
        return false;
    if constexpr (Model::thermal)
        return positive_parameter(c.temperature, errors);
    return true;
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(SupersonicInlet<Real>& c, const Model&, const Errors& errors,
                                             bool normalize) {
    if (!positive_parameter(c.pressure, errors) || !positive_parameter(c.temperature, errors) ||
        !finite_vector(c.velocity, errors) ||
        !require_device(Kokkos::isfinite(c.mach) && c.mach >= Real{0}, errors, Error::Parameters,
                        "Physical boundary: invalid prescribed Mach number"))
        return false;
    return c.mach == Real{0} || direction_parameter(c.direction, errors, normalize);
}

template<class Real, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_parameters(PressureOutlet<Real>& c, const Model& model, const Errors& errors,
                                             bool normalize) {
    if (!require_device(Kokkos::isfinite(c.pressure), errors, Error::Parameters,
                        "Physical boundary: pressure must be finite"))
        return false;
    if constexpr (Model::compressible) {
        if (!positive_parameter(c.pressure, errors))
            return false;
    }

    const auto mode = c.backflow.mode;
    if (const char* issue = incompatible_backflow<Model>(mode)) {
        errors.fail(Error::Unsupported, issue);
        return false;
    }
    switch (mode) {
        case BackflowMode::Reject:
            return true;
        case BackflowMode::TotalConditions:
            // A compressible pressure outlet already requires a characteristic policy.
            return valid_parameters(c.backflow.reservoir, model, errors, normalize);
        case BackflowMode::DirectionTemperature:
            if (!direction_parameter(c.backflow.direction, errors, normalize))
                return false;
            if constexpr (Model::thermal)
                return positive_parameter(c.backflow.temperature, errors);
            return true;
    }
    errors.fail(Error::Parameters, "Physical boundary: invalid backflow mode");
    return false;
}

/// @brief Applies the shared model and parameter checks to a numeric condition.
/// @tparam C Type of the concrete condition whose numeric values are available.
/// @tparam Model Type determining whether each condition is meaningful for this flow.
/// @tparam Errors Error handler used by the caller.
/// @param condition Numeric condition. Its direction may be normalized in place.
/// @param model Flow model that will evaluate the condition.
/// @param errors Receives the first failure.
/// @param normalize True when preparing a newly bound or sampled direction.
template<class C, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_parameters(C& condition, const Model& model, const Errors& errors,
                                              bool normalize) {
    if (const char* issue = incompatible_model(std::type_identity<C>{}, model)) {
        errors.fail(Error::Unsupported, issue);
        return false;
    }
    return valid_parameters(condition, model, errors, normalize);
}

/// @brief Checks a primitive state before thermodynamic evaluation or after a boundary closure.
/// @tparam Real Floating-point type of the state.
/// @tparam Thermal True when the state includes temperature.
/// @tparam Model Type determining whether pressure must be positive.
/// @tparam Errors Error handler used by the caller.
/// @param state Primitive variables to inspect.
/// @param errors Receives any failure.
/// @param code Distinguishes invalid interior input from an invalid result.
template<class Real, bool Thermal, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_state(const Primitive<Real, Thermal>& state, const Model&, const Errors& errors,
                                       Error code) {
    bool valid = Kokkos::isfinite(state.pressure) && math::isfinite(state.velocity);
    if constexpr (Model::compressible)
        valid = valid && state.pressure > Real{0};
    if constexpr (Thermal)
        valid = valid && Kokkos::isfinite(state.temperature) && state.temperature > Real{0};
    return require_device(valid, errors, code,
                          "Physical boundary: invalid primitive state (finite values, positive thermal T and compressible p required)");
}

} // namespace krona::boundary::physical::detail
