#pragma once

#include <krona/boundary/detail/config_binding.hpp>
#include <krona/boundary/scalar/conditions.hpp>
#include <krona/boundary/scalar/config.hpp>

#include <type_traits>
#include <utility>
#include <variant>

namespace krona::boundary::scalar::detail {

/// @brief Checks the condition's host metadata before using it on a patch.
/// @tparam Exec Kokkos execution space that will read any provider data.
/// @tparam C Built-in condition or custom condition with an optional validate<Exec>(range) method.
/// @param condition Condition about to be bound or finalized.
/// @param range Local patch face range expected by a face-indexed provider.
template<class Exec, class C>
void validate_condition(const C& condition, PatchRange range) {
    if constexpr (is_fixed_value<C>) profile::validate<Exec>(condition.value, range);
    else if constexpr (is_normal_gradient<C>) profile::validate<Exec>(condition.gradient, range);
    else if constexpr (is_fixed_flux<C>) profile::validate<Exec>(condition.flux, range);
    else if constexpr (requires { condition.template validate<Exec>(range); })
        condition.template validate<Exec>(range);
}

/// @brief Constructs the selected built-in condition from one numeric parameter.
/// @tparam Real Boundary calculation type.
/// @tparam ConditionVariant Manager's storage for a patch condition.
/// @param destination Patch slot that will hold the built-in condition.
/// @param kind Selects the meaning of the parameter.
/// @param parameter Number read from the configuration.
template<class Real, class ConditionVariant>
void assign_constant(ConditionVariant& destination, Kind kind, double parameter) {
    const Real value = boundary::detail::checked_parameter<Real>(parameter, "Scalar boundary");
    switch (kind) {
        case Kind::FixedValue: destination.template emplace<FixedValue<Real>>(FixedValue<Real>{value}); return;
        case Kind::NormalGradient: destination.template emplace<NormalGradient<Real>>(NormalGradient<Real>{value}); return;
        case Kind::FixedFlux: destination.template emplace<FixedFlux<Real>>(FixedFlux<Real>{value}); return;
    }
    mpi::fatal("Scalar boundary: unknown condition kind");
}

/// @brief Binds a constant directly or asks the caller to resolve a profile into a concrete condition.
/// @tparam Exec Kokkos execution space that will read any provider data.
/// @tparam Real Boundary calculation type.
/// @tparam ConditionVariant Manager's storage for a patch condition.
/// @tparam Resolver Host callback that supplies the resolved condition through an assignment callback.
/// @param destination Patch slot to fill.
/// @param config Condition kind and configured number or profile path.
/// @param range Local face range for validating the resolved condition.
/// @param resolver Callback invoked for profile paths.
template<class Exec, class Real, class ConditionVariant, class Resolver>
void assign_resolved(ConditionVariant& destination, const ConditionConfig& config,
                     PatchRange range, Resolver& resolver) {
    if (const auto* value = std::get_if<double>(&config.parameter)) {
        assign_constant<Real>(destination, config.kind, *value);
        return;
    }

    const auto assign = [&](auto condition) {
        using C = std::remove_cvref_t<decltype(condition)>;
        static_assert(std::is_constructible_v<ConditionVariant, C>, "Register the resolved condition type in Manager");
        validate_condition<Exec>(condition, range);
        destination.template emplace<C>(std::move(condition));
    };
    boundary::detail::resolve_profile_once(resolver, config, range, assign, "Scalar boundary");
}

} // namespace krona::boundary::scalar::detail
