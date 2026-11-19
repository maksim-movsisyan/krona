#pragma once

#include <krona/boundary/physical/detail/parameters.hpp>
#include <krona/boundary/physical/profiles.hpp>
#include <krona/mpi/error.hpp>

#include <type_traits>

namespace krona::boundary::physical::detail {

/// @brief Captures the reason for a parameter error so the host can report it through MPI.
struct ParameterError {
    mutable const char* message = nullptr;

    /// @brief Records the failure reported by the shared condition checks.
    /// @param explanation Text describing the failing parameter or model requirement.
    KOKKOS_INLINE_FUNCTION void fail(Error, const char* explanation) const { message = explanation; }
};

/// @brief Checks a binding before storing it and normalizes constant direction vectors once.
/// @tparam C Numeric condition, profiled condition or custom condition.
/// @tparam Model Boundary model that will evaluate the condition.
/// @param condition Condition to store on the patch after preparation.
/// @param model Model used for compatibility checks and optional custom validation.
template<class C, class Model>
C prepare(C condition, const Model& model) {
    if constexpr (is_profiled<C>) {
        // Provider values are available later at a specific face and time.
        if (const char* issue = incompatible_model(std::type_identity<condition_type_t<C>>{}, model))
            mpi::fatal(issue);
    } else {
        ParameterError error;
        if (!prepare_parameters(condition, model, error, true))
            mpi::fatal(error.message);
        if constexpr (requires { condition.validate(model); })
            condition.validate(model);
    }
    return condition;
}

} // namespace krona::boundary::physical::detail
