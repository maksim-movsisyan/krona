#pragma once

#include <krona/mpi/error.hpp>
#include <cmath>

namespace krona::transport::properties::detail {

/// @brief Checks a host parameter whose admissible range includes zero.
/// @tparam T Stored type of the configured parameter.
template<class T>
inline bool nonnegative(T value)
{ return std::isfinite(value) && value >= T{0}; }

/// @brief Checks a host parameter that must be finite and strictly greater than zero.
/// @tparam T Stored type of the configured parameter.
template<class T>
inline bool positive(T value)
{ return std::isfinite(value) && value > T{0}; }

/// @brief Calls a policy's optional host check through the MPI error wrapper.
/// @tparam Model Property or heat-capacity policy, optionally providing validate() const.
/// @param model Policy whose configured parameters should be checked.
template<class Model>
inline void validate(const Model& model) noexcept {
    if constexpr (requires { model.validate(); })
        mpi::guarded([&] { model.validate(); });
}

} // namespace krona::transport::properties::detail
