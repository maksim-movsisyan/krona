#pragma once

#include <krona/mpi/error.hpp>
#include <cmath>

namespace krona::transport::properties::detail {

template<class T>
inline bool nonnegative(T value)
{ return std::isfinite(value) && value >= T{0}; }

template<class T>
inline bool positive(T value)
{ return std::isfinite(value) && value > T{0}; }

/// @brief Custom policies without parameters need not implement validate().
template<class Model>
inline void validate(const Model& model)
{
    if constexpr (requires { model.validate(); }) model.validate();
}

} // namespace krona::transport::properties::detail
