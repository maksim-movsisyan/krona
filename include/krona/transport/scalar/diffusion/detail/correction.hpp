#pragma once

#include <krona/numerics/gradient/correction.hpp>

#include <type_traits>

namespace krona::transport::scalar::diffusion::detail {

template<class C>
inline constexpr bool corrected = !std::is_same_v<std::remove_cvref_t<C>, numerics::gradient::NoCorrection>;

/// @brief Recognizes the common gradient adapter and optional custom-provider metadata.
/// A provider without metadata remains responsible for its storage and execution-space contracts.
template<class Exec, class Gradients>
void validate_gradients(const Gradients& gradients, std::size_t count) {
    if constexpr (requires { gradients.template validate<Exec>(count); }) {
        mpi::guarded([&] { gradients.template validate<Exec>(count); });
    } else {
        if constexpr (requires { typename Gradients::memory_space; })
            static_assert(Kokkos::SpaceAccessibility<Exec, typename Gradients::memory_space>::accessible);
        if constexpr (requires { gradients.size(); })
            mpi::require(gradients.size() >= count, "Scalar diffusion: gradient extent is too short");
    }
}

} // namespace krona::transport::scalar::diffusion::detail
