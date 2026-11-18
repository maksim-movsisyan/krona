#pragma once

#include <krona/numerics/reconstruction/face_profiles.hpp>
#include <krona/transport/scalar/convection/types.hpp>

namespace krona::transport::scalar::convection {

/// @brief Adapts a component of a face profile to deferred scalar upwind corrections.
/// @tparam Profiles Bound interior and physical-face formulas with their shared value provider.
template <class Profiles>
struct Reconstruction {
    Profiles profiles;
    std::size_t component = 0;

    /// @brief Returns the increment on the upwind side chosen by the mass flux.
    /// @tparam Real Arithmetic type used by the transport kernel.
    template <class Real>
    KOKKOS_INLINE_FUNCTION Real interior(const InteriorFaceContext<Real>& context, bool from_owner) const {
        return profiles.template inner_increment<Real>(static_cast<std::size_t>(context.face), from_owner,
                                                       component);
    }

    /// @brief Returns the interior increment at a physical outflow face.
    /// @tparam Real Arithmetic type used by the transport kernel.
    template <class Real>
    KOKKOS_INLINE_FUNCTION Real boundary(const boundary::FaceContext<Real>& context) const {
        return profiles.template boundary_increment<Real>(static_cast<std::size_t>(context.face), component);
    }
};

/// @brief Selects the scalar component used by this transport operator.
/// @tparam Profiles Bound formulas and sampled field.
/// @param profiles Profiles to evaluate. Their arrays must stay alive through the kernel.
/// @param component Index of this scalar in the value pack.
template <class Profiles>
auto make_reconstruction(Profiles profiles, std::size_t component = 0) {
    return Reconstruction<Profiles>{profiles, component};
}

namespace detail {
template <class R>
inline constexpr bool reconstructed = !std::is_same_v<R, NoReconstruction>;

template <class Exec, class R>
void validate_reconstruction(const R&, std::size_t) {}

template <class Exec, class Profiles>
void validate_reconstruction(const Reconstruction<Profiles>& reconstruction, std::size_t cells) {
    using Values = decltype(reconstruction.profiles.values);
    mpi::require(reconstruction.component < Values::components,
                 "Scalar convection: invalid reconstruction component");
    reconstruction.profiles.template validate<Exec>(
        cells, cells, true, "Scalar convection: reconstruction input extent is too short");
}

} // namespace detail
} // namespace krona::transport::scalar::convection