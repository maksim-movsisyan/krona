#pragma once

#include <krona/math/vector3.hpp>

namespace krona::flow {

/// @brief The mass, momentum and total energy densities of a flow state.
/// @tparam Scalar Floating-point type used to store the densities.
template <class Scalar> 
struct Conservative {
    Scalar density{};                 ///< Mass per unit volume.
    math::Vector3<Scalar> momentum{}; ///< Momentum per unit volume, rho*u.
    Scalar total_energy_density{};    ///< Total energy per unit volume, rho*E.
};

/// @brief Density, velocity and internal energy per unit mass recovered from a flow state.
/// @tparam Scalar Floating-point type of the recovered values.
template <class Scalar> 
struct InternalEnergyState {
    Scalar density{};
    math::Vector3<Scalar> velocity{};
    Scalar internal_energy{};
};

/// @brief Forms conservative densities from rho, velocity and specific internal energy.
/// @tparam Scalar Floating-point type used for storage and arithmetic.
/// @param rho Positive mass density.
/// @param velocity Cartesian velocity.
/// @param e Internal energy per unit mass with the material's energy reference.
/// The caller supplies finite inputs and decides how to handle arithmetic overflow.
template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Conservative<Scalar>
to_conservative(Scalar rho, math::Vector3<Scalar> velocity, Scalar e) noexcept {
    const Scalar kinetic_energy = Scalar{0.5} * velocity.norm2();
    return {rho, velocity * rho, rho * (e + kinetic_energy)};
}

/// @brief Recovers velocity and specific internal energy from conservative densities.
/// @tparam Scalar Floating-point type used for storage and arithmetic.
/// @param conserved Mass, momentum and total energy per unit volume.
/// @param state Output, written only on success.
/// @return True if density is positive and the inputs and recovered values are finite.
/// The caller or its EOS decides whether the recovered internal energy is admissible.
template <class Scalar>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool
try_split_conservative(const Conservative<Scalar>& conserved, InternalEnergyState<Scalar>& state) noexcept {
    const Scalar rho = conserved.density;
    if (!(rho > Scalar{0}) || !Kokkos::isfinite(rho) || !math::isfinite(conserved.momentum) ||
        !Kokkos::isfinite(conserved.total_energy_density))
        return false;

    // Divide by density to recover values per unit mass, then remove kinetic energy from E.
    const Scalar inverse_density = Scalar{1} / rho;
    const auto velocity = conserved.momentum * inverse_density;
    const Scalar e = conserved.total_energy_density * inverse_density - Scalar{0.5} * velocity.norm2();
    if (!math::isfinite(velocity) || !Kokkos::isfinite(e))
        return false;

    state = {rho, velocity, e};
    return true;
}

} // namespace krona::flow
