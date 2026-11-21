#pragma once

#include <krona/transport/scalar/detail/views.hpp>
#include <krona/transport/scalar/transient/mass.hpp>

namespace krona::transport::scalar::transient::detail {

/// @brief Leaves storage validation for custom mass providers to the caller.
/// @tparam Exec Kokkos execution space that will read the mass.
/// @tparam Mass Provider without a built-in view contract.
template<class Exec, class Mass>
void validate_mass(const Mass&, std::size_t) {}

/// @brief Checks the integrated mass view for one time level.
/// @tparam Exec Kokkos execution space that will read the mass.
/// @tparam View Rank-one view of integrated cell masses.
/// @param mass Provider for the selected time level.
/// @param count Number of owned cells.
template<class Exec, class View>
void validate_mass(const CellMass<View>& mass, std::size_t count) {
    scalar::detail::validate_view<Exec>(mass.values, count);
}

/// @brief Checks the views used to compute integrated cell masses at one time level.
/// @tparam Exec Kokkos execution space that will read the mass inputs.
/// @tparam Volumes Rank-one view of cell volumes.
/// @tparam Density Uniform density or a rank-one view of cell densities.
/// @param mass Density-volume provider for the selected time level.
/// @param count Number of owned cells.
template<class Exec, class Volumes, class Density>
void validate_mass(const DensityVolumeMass<Volumes, Density>& mass, std::size_t count) {
    scalar::detail::validate_view<Exec>(mass.volumes, count);
    if constexpr (!std::is_arithmetic_v<Density>)
        scalar::detail::validate_view<Exec>(mass.density, count);
}

/// @brief Ends the history view checks after the last accepted time level.
/// @tparam Exec Kokkos execution space used by the operator.
template<class Exec>
void validate_history(NoHistory, std::size_t) {}

/// @brief Checks the mass and scalar view at this accepted level, then visits the older levels.
/// @tparam Level TimeLevel holding this level's accepted mass and scalar.
/// @tparam Tail History of any older accepted levels.
/// @tparam Exec Kokkos execution space that will read the history.
/// @param history Accepted levels still to check.
/// @param count Number of owned cells required at each level.
template<class Exec, class Level, class Tail>
void validate_history(const History<Level, Tail>& history, std::size_t count) {
    validate_mass<Exec>(history.level.mass, count);
    scalar::detail::validate_view<Exec>(history.level.field, count);
    validate_history<Exec>(history.tail, count);
}

} // namespace krona::transport::scalar::transient::detail
