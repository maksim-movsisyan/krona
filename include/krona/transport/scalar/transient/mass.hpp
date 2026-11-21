#pragma once

#include <krona/transport/scalar/status.hpp>
#include <Kokkos_NumericTraits.hpp>

#include <cstddef>
#include <type_traits>

namespace krona::transport::scalar::transient {

/// @brief Supplies the same integrated cell mass rho*V for every owned cell.
/// @tparam Real Floating-point type of the stored mass.
template<class Real = ComputeReal>
struct UniformMass {
    Real value = 1; ///< Integrated mass of each cell.

    /// @brief Returns the prescribed mass for any owned cell.
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};

/// @brief Reads integrated mass from an existing cell array. For unit density, pass cell volumes.
/// @tparam View Rank-one Kokkos view indexed by local cell ID, with at least one entry per owned cell.
template<class View>
struct CellMass {
    View values; ///< Integrated masses accessible to the execution space.

    /// @brief Returns the mass of the selected owned cell.
    /// @param cell Local index of the owned cell.
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell) const { return values(cell); }
};

/// @brief Computes a cell's integrated mass as rho*V when it is read.
/// @tparam Volumes Rank-one Kokkos view of cell volumes.
/// @tparam Density Uniform density or rank-one Kokkos view of cell densities.
template<class Volumes, class Density>
struct DensityVolumeMass {
    Volumes volumes; ///< Cell volumes, with at least one entry per owned cell.
    Density density; ///< Density at the time level to which this provider belongs.

    /// @brief Multiplies density and volume in the requested arithmetic type.
    /// The transient operator passes the same computation type for the new time and accepted history.
    /// @tparam Real Floating-point type to which both factors are converted before multiplication.
    /// @param cell Local index of the owned cell.
    /// @return Integrated mass, or NaN if either factor is nonfinite or nonpositive.
    template<class Real = ComputeReal>
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t cell) const {
        static_assert(std::is_floating_point_v<Real>);
        const Real rho = static_cast<Real>(density_at(cell));
        const Real volume = static_cast<Real>(volumes(cell));
        if (!Kokkos::isfinite(rho) || !(rho > Real{0}) ||
            !Kokkos::isfinite(volume) || !(volume > Real{0}))
            return Kokkos::quiet_NaN_v<Real>;
        return rho * volume;
    }

private:
    KOKKOS_INLINE_FUNCTION auto density_at(std::size_t cell) const {
        if constexpr (std::is_arithmetic_v<Density>)
            return density;
        else
            return density(cell);
    }
};

/// @brief Pairs the scalar of one accepted time level with the mass from that same level.
/// @tparam Mass Device-callable provider of integrated mass at the accepted time.
/// @tparam Field Rank-one Kokkos view of the accepted cell values in owned-cell order.
template<class Mass, class Field>
struct TimeLevel {
    Mass mass;   ///< Integrated cell mass at the accepted time.
    Field field; ///< Scalar values from the same accepted step.
};

namespace detail {

/// @brief Reads the result of a custom or array-backed mass provider in the operator's precision.
/// @tparam Real Arithmetic type of the transient operator.
/// @tparam Mass Device-callable provider of integrated mass.
/// @param mass Provider for the selected time level.
/// @param cell Local index of the owned cell.
template<class Real, class Mass>
KOKKOS_INLINE_FUNCTION Real mass_value(const Mass& mass, std::size_t cell) {
    return static_cast<Real>(mass(cell));
}

/// @brief Evaluates a density-volume product in the operator's precision.
/// @tparam Real Arithmetic type used to convert and multiply the factors.
/// @tparam Volumes View of cell volumes.
/// @tparam Density Uniform density or a cell density view.
/// @param mass Density-volume provider for the selected time level.
/// @param cell Local index of the owned cell.
template<class Real, class Volumes, class Density>
KOKKOS_INLINE_FUNCTION Real mass_value(const DensityVolumeMass<Volumes, Density>& mass, std::size_t cell) {
    return mass.template operator()<Real>(cell);
}

/// @brief Reads one mass and flags a nonfinite or nonpositive value for the host to report.
/// @tparam Real Arithmetic type of the transient operator.
/// @tparam Mass Device-callable provider of integrated mass.
/// @tparam Flags Writable view of the batch's diagnostic flags.
/// @param mass Provider for the time level being read.
/// @param cell Local index of the owned cell.
/// @param flags Device flags for the open assembly batch.
template<class Real, class Mass, class Flags>
KOKKOS_INLINE_FUNCTION Real read_mass(Mass mass, std::size_t cell, Flags flags) {
    const Real value = mass_value<Real>(mass, cell);
    if (!Kokkos::isfinite(value) || !(value > Real{0})) {
        scalar::detail::flag(flags, scalar::detail::BadTemporalMass);
        return Real{0};
    }
    return value;
}

/// @brief Ends the history list with a zero contribution to the right-hand side.
struct NoHistory {
    static constexpr std::size_t size = 0;

    /// @brief Returns zero once all accepted levels have contributed.
    /// @tparam Real Arithmetic type of the right-hand side.
    /// @tparam J Index after the last history weight.
    /// @tparam Weights Device array of time derivative weights.
    /// @tparam Flags Writable view of the batch's diagnostic flags.
    template<class Real, std::size_t J, class Weights, class Flags>
    KOKKOS_INLINE_FUNCTION Real rhs(std::size_t, const Weights&, Flags) const { return Real{0}; }
};

/// @brief Holds one accepted time level followed by all older accepted levels.
/// @tparam Level TimeLevel holding this level's mass provider and scalar view.
/// @tparam Tail History of the older levels, ending in NoHistory.
template<class Level, class Tail>
struct History {
    static constexpr std::size_t size = Tail::size + 1;
    Level level; ///< First accepted level in this part of the list.
    Tail tail;   ///< Remaining accepted levels, each earlier than level.

    /// @brief Computes this cell's RHS contribution from this accepted level and the older ones.
    /// @tparam Real Arithmetic type of the mass, scalar and RHS sum.
    /// @tparam J Weight index for level. Index 1 is t^n, index 2 is t^(n-1).
    /// @tparam Weights Device array of time derivative weights, with t^(n+1) at index 0.
    /// @tparam Flags Writable view of the batch's diagnostic flags.
    /// @param cell Local index of the owned cell.
    /// @param weights Weights for the new time and every accepted level used by the formula.
    /// @param flags Device flags for the open assembly batch.
    template<class Real, std::size_t J, class Weights, class Flags>
    KOKKOS_INLINE_FUNCTION Real rhs(std::size_t cell, const Weights& weights, Flags flags) const {
        const Real mass = read_mass<Real>(level.mass, cell, flags);
        const Real value = static_cast<Real>(level.field(cell));
        if (!Kokkos::isfinite(value)) {
            scalar::detail::flag(flags, scalar::detail::BadTemporalState);
            return Real{0};
        }

        // Move the known derivative terms to the RHS, starting with this accepted level.
        const Real result = -weights[J] * mass * value + tail.template rhs<Real, J + 1>(cell, weights, flags);
        if (!Kokkos::isfinite(result))
            scalar::detail::flag(flags, scalar::detail::BadSystem);
        return result;
    }
};

} // namespace detail

/// @brief Creates the empty tail used to end an accepted history list.
inline auto time_history() { return detail::NoHistory{}; }

/// @brief Binds the already accepted levels in order t^n, t^(n-1), and so on.
/// The unknown at t^(n+1) is supplied separately to the operator. The history stores
/// descriptors by value and reads their data in place, so keep the accepted values fixed through the step.
/// @tparam Level TimeLevel for the most recent accepted step in this list.
/// @tparam Rest Types of any earlier accepted TimeLevels, in descending time order.
/// @param level Most recent accepted level (t^n at the outer call).
/// @param rest Earlier accepted levels, starting with t^(n-1) at the outer call.
template<class Level, class... Rest>
auto time_history(Level level, Rest... rest) {
    auto tail = time_history(rest...);
    return detail::History<Level, decltype(tail)>{level, tail};
}

} // namespace krona::transport::scalar::transient
