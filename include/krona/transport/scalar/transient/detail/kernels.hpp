#pragma once

#include <krona/transport/scalar/transient/mass.hpp>

namespace krona::transport::scalar::transient::detail {

using scalar::detail::BadSystem;
using scalar::detail::BadTemporalState;
using scalar::detail::flag;

/// @brief Assembles the time contribution for one owned equation row.
/// @tparam AddRhs Include the accepted history term in the RHS.
/// @tparam Real Arithmetic type used for the diagonal and RHS contributions.
/// @tparam Target Writable system slice with one diagonal entry ID per owned cell.
/// @tparam Weights Array with the t^(n+1) weight followed by the accepted history weights.
/// @tparam Mass Provider of integrated cell mass at t^(n+1).
/// @tparam History Accepted mass and scalar at t^n, t^(n-1), and so on.
/// @tparam Flags Writable view of the batch's diagnostic flags.
template<bool AddRhs, class Real, class Target, class Weights, class Mass, class History, class Flags>
struct AddSystem {
    Target target;
    Weights weights;
    Mass mass_at_new_time;
    History history;
    Flags flags;

    /// @brief Adds the coefficient of the unknown and optionally the known history term.
    /// @param cell Local index of the owned cell.
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const {
        const auto entry = static_cast<std::size_t>(target.entries.diagonal(cell));
        if (entry >= target.values.extent(0)) {
            flag(flags, BadSystem);
            return;
        }

        // The new scalar is unknown. Its mass times the new-time weight goes on the diagonal.
        const Real diagonal = weights[0] * read_mass<Real>(mass_at_new_time, cell, flags);
        using Value = typename decltype(target.values)::non_const_value_type;
        if (!(static_cast<Value>(diagonal) > Value{0})) {
            flag(flags, BadSystem);
            return;
        }
        scalar::detail::add_checked(target.values, entry, diagonal, flags);

        // The accepted values are already known. Their weights start at index 1 (t^n).
        if constexpr (AddRhs)
            scalar::detail::add_checked(target.rhs, cell, history.template rhs<Real, 1>(cell, weights, flags), flags);
    }
};

/// @brief Adds the negative integrated time derivative to one owned residual entry.
/// @tparam Real Arithmetic type used to evaluate the derivative.
/// @tparam Field Rank-one view of the candidate scalar at t^(n+1).
/// @tparam Output Writable rank-one residual view indexed by owned cell ID.
/// @tparam Weights Array with the t^(n+1) weight followed by the accepted history weights.
/// @tparam Mass Provider of integrated cell mass at t^(n+1).
/// @tparam History Accepted mass and scalar at t^n, t^(n-1), and so on.
/// @tparam Flags Writable view of the batch's diagnostic flags.
template<class Real, class Field, class Output, class Weights, class Mass, class History, class Flags>
struct AddResidual {
    Field field_at_new_time;
    Output output;
    Weights weights;
    Mass mass_at_new_time;
    History history;
    Flags flags;

    /// @brief Evaluates the time term in b - A*phi using the supplied candidate at t^(n+1).
    /// @param cell Local index of the owned cell.
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const {
        const Real mass = read_mass<Real>(mass_at_new_time, cell, flags);
        const Real value = static_cast<Real>(field_at_new_time(cell));
        if (!Kokkos::isfinite(value)) {
            flag(flags, BadTemporalState);
            return;
        }

        // Use the same history RHS as assembly and subtract the new-time diagonal times the candidate.
        const Real residual = history.template rhs<Real, 1>(cell, weights, flags) - weights[0] * mass * value;
        scalar::detail::add_checked(output, cell, residual, flags);
    }
};

} // namespace krona::transport::scalar::transient::detail
