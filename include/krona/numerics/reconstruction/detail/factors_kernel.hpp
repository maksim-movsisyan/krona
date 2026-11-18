#pragma once

#include <krona/numerics/reconstruction/types.hpp>
#include <krona/execution/types.hpp>

#include <utility>

namespace krona::numerics::reconstruction::detail {

// Runtime access to a heterogeneous fields::pack returns a value. Store through
// the typed component instead, so each destination keeps its own scalar type.
template <std::size_t C, class Output>
KOKKOS_INLINE_FUNCTION decltype(auto) factor_destination(const Output& output, std::size_t cell) {
    if constexpr (requires { output.template get<C>(cell); })
        return output.template get<C>(cell);
    else
        return output(cell, C);
}

template <std::size_t C, class Output, class R>
KOKKOS_INLINE_FUNCTION void store_factor(const Output& output, std::size_t cell, R factor) {
    using Reference = decltype(factor_destination<C>(output, cell));
    using T = std::remove_reference_t<Reference>;
    static_assert(std::is_lvalue_reference_v<Reference> && !std::is_const_v<T> && std::is_floating_point_v<T>,
                  "Reconstruction: factor destinations must be writable floating-point components");
    factor_destination<C>(output, cell) = static_cast<T>(factor);
}

template <class Output, class R, std::size_t... C>
KOKKOS_INLINE_FUNCTION void store_shared(const Output& output, std::size_t cell, R factor,
                                         std::index_sequence<C...>) {
    (store_factor<C>(output, cell, factor), ...);
}

template <class Output, class R, std::size_t N, std::size_t... C>
KOKKOS_INLINE_FUNCTION void store_factors(const Output& output, std::size_t cell,
                                          const Kokkos::Array<R, N>& factors, std::index_sequence<C...>) {
    (store_factor<C>(output, cell, factors[C]), ...);
}

// Load the same evaluation geometry and formula the flux will use. Keep the component
// loop inside this call, so a face point is assembled only once per base cell.
template <class Real, std::size_t N, class Values, class Limiter>
struct LimitPoint {
    std::size_t cell;
    const Values& values;
    const Limiter& limiter;
    const Kokkos::Array<Real, N>& base;
    const Kokkos::Array<Real, N>& minimum;
    const Kokkos::Array<Real, N>& maximum;
    Kokkos::Array<Real, N>& factors;

    template <class Formula, class Side>
    KOKKOS_INLINE_FUNCTION void operator()(const Formula& formula, const Side& side) const {
        KOKKOS_ASSERT(side.cell == cell);
        for (std::size_t c = 0; c < N; ++c) {
            const Real delta = formula.template increment<Real>(side, values, c);
            const Real factor = static_cast<Real>(
                limiter(LimiterContext<Real>{cell, c, base[c], minimum[c], maximum[c], delta}));
            KOKKOS_ASSERT(Kokkos::isfinite(factor) && factor >= Real{0} && factor <= Real{1});
            factors[c] = Kokkos::min(factors[c], factor);
        }
    }
};

/// @brief Computes and stores the reconstruction limiter factors for one cell per call.
/// @tparam Real Floating-point type used to compute bounds and factors.
/// @tparam Profiles FaceProfiles for mesh faces or CellProfiles for a custom stencil.
/// Both provide the values for the bounds and the reconstruction to test at each point.
/// @tparam Output Writable field or pack where the factors are stored by cell and component.
/// @tparam Limiter Rule, such as BarthJespersen, that uses the bounds and a proposed correction
/// to choose a factor between zero and one at a test point.
/// @tparam Coupling Componentwise keeps each component's factor. SharedMinimum uses the
/// smallest factor across components and writes it to every output component.
template <class Real, class Profiles, class Output, class Limiter, class Coupling>
struct PrepareFactors {
    Profiles profiles;
    Output output;
    Limiter limiter;

    /// @brief Finds the cell's bounds, tests the reconstruction at each point and writes the factors.
    /// @param cell Local index of the owned cell whose factors are being computed.
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const {
        using Values = decltype(profiles.values);
        constexpr auto N = Values::components;
        // Start the bounds with the cell's own values and allow the full correction initially.
        Kokkos::Array<Real, N> base{}, minimum{}, maximum{}, factors{};
        for (std::size_t c = 0; c < N; ++c) {
            base[c] = minimum[c] = maximum[c] = static_cast<Real>(profiles.values(cell, c));
            factors[c] = Real{1};
        }

        // Extend the bounds with the sampled values, such as neighbours and boundary values.
        const auto samples = profiles.sample_count(cell);
        for (std::size_t k = 0; k < samples; ++k) {
            const auto sample = profiles.stencil.sample(cell, k);
            for (std::size_t c = 0; c < N; ++c) {
                const Real value = static_cast<Real>(profiles.values(sample, c));
                minimum[c] = Kokkos::min(minimum[c], value);
                maximum[c] = Kokkos::max(maximum[c], value);
            }
        }

        // Test the correction at each point and keep the smallest factor for each component.
        const LimitPoint<Real, N, Values, Limiter> limit{cell, profiles.values, limiter, base,
                                                         minimum, maximum, factors};
        const auto points = profiles.point_count(cell);
        for (std::size_t k = 0; k < points; ++k)
            profiles.visit_point(cell, k, limit);

        // A shared factor uses the most restrictive component in this cell.
        if constexpr (std::is_same_v<Coupling, SharedMinimum>) {
            Real shared = Real{1};
            for (std::size_t c = 0; c < N; ++c)
                shared = Kokkos::min(shared, factors[c]);
            store_shared(output, cell, shared, std::make_index_sequence<Output::components>{});
        } else {
            store_factors(output, cell, factors, std::make_index_sequence<N>{});
        }
    }
};

} // namespace krona::numerics::reconstruction::detail
