#pragma once

#include <krona/numerics/reconstruction/schemes.hpp>

#include <utility>

namespace krona::numerics::reconstruction::detail {

// Runtime access to a heterogeneous fields::pack returns a value. Store through
// the typed component instead, so each destination keeps its own scalar type.
template<std::size_t C, class Output>
KOKKOS_INLINE_FUNCTION decltype(auto) factor_destination(const Output& output, std::size_t cell) {
    if constexpr (requires { output.template get<C>(cell); }) return output.template get<C>(cell);
    else return output(cell, C);
}

template<std::size_t C, class Output, class R>
KOKKOS_INLINE_FUNCTION void store_factor(const Output& output, std::size_t cell, R factor) {
    using Reference = decltype(factor_destination<C>(output, cell));
    using T = std::remove_reference_t<Reference>;
    static_assert(std::is_lvalue_reference_v<Reference> && !std::is_const_v<T> && std::is_floating_point_v<T>,
                  "Reconstruction: factor destinations must be writable floating-point components");
    factor_destination<C>(output, cell) = static_cast<T>(factor);
}

template<class Output, class R, std::size_t... C>
KOKKOS_INLINE_FUNCTION void store_shared(const Output& output, std::size_t cell, R factor, std::index_sequence<C...>)
{ (store_factor<C>(output, cell, factor), ...); }

template<class Output, class R, std::size_t N, std::size_t... C>
KOKKOS_INLINE_FUNCTION void store_factors(const Output& output, std::size_t cell,
                                         const Kokkos::Array<R, N>& factors, std::index_sequence<C...>)
{ (store_factor<C>(output, cell, factors[C]), ...); }

template<class R, class Stencil, class D, class Output, class Limiter, class Scheme, class Coupling>
struct PrepareFactors {
    Stencil stencil; D data; Output output; Limiter limiter; Scheme scheme;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const {
        constexpr auto N = D::components;
        Kokkos::Array<R, N> base{}, minimum{}, maximum{}, theta{};
        for (std::size_t c = 0; c < N; ++c) {
            base[c] = minimum[c] = maximum[c] = static_cast<R>(data.values(cell, c));
            theta[c] = R{1};
        }
        const auto count = stencil.size(cell);
        // Read every sample once for the entire component batch.
        for (std::size_t k = 0; k < count; ++k) {
            const auto entry = stencil(cell, k);
            for (std::size_t c = 0; c < N; ++c) {
                const R value = static_cast<R>(data.values(entry.sample, c));
                minimum[c] = Kokkos::min(minimum[c], value);
                maximum[c] = Kokkos::max(maximum[c], value);
            }
        }
        for (std::size_t k = 0; k < count; ++k) {
            const auto entry = stencil(cell, k);
            for (std::size_t c = 0; c < N; ++c) {
                const R increment = scheme.template increment<R>(entry.side, data, c);
                const R factor = static_cast<R>(limiter(LimiterContext<R>{cell, c, base[c], minimum[c], maximum[c], increment}));
                KOKKOS_ASSERT(Kokkos::isfinite(factor) && factor >= R{0} && factor <= R{1});
                theta[c] = Kokkos::min(theta[c], factor);
            }
        }
        if constexpr (std::is_same_v<Coupling, SharedMinimum>) {
            R shared = R{1};
            for (std::size_t c = 0; c < N; ++c) shared = Kokkos::min(shared, theta[c]);
            store_shared(output, cell, shared, std::make_index_sequence<Output::components>{});
        } else {
            store_factors(output, cell, theta, std::make_index_sequence<N>{});
        }
    }
};
} // namespace krona::numerics::reconstruction::detail
