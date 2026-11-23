#pragma once

#include <krona/flow/flux.hpp>
#include <krona/math/matrix5.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <type_traits>

namespace krona::transport::detail {

template<class A, class R>
KOKKOS_INLINE_FUNCTION void accumulate_flux(Kokkos::Array<A, 5>& sum, flow::Flux<R> flux, A factor) {
    sum[0] += factor * static_cast<A>(flux.mass);
    sum[1] += factor * static_cast<A>(flux.momentum.x);
    sum[2] += factor * static_cast<A>(flux.momentum.y);
    sum[3] += factor * static_cast<A>(flux.momentum.z);
    sum[4] += factor * static_cast<A>(flux.energy);
}

template<class A, class R>
KOKKOS_INLINE_FUNCTION void accumulate_block(math::Matrix5<A>& sum, const math::Matrix5<R>& block, A factor) {
    for (std::size_t k = 0; k < sum.values.size(); ++k)
        sum.values[k] += factor * static_cast<A>(block.values[k]);
}

/// @brief Validates a narrowing store before converting a residual or matrix value.
template<class Value, class Sum>
KOKKOS_INLINE_FUNCTION bool representable(Sum value) {
    using R = std::common_type_t<Value, Sum>;
    return Kokkos::isfinite(value) && Kokkos::abs(static_cast<R>(value)) <=
        static_cast<R>(Kokkos::finite_max_v<Value>);
}

/// @brief Adds five residual contributions to one cell.
/// @tparam A Arithmetic type used for the addition.
/// @tparam Output Writable residual storage with value_type and output(cell, component) access.
/// @param cell Local cell index.
/// @param sum Contributions in mass, x/y/z momentum, and energy order.
/// @return True after writing all five components. If any result is nonfinite or
/// outside the storage type's range, returns false and leaves the row unchanged.
template<class A, class Output>
KOKKOS_INLINE_FUNCTION bool add_row(Output output, std::size_t cell, const Kokkos::Array<A, 5>& sum) {
    using Value = std::remove_const_t<typename Output::value_type>;
    Kokkos::Array<Value, 5> result;
    for (std::size_t c = 0; c < result.size(); ++c) {
        const A next = static_cast<A>(output(cell, c)) + sum[c];
        if (!representable<Value>(next)) return false;
        result[c] = static_cast<Value>(next);
    }
    for (std::size_t c = 0; c < result.size(); ++c) output(cell, c) = result[c];
    return true;
}

/// @brief Adds a block when each destination can represent the resulting coefficient.
template<class Blocks, class R, class A>
KOKKOS_INLINE_FUNCTION bool add_block(Blocks blocks, std::size_t entry, const math::Matrix5<R>& block, A factor) {
    using Value = typename Blocks::non_const_value_type;
    using Sum = std::common_type_t<Value, R, A>;
    if (entry >= blocks.extent(0)) return false;
    math::Matrix5<Value> result;
    for (std::size_t row = 0; row < 5; ++row)
        for (std::size_t column = 0; column < 5; ++column) {
            const Sum next = static_cast<Sum>(blocks(entry, row, column)) +
                             static_cast<Sum>(factor) * static_cast<Sum>(block(row, column));
            if (!representable<Value>(next)) return false;
            result(row, column) = static_cast<Value>(next);
        }
    for (std::size_t row = 0; row < 5; ++row)
        for (std::size_t column = 0; column < 5; ++column)
            blocks(entry, row, column) = result(row, column);
    return true;
}

/// @brief Adds a coefficient shared by boundary face work items and detects overflow.
template<class Value, class A>
KOKKOS_INLINE_FUNCTION bool atomic_add(Value* destination, A contribution) {
    if (!representable<Value>(contribution)) return false;
    const auto value = static_cast<Value>(contribution);
    const auto previous = Kokkos::atomic_fetch_add(destination, value);
    return Kokkos::isfinite(previous + value);
}

} // namespace krona::transport::detail
