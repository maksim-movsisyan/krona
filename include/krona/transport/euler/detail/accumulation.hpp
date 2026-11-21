#pragma once

#include <krona/numerics/riemann/types.hpp>
#include <krona/transport/euler/status.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <type_traits>

namespace krona::transport::euler::detail {

/// @brief Scales a flux by signed area and adds it to a five-component sum.
/// @tparam A Scalar type of the sum and area.
/// @tparam R Scalar type of the flux per unit area.
template<class A, class R>
KOKKOS_INLINE_FUNCTION void accumulate(Kokkos::Array<A, 5>& sum, const numerics::riemann::Flux<R>& flux, A factor) {
    sum[0] += factor * static_cast<A>(flux.mass);
    sum[1] += factor * static_cast<A>(flux.momentum.x);
    sum[2] += factor * static_cast<A>(flux.momentum.y);
    sum[3] += factor * static_cast<A>(flux.momentum.z);
    sum[4] += factor * static_cast<A>(flux.energy);
}

/// @brief Checks all five sums before writing back a row owned by this thread.
/// @tparam A Scalar type used for the sums.
/// @tparam Output Writable residual pack.
/// @tparam Flags Writable batch diagnostics.
template<class A, class Output, class Flags>
KOKKOS_INLINE_FUNCTION void add(Output output, std::size_t cell, const Kokkos::Array<A, 5>& sum, Flags flags) {
    using Value = std::remove_const_t<typename Output::value_type>;
    Kokkos::Array<Value, 5> result;
    for (std::size_t c = 0; c < result.size(); ++c) {
        const A next = static_cast<A>(output(cell, c)) + sum[c];
        result[c] = static_cast<Value>(next);
        if (!Kokkos::isfinite(sum[c]) || !Kokkos::isfinite(next) || !Kokkos::isfinite(result[c])) {
            flag(flags, BadOutput);
            return;
        }
    }
    for (std::size_t c = 0; c < result.size(); ++c)
        output(cell, c) = result[c];
}

/// @brief Checks the increment and the result of an atomic add to the stored scalar.
/// @tparam Value Stored scalar type.
/// @tparam A Scalar type used to compute the face contribution.
/// @tparam Flags Writable batch diagnostics.
/// @param destination Address of the row entry shared by face work items.
/// @param contribution Value to add after checking that it fits.
/// @param flags Active diagnostics.
/// @param error Flag to set on overflow or nonfinite data.
template<class Value, class A, class Flags>
KOKKOS_INLINE_FUNCTION void atomic_add_checked(Value* destination, A contribution, Flags flags, Diagnostic error) {
    using Sum = std::common_type_t<Value, A>;
    constexpr Sum limit = static_cast<Sum>(Kokkos::finite_max_v<Value>);
    const Sum increment = static_cast<Sum>(contribution);
    if (!Kokkos::isfinite(increment) || Kokkos::abs(increment) > limit) {
        flag(flags, error);
        return;
    }
    const Value value = static_cast<Value>(increment);
    const Value previous = Kokkos::atomic_fetch_add(destination, value);
    if (!Kokkos::isfinite(previous + value)) flag(flags, error);
}

} // namespace krona::transport::euler::detail
