#pragma once

#include <krona/numerics/reconstruction/types.hpp>

namespace krona::numerics::reconstruction::detail {

template<std::size_t N, class Pack>
KOKKOS_INLINE_FUNCTION constexpr void check_component_count() {
    if constexpr (requires { Pack::components; })
        static_assert(Pack::components == N, "Reconstruction: component counts do not match");
}

template<class Scheme, class D>
KOKKOS_INLINE_FUNCTION constexpr void check_input_components() {
    static_assert(D::components > 0);
    if constexpr ((Scheme::requirements & Gradients) != 0)
        check_component_count<D::components, decltype(D::gradients)>();
    if constexpr ((Scheme::requirements & Hessians) != 0)
        check_component_count<D::components, decltype(D::hessians)>();
    if constexpr ((Scheme::requirements & Factors) != 0) {
        using FactorsType = decltype(D::factors);
        static_assert(FactorsType::components == 1 || FactorsType::components == D::components,
                      "Reconstruction: factors must have one or N components");
    }
}

} // namespace krona::numerics::reconstruction::detail
