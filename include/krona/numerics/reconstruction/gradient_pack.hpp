#pragma once

#include <krona/fields/pack.hpp>
#include <krona/numerics/reconstruction/geometry.hpp>

#include <utility>

namespace krona::numerics::reconstruction {

/// @brief Assembles the three derivatives of one variable into a Cartesian vector.
/// @tparam X Pack of x derivatives, in the order of the value components.
/// @tparam Y Pack of y derivatives in the same order.
/// @tparam Z Pack of z derivatives in the same order.
template <class X, class Y, class Z>
struct GradientPack {
    static_assert(X::components == Y::components && X::components == Z::components);
    static constexpr std::size_t components = X::components;
    using value_type =
        std::common_type_t<typename X::value_type, typename Y::value_type, typename Z::value_type>;
    using memory_space = typename X::memory_space;
    static_assert(std::is_same_v<memory_space, typename Y::memory_space> &&
                  std::is_same_v<memory_space, typename Z::memory_space>);
    X x;
    Y y;
    Z z;

    KOKKOS_INLINE_FUNCTION std::size_t size() const {
        return Kokkos::min(x.size(), Kokkos::min(y.size(), z.size()));
    }

    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell, std::size_t c) const {
        return Vector<value_type>{x(cell, c), y(cell, c), z(cell, c)};
    }
};

/// @brief Groups three axis packs that share a common variable order.
/// @tparam X Provider of x derivatives.
/// @tparam Y Provider of y derivatives.
/// @tparam Z Provider of z derivatives.
template <class X, class Y, class Z>
KOKKOS_INLINE_FUNCTION auto gradients(X x, Y y, Z z) {
    return GradientPack<X, Y, Z>{x, y, z};
}

namespace detail {

template <std::size_t Axis, class Pack, std::size_t... C>
auto gradient_axis(Pack packed, std::index_sequence<C...>) {
    return fields::pack(packed.template component<3 * C + Axis>()...);
}

} // namespace detail

/// @brief Adapts interleaved 3*c+axis components without copying values.
/// @tparam Field FieldView or component pack with triples of derivatives.
/// For two variables use the order qx, qy, qz, Tx, Ty, Tz.
template <class Field>
auto gradient_pack(Field field) {
    static_assert(Field::components > 0 && Field::components % 3 == 0,
                  "Reconstruction: gradient components must be ordered 3*c+axis");
    const auto packed = fields::pack(field);
    using Indices = std::make_index_sequence<Field::components / 3>;
    return gradients(detail::gradient_axis<0>(packed, Indices{}), detail::gradient_axis<1>(packed, Indices{}),
                     detail::gradient_axis<2>(packed, Indices{}));
}

} // namespace krona::numerics::reconstruction
