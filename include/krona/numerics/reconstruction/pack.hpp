#pragma once

#include <krona/numerics/reconstruction/types.hpp>

#include <algorithm>
#include <stdexcept>

namespace krona::numerics::reconstruction {
namespace detail {
template<class T>
struct Component {
    T* pointer = nullptr;
    std::size_t stride = 1;
};
template<class T, class View>
auto component(View view)
{
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    if (view.extent(0) != 0 && (view.data() == nullptr || view.stride(0) == 0))
        throw std::invalid_argument("Reconstruction: invalid component storage");
    return Component<T>{view.data(), view.stride(0)};
}
} // namespace detail

/// @brief No allocation or ownership; components may have different lengths/strides but share scalar and memory space.
template<class T, std::size_t N, class Space>
struct ComponentPack {
    static_assert(N > 0 && std::is_floating_point_v<std::remove_const_t<T>>);
    using value_type = T;
    using memory_space = Space;
    static constexpr std::size_t components = N;
    Kokkos::Array<detail::Component<T>, N> views{};
    std::size_t extent = 0; // Minimum component extent.
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return extent; }
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t cell, std::size_t component) const
    {
        KOKKOS_ASSERT(cell < extent && component < N);
        return views[component].pointer[cell*views[component].stride];
    }
};

template<class First, class... Rest>
auto pack(First first, Rest... rest)
{
    using Value = std::remove_const_t<typename First::value_type>;
    using Space = typename First::memory_space;
    static_assert((std::is_same_v<Value, std::remove_const_t<typename Rest::value_type>> && ...));
    static_assert((std::is_same_v<Space, typename Rest::memory_space> && ...));
    using T = std::conditional_t<std::is_const_v<typename First::value_type> ||
        (std::is_const_v<typename Rest::value_type> || ...), const Value, Value>;
    ComponentPack<T, 1+sizeof...(Rest), Space> result;
    result.views = {detail::component<T>(first), detail::component<T>(rest)...};
    result.extent = first.extent(0);
    ((result.extent = std::min(result.extent, rest.extent(0))), ...);
    return result;
}

/// @brief Structural adapter for KRONA FieldView; this header does not depend on the fields module.
template<class Field>
auto field_pack(Field field)
{
    using T = typename Field::value_type;
    ComponentPack<T, static_cast<std::size_t>(Field::components), typename Field::memory_space> result;
    result.extent = field.size();
    for (int c = 0; c < Field::components; ++c)
        result.views[static_cast<std::size_t>(c)] = detail::component<T>(field.component(c));
    return result;
}

template<class X, class Y, class Z>
struct GradientPack {
    static_assert(X::components == Y::components && X::components == Z::components);
    static constexpr std::size_t components = X::components;
    using value_type = std::common_type_t<typename X::value_type, typename Y::value_type, typename Z::value_type>;
    using memory_space = typename X::memory_space;
    static_assert(std::is_same_v<memory_space, typename Y::memory_space> && std::is_same_v<memory_space, typename Z::memory_space>);
    X x; Y y; Z z;
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return Kokkos::min(x.size(), Kokkos::min(y.size(), z.size())); }
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell, std::size_t c) const
    { return Vector<value_type>{x(cell, c), y(cell, c), z(cell, c)}; }
};
template<class X, class Y, class Z>
KOKKOS_INLINE_FUNCTION auto gradients(X x, Y y, Z z) { return GradientPack<X, Y, Z>{x, y, z}; }

/// @brief Adapts the gradient module's 3*c+axis component order without copying data.
template<class Field>
auto gradient_pack(Field field)
{
    static_assert(Field::components % 3 == 0);
    using T = typename Field::value_type;
    using Pack = ComponentPack<T, static_cast<std::size_t>(Field::components/3), typename Field::memory_space>;
    Pack x, y, z;
    x.extent = y.extent = z.extent = field.size();
    for (int c = 0; c < Field::components/3; ++c) {
        const auto i = static_cast<std::size_t>(c);
        x.views[i] = detail::component<T>(field.component(3*c));
        y.views[i] = detail::component<T>(field.component(3*c+1));
        z.views[i] = detail::component<T>(field.component(3*c+2));
    }
    return gradients(x, y, z);
}

} // namespace krona::numerics::reconstruction