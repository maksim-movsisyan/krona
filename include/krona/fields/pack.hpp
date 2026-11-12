#pragma once

#include <krona/fields/component.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/mpi/error.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace krona::fields {
    
namespace detail {

// Indexed bases keep heterogeneous descriptors device-copyable; no std::tuple is stored.
template<std::size_t I, class Component>
struct PackSlot { Component descriptor; };

template<class Indices, class... Components>
struct PackStorage;

template<std::size_t... I, class... Components>
struct PackStorage<std::index_sequence<I...>, Components...> : PackSlot<I, Components>... {
    KOKKOS_INLINE_FUNCTION explicit PackStorage(Components... c) : PackSlot<I, Components>{c}... {}

    template<std::size_t C>
    KOKKOS_INLINE_FUNCTION auto descriptor() const {
        using Component = std::tuple_element_t<C, std::tuple<Components...>>;
        return static_cast<const PackSlot<C, Component>&>(*this).descriptor;
    }
};

template<class Input>
constexpr std::size_t pack_width() {
    if constexpr (Kokkos::is_view<Input>::value) {
        static_assert(Input::rank == 1, "Fields: use a FieldView or rank-one component views");
        return 1;
    } else {
        static_assert(Input::components > 0);
        return static_cast<std::size_t>(Input::components);
    }
}

template<std::size_t C, class First, class... Rest>
auto packed_input_component(const First& first, const Rest&... rest) {
    constexpr auto width = pack_width<First>();
    if constexpr (C < width) {
        if constexpr (Kokkos::is_view<First>::value) return first;
        else if constexpr (requires { first.template component<C>(); })
            return first.template component<C>();
        else return first.component(static_cast<int>(C));
    } else return packed_input_component<C-width>(rest...);
}

template<class Input>
std::size_t pack_extent(const Input& input) {
    if constexpr (Kokkos::is_view<Input>::value) return input.extent(0);
    else return input.size();
}

} // namespace detail

/// @brief Borrowed numeric components in argument order; no allocation, copy, submission or fence.
/// All components share a memory space. size() is their minimum entity extent.
template<class... Components>
class ComponentPack : private detail::PackStorage<std::index_sequence_for<Components...>, Components...> {
    static_assert(sizeof...(Components) > 0);
    using Storage = detail::PackStorage<std::index_sequence_for<Components...>, Components...>;
    using First = std::tuple_element_t<0, std::tuple<Components...>>;
    using Scalar = std::common_type_t<std::remove_const_t<typename Components::value_type>...>;
    static constexpr bool any_const = (std::is_const_v<typename Components::value_type> || ...);
    static_assert((std::is_arithmetic_v<std::remove_const_t<typename Components::value_type>> && ...),
                  "Fields: packed components must have arithmetic scalar types");
public:
    using value_type = std::conditional_t<any_const, const Scalar, Scalar>;
    using memory_space = typename First::memory_space;
    static constexpr std::size_t components = sizeof...(Components);
    static constexpr bool homogeneous =
        (std::is_same_v<std::remove_const_t<typename First::value_type>,
                        std::remove_const_t<typename Components::value_type>> && ...);
    static constexpr bool writable = homogeneous && !any_const;
    static_assert((std::is_same_v<memory_space, typename Components::memory_space> && ...),
                  "Fields: packed components must share one memory space");

    KOKKOS_INLINE_FUNCTION explicit ComponentPack(std::size_t extent, Components... c)
        : Storage(c...), extent_(extent) {}

    KOKKOS_INLINE_FUNCTION std::size_t size() const { return extent_; }
    KOKKOS_INLINE_FUNCTION bool empty() const { return extent_ == 0; }

    /// @brief Exact scalar reference; preserves this component's individual constness.
    template<std::size_t C>
    KOKKOS_INLINE_FUNCTION decltype(auto) get(std::size_t entity) const {
        static_assert(C < components);
        KOKKOS_ASSERT(entity < extent_);
        const auto c = this->template descriptor<C>();
        return (c.data[entity*c.stride]);
    }

    /// @brief Borrowed rank-one view of one component, restricted to the common pack extent.
    template<std::size_t C>
    KOKKOS_INLINE_FUNCTION auto component() const {
        static_assert(C < components);
        const auto c = this->template descriptor<C>();
        using T = typename decltype(c)::value_type;
        using View = Kokkos::View<T*, Kokkos::LayoutStride, memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        return View(c.data, Kokkos::LayoutStride(extent_, c.stride));
    }

    /// @brief Same scalar types: reference (read-only if any input is const).
    /// Mixed scalar types: common-type value; write through get<C>() or component<C>().
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(std::size_t entity, std::size_t c) const {
        KOKKOS_ASSERT(entity < extent_ && c < components);
        return access(entity, c, std::index_sequence_for<Components...>{});
    }

    /// @brief Read-only descriptors of the same buffers; does not copy field values.
    KOKKOS_INLINE_FUNCTION auto as_const() const
    { return read_only(std::index_sequence_for<Components...>{}); }

private:
    std::size_t extent_;

    template<std::size_t... I>
    KOKKOS_INLINE_FUNCTION decltype(auto) access(std::size_t entity, std::size_t c,
                                                std::index_sequence<I...>) const {
        if constexpr (homogeneous) {
            value_type* result = nullptr;
            ((c == I ? result = &this->template get<I>(entity) : result), ...);
            return (*result);
        } else {
            Scalar result{};
            ((c == I ? (result = static_cast<Scalar>(this->template get<I>(entity)), 0) : 0), ...);
            return result;
        }
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION auto const_descriptor() const {
        const auto c = this->template descriptor<I>();
        using T = const std::remove_const_t<typename decltype(c)::value_type>;
        return ComponentDescriptor<T, memory_space>{c.data, c.stride};
    }

    template<std::size_t... I>
    KOKKOS_INLINE_FUNCTION auto read_only(std::index_sequence<I...>) const
    { return ComponentPack<decltype(const_descriptor<I>())...>{extent_, const_descriptor<I>()...}; }
};

template<class... Components>
ComponentPack(std::size_t, Components...) -> ComponentPack<Components...>;

namespace detail {
template<std::size_t... I, class... Inputs>
auto make_component_pack(std::size_t extent, std::index_sequence<I...>, const Inputs&... inputs)
{ return ComponentPack{extent, descriptor_from_view(packed_input_component<I>(inputs...))...}; }
} // namespace detail

/// @brief Flattens FieldViews, rank-one Kokkos views or existing packs, in argument/component order.
/// Keep all backing allocations alive until every user finishes; no ownership is retained.
template<class First, class... Rest>
[[nodiscard]] auto pack(First first, Rest... rest) {
    constexpr auto width = detail::pack_width<First>() + (detail::pack_width<Rest>() + ... + 0);
    std::size_t extent = detail::pack_extent(first);
    ((extent = extent < detail::pack_extent(rest) ? extent : detail::pack_extent(rest)), ...);
    return detail::make_component_pack(extent, std::make_index_sequence<width>{}, first, rest...);
}

/// @brief An existing valid pack already has the required representation.
template<class... Components>
[[nodiscard]] auto pack(ComponentPack<Components...> packed) {
    return packed;
}

/// @brief Adapts one field; equivalent to pack(field).
template<class Field>
[[nodiscard]] auto field_pack(Field field) { return pack(field); }

} // namespace krona::fields
