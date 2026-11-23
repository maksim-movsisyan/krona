#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/mpi/error.hpp>
#include <array>
#include <type_traits>
#include <utility>

namespace krona::transport::detail {

/// @brief Reads the p,u,v,w,T layout shared by compressible operators.
template<class R, class Pack>
KOKKOS_INLINE_FUNCTION boundary::physical::Primitive<R, true> read_primitive(const Pack& values, std::size_t cell) {
    return {static_cast<R>(values(cell, 0)),
        {static_cast<R>(values(cell, 1)), static_cast<R>(values(cell, 2)), static_cast<R>(values(cell, 3))},
        static_cast<R>(values(cell, 4))};
}

template<class Pack, std::size_t... C>
auto component_pointers(const Pack& pack, std::index_sequence<C...>) {
    return std::array<const void*, sizeof...(C)>{static_cast<const void*>(pack.template component<C>().data())...};
}

template<class Pack, std::size_t... C>
void validate_components(const Pack& pack, std::size_t count, std::index_sequence<C...>) {
    const bool valid = ((pack.template component<C>().data() && pack.template component<C>().stride(0) > 0) && ...);
    mpi::require(count == 0 || valid, "Transport: invalid component storage");
}

/// @brief Checks field access on the selected device and validates visible storage descriptors.
template<class Exec, int Components, bool Write, class Pack>
void validate_pack(const Pack& pack, std::size_t count) {
    static_assert(Pack::components == Components);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename Pack::memory_space>::accessible);
    static_assert(std::is_floating_point_v<std::remove_const_t<typename Pack::value_type>>);
    if constexpr (Write) {
        static_assert(!std::is_const_v<typename Pack::value_type>);
        if constexpr (requires { Pack::writable; })
            static_assert(Pack::writable, "Transport: residual access requires a homogeneous writable pack");
    }

    mpi::require(pack.size() >= count, "Transport: undersized component pack");
    if constexpr (requires { pack.template component<0>(); }) {
        constexpr auto indices = std::make_index_sequence<Pack::components>{};
        validate_components(pack, count, indices);
        if constexpr (Write) {
            const auto pointers = component_pointers(pack, indices);
            for (std::size_t a = 0; a < pointers.size(); ++a)
                for (std::size_t b = a + 1; b < pointers.size(); ++b)
                    mpi::require(count == 0 || pointers[a] != pointers[b], "Transport: duplicate residual components");
        }
    }
    if constexpr (Write && requires { pack.views[0].pointer; }) {
        for (std::size_t a = 0; a < Components; ++a)
            for (std::size_t b = a + 1; b < Components; ++b)
                mpi::require(count == 0 || pack.views[a].pointer != pack.views[b].pointer,
                             "Transport: duplicate residual components");
    }
    if constexpr (requires { pack.views[0].pointer; pack.views[0].stride; }) {
        for (const auto& view : pack.views)
            mpi::require(count == 0 || (view.pointer && view.stride > 0), "Transport: invalid component storage");
    }
}

/// @brief Checks visible starting addresses. The caller ensures other subviews are disjoint.
template<class Input, class Output>
void reject_exact_alias(const Input& input, const Output& output) {
    constexpr auto message = "Transport: input/output alias is unsupported";
    if constexpr (requires { input.template component<0>(); output.template component<0>(); }) {
        const auto inputs = component_pointers(input, std::make_index_sequence<Input::components>{});
        const auto outputs = component_pointers(output, std::make_index_sequence<Output::components>{});
        for (const auto a : inputs)
            for (const auto b : outputs)
                mpi::require(!a || !b || a != b, message);
    }
    if constexpr (requires { input.views[0].pointer; output.views[0].pointer; }) {
        for (const auto& a : input.views)
            for (const auto& b : output.views)
                mpi::require(!a.pointer || !b.pointer ||
                    static_cast<const void*>(a.pointer) != static_cast<const void*>(b.pointer), message);
    }
}

} // namespace krona::transport::detail
