#pragma once

#include <krona/boundary/patch_range.hpp>
#include <krona/fields/component.hpp>
#include <krona/math/vector3.hpp>

#include <type_traits>

namespace krona::boundary::profile {

namespace detail {
template<class T> inline constexpr bool is_vector = false;
template<class R> inline constexpr bool is_vector<math::Vector3<R>> = true;
template<class T>
bool finite(const T& value)
{
    if constexpr (std::is_arithmetic_v<T>) return std::isfinite(value);
    else return math::isfinite(value);
}
} // namespace detail

/// @brief Any device-copyable value, including math::Vector3; independent of condition physics.
template<class T>
struct Constant {
    T value{};
    template<class Context>
    KOKKOS_INLINE_FUNCTION T operator()(const Context&) const { return value; }
    template<class Exec>
    void validate(PatchRange) const
    { mpi::require(detail::finite(value), "Boundary profile: constant must be finite"); }
};

/// @brief Read-only borrowed values in local patch-face order, not boundary-slot or global order.
template<class T, class Space>
struct PatchValues {
    fields::ComponentDescriptor<const T, Space> values;
    PatchRange range;
    std::size_t extent = 0;

    template<class Context>
    KOKKOS_INLINE_FUNCTION T operator()(const Context& c) const
    {
        KOKKOS_ASSERT(c.face >= range.begin && c.face < range.end);
        return values(static_cast<std::size_t>(c.face-range.begin));
    }
    template<class Exec>
    void validate(PatchRange expected) const
    {
        static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
        mpi::require(expected.patch_id == range.patch_id && expected.begin == range.begin &&
            expected.end == range.end && range.begin >= 0 && range.end >= range.begin,
            "Boundary profile: patch range mismatch");
        mpi::require(extent == static_cast<std::size_t>(range.end-range.begin),
                     "Boundary profile: extent must equal the local patch face count");
        mpi::require(extent == 0 || (values.data != nullptr && values.stride > 0),
                     "Boundary profile: invalid component storage");
    }
};

/// @brief No allocation/copy/submission; buffers remain owned by the caller. Strides are preserved.
template<class View>
[[nodiscard]] auto patch_values(PatchRange range, View view)
{
    const auto d = fields::descriptor_from_view(view);
    using T = std::remove_const_t<typename View::value_type>;
    return PatchValues<T, typename View::memory_space>{{d.data, d.stride}, range, view.extent(0)};
}

/// @brief Plain arithmetic parameters are constant; other providers implement provider(context).
template<class Provider, class Context>
KOKKOS_INLINE_FUNCTION decltype(auto) sample(const Provider& provider, const Context& context)
{
    if constexpr (std::is_arithmetic_v<Provider> || detail::is_vector<Provider>) return provider;
    else return provider(context);
}

/// @brief Host metadata only. A custom provider may implement validate<Exec>(PatchRange).
template<class Exec, class Provider>
void validate(const Provider& provider, PatchRange range)
{
    if constexpr (std::is_arithmetic_v<Provider>)
        mpi::require(std::isfinite(provider), "Boundary profile: parameter must be finite");
    else if constexpr (detail::is_vector<Provider>)
        mpi::require(math::isfinite(provider), "Boundary profile: vector must be finite");
    else if constexpr (requires { provider.template validate<Exec>(range); })
        provider.template validate<Exec>(range);
}

} // namespace krona::boundary::profile