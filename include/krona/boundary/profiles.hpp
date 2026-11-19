#pragma once

#include <krona/boundary/patch_range.hpp>
#include <krona/fields/component.hpp>
#include <krona/math/vector3.hpp>

#include <cmath>
#include <type_traits>

namespace krona::boundary::profile {

namespace detail {
/// @brief Identifies a Cartesian vector that can be used directly as a constant profile.
/// @tparam T Parameter type to inspect.
template<class T> inline constexpr bool is_vector = false;
template<class R> inline constexpr bool is_vector<math::Vector3<R>> = true;

/// @brief Checks a constant parameter on the host, including all three components of a vector.
/// @tparam T Arithmetic type or math::Vector3.
/// @param value Parameter to check.
template<class T>
bool finite(const T& value) {
    if constexpr (std::is_arithmetic_v<T>) return std::isfinite(value);
    else return math::isfinite(value);
}
} // namespace detail

/// @brief Wraps a scalar or Cartesian vector that has the same value on every face and at every time.
/// @tparam T Type of the stored constant.
template<class T>
struct Constant {
    T value{}; ///< Parameter returned by the provider.

    /// @brief Returns the stored constant for any face context.
    /// @tparam Context Type supplied by the caller, typically FaceContext.
    template<class Context>
    KOKKOS_INLINE_FUNCTION T operator()(const Context&) const { return value; }

    /// @brief Checks the stored constant on the host.
    /// @tparam Exec Execution space of the condition using this provider.
    /// @details The patch range is accepted for the common provider interface. The same constant serves any patch.
    template<class Exec>
    void validate(PatchRange) const
    { mpi::require(detail::finite(value), "Boundary profile: constant must be finite"); }
};

/// @brief Reads one parameter per face from a borrowed array in local patch order.
/// @tparam T Type of one parameter in the array.
/// @tparam Space Kokkos memory space holding the array.
/// @details Entry zero belongs to range.begin. Keep the source array alive until the kernels finish.
template<class T, class Space>
struct PatchValues {
    fields::ComponentDescriptor<const T, Space> values; ///< Read-only pointer and stride.
    PatchRange range;                                   ///< Patch and interval used to interpret the array.
    std::size_t extent = 0;                             ///< Number of entries available in the array.

    /// @brief Reads the parameter for this face.
    /// @tparam Context Face context exposing the local mesh face ID as face.
    /// @param c Context for a face within the bound patch range.
    template<class Context>
    KOKKOS_INLINE_FUNCTION T operator()(const Context& c) const {
        KOKKOS_ASSERT(c.face >= range.begin && c.face < range.end);
        return values(static_cast<std::size_t>(c.face-range.begin));
    }

    /// @brief Checks that the patch range matches and every local face has one array entry.
    /// @tparam Exec Execution space that will read from the array.
    /// @param expected Local face range of the patch receiving this provider.
    template<class Exec>
    void validate(PatchRange expected) const {
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

/// @brief Borrows a patch's values from a rank-one Kokkos view and preserves its stride.
/// @tparam View Rank-one Kokkos view, for example a component subview of a field.
/// @param range Patch whose first face maps to view(0).
/// @param view Caller-owned storage with one entry per face in range.
template<class View>
[[nodiscard]] auto patch_values(PatchRange range, View view) {
    const auto d = fields::descriptor_from_view(view);
    using T = std::remove_const_t<typename View::value_type>;
    return PatchValues<T, typename View::memory_space>{{d.data, d.stride}, range, view.extent(0)};
}

/// @brief Reads a plain scalar or vector directly, or calls provider(context) to sample a profile.
/// @tparam Provider Constant parameter or device-callable provider.
/// @tparam Context Face context accepted by the provider.
/// @param provider Parameter or object that evaluates it.
/// @param context Face and time for this evaluation.
template<class Provider, class Context>
KOKKOS_INLINE_FUNCTION decltype(auto) sample(const Provider& provider, const Context& context) {
    if constexpr (std::is_arithmetic_v<Provider> || detail::is_vector<Provider>) return provider;
    else return provider(context);
}

/// @brief Checks an inline constant or asks a provider to validate its host metadata.
/// @tparam Exec Execution space that will read any data borrowed by the provider.
/// @tparam Provider Constant parameter or a provider with an optional validate<Exec>(range) method.
/// @param provider Parameter or provider about to be used on a patch.
/// @param range Local face range the provider should accept.
template<class Exec, class Provider>
void validate(const Provider& provider, PatchRange range) {
    if constexpr (std::is_arithmetic_v<Provider>)
        mpi::require(detail::finite(provider), "Boundary profile: parameter must be finite");
    else if constexpr (detail::is_vector<Provider>)
        mpi::require(detail::finite(provider), "Boundary profile: vector must be finite");
    else if constexpr (requires { provider.template validate<Exec>(range); })
        provider.template validate<Exec>(range);
}

} // namespace krona::boundary::profile
