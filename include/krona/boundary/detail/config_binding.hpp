#pragma once

#include <krona/boundary/patches.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace krona::boundary::detail {

/// @brief Visits configured patches in file order after matching each entry to the mesh catalog.
/// @tparam Patches Configured entries with a patch_id and name.
/// @tparam Bind Host callback accepting the entry and its mesh face range.
/// @param catalog Labelled patches in mesh ID order.
/// @param patches Configuration entries, one per labelled mesh patch, even if it has no faces on this rank.
/// @param bind Binds the condition after the patch ID and name have been checked.
template<class Patches, class Bind>
void bind_configured_patches(const PatchCatalog& catalog, const Patches& patches, const Bind& bind) {
    const auto mesh_patches = catalog.patches();
    mpi::require(patches.size() == mesh_patches.size(), "Boundary: configuration must describe every mesh patch");

    std::vector<bool> seen(mesh_patches.size(), false);
    for (const auto& entry : patches) {
        mpi::require(entry.patch_id >= 0 && static_cast<std::uint64_t>(entry.patch_id) < mesh_patches.size(),
                     "Boundary: configuration patch ID is out of range");
        const auto id = static_cast<std::size_t>(entry.patch_id);
        mpi::require(!seen[id], "Boundary: duplicate configuration patch ID");
        mpi::require(entry.name == mesh_patches[id].name, "Boundary: patch name does not match the mesh");
        seen[id] = true;
        bind(entry, mesh_patches[id].range);
    }
}

/// @brief Calls a profile resolver and checks that it binds exactly one typed condition.
/// @tparam Resolver Host callback accepting (spec, range, bind) or (spec, bind).
/// @tparam Spec Configuration of one file-backed condition.
/// @tparam Assign Callback that stores the resolved condition.
/// @param resolver Resolves the profile, even on ranks whose patch range is empty.
/// @param spec Condition and its configured profile paths.
/// @param range Local interval of patch faces.
/// @param assign Assigns the concrete condition supplied by the resolver.
/// @param source Module name used in diagnostics.
template<class Resolver, class Spec, class Assign>
void resolve_profile_once(Resolver& resolver, const Spec& spec, PatchRange range,
                          const Assign& assign, std::string_view source) {
    bool assigned = false;
    const auto bind = [&](auto condition) {
        if (assigned)
            mpi::fatal(std::string{source} + ": profile resolver assigned more than once");
        assign(std::move(condition));
        assigned = true;
    };
    if constexpr (requires { resolver(spec, range, bind); })
        resolver(spec, range, bind);
    else
        resolver(spec, bind);
    if (!assigned)
        mpi::fatal(std::string{source} + ": profile resolver did not assign a condition");
}

/// @brief Checks whether a configuration value fits the selected floating-point type before casting.
/// @tparam Real Type used by the boundary calculation.
/// @param value Value read from TOML or interpolated from a CSV profile.
/// @param source Module name used in diagnostics.
template<class Real>
[[nodiscard]] Real checked_parameter(double value, std::string_view source) {
    if (!std::isfinite(value) || std::abs(value) > static_cast<double>(std::numeric_limits<Real>::max()))
        mpi::fatal(std::string{source} + ": parameter is not representable in the selected precision");
    return static_cast<Real>(value);
}

} // namespace krona::boundary::detail
