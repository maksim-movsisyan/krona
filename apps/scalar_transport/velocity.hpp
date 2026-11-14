#pragma once

#include "cli.hpp"

#include <krona/boundary/patches.hpp>
#include <krona/math/vector3.hpp>

namespace krona::apps::scalar_transport {

inline constexpr double gaussian_speed = 0.8;

/// @brief Prescribed velocity at a face centre and time. Customize the custom case here.
template<class Real>
KOKKOS_INLINE_FUNCTION math::Vector3<Real> velocity(Case problem, Real /*x*/, Real /*y*/, Real /*z*/, Real /*time*/) {
    if (problem == Case::gaussian) return {static_cast<Real>(gaussian_speed), Real{0}, Real{0}};
    return {};
}

/// @brief Converts the prescribed velocity to integrated, owner-outward face mass flux.
template<class Real>
struct MassFlux {
    Case problem;
    Real density;
    boundary::PatchRange impermeable;

    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const {
        // Prescribe an exact zero on impermeable faces, even with rounded mesh normals.
        if constexpr (requires { c.boundary_index; })
            if (c.face >= impermeable.begin && c.face < impermeable.end) return Real{0};
        const auto u = velocity(problem, c.x, c.y, c.z, c.time);
        return density * c.area * (u.x*c.nx + u.y*c.ny + u.z*c.nz);
    }
};

/// @brief Binds the prescribed velocity to the mesh's boundary patches.
template<class Real>
[[nodiscard]] MassFlux<Real> make_mass_flux(Case problem, Real density, const boundary::PatchCatalog& catalog) {
    if (problem != Case::gaussian) return {problem, density, {}};
    for (const auto& patch : catalog.patches())
        if (patch.name == "walls") return {problem, density, patch.range};
    mpi::fatal("Scalar transport: Gaussian channel requires the walls patch");
}

} // namespace krona::apps::scalar_transport
