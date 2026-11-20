#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/transport/scalar/convection/detail/face_flux.hpp>
#include <krona/transport/scalar/convection/detail/stencil_data.hpp>

namespace krona::transport::scalar::convection::detail {

/// @brief Adds the reconstructed part of the inner fluxes to a frozen upwind RHS.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Type used to evaluate and accumulate face terms.
/// @tparam Output Writable rank-one view of the owned cell entries to increment.
/// @tparam Flux Callable returning the signed mass rate integrated over a face.
/// @tparam Reconstruction Policy returning a face increment relative to the upwind cell value.
/// @tparam Flags Device view holding the batch's error flags.
template<class Space, class Real, class Output, class Flux, class Reconstruction, class Flags>
struct CorrectionRows {
    StencilData<Space> data;
    Output output;
    Flux flux;
    Reconstruction reconstruction;
    Flags flags;
    Real time;
    FaceRange faces; ///< Local face interval contributing to this row.

    /// @brief Computes the terms seen by this owned cell.
    /// @param cell Local index of the cell.
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        Real rhs = 0;

        for (auto k = data.offsets(i); k < data.offsets(i + 1); ++k) {
            const auto face = data.faces(k);
            if (face < faces.begin || face >= faces.end)
                continue;

            const auto c = context<Real>(data, face, time);
            const Real m = mass_flux<Real>(flux, c, flags);
            if (m == Real{0})
                continue;

            // Add only the change from the donor cell value to its reconstructed trace.
            const Real outward = c.owner == cell ? m : -m;
            rhs -= outward * interior_increment<Real>(reconstruction, c, m, flags);
        }

        scalar::detail::add_checked(output, i, rhs, flags);
    }
};

/// @brief Adds the change in outflow caused by reconstructing the cell-side trace.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Type used to evaluate and accumulate face terms.
/// @tparam Output Writable rank-one view of the owned cell entries to increment.
/// @tparam Flux Callable returning the signed mass rate integrated over a face.
/// @tparam Reconstruction Policy returning a face increment relative to the upwind cell value.
/// @tparam Flags Device view holding the batch's error flags.
template<class Space, class Real, class Output, class Flux, class Reconstruction, class Flags>
struct CorrectionBoundary {
    boundary::GeometryView<Space> geometry;
    Output output;
    Flux flux;
    Reconstruction reconstruction;
    Flags flags;
    Real time;

    /// @brief Updates the adjacent cell's entries.
    /// @param face Local index of a face on this patch.
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const auto c = geometry.template context<Real>(face, time);
        const Real m = mass_flux<Real>(flux, c, flags);
        // Prescribed inflow is already part of the base RHS.
        if (m > Real{0})
            scalar::detail::atomic_add_checked(output, static_cast<std::size_t>(c.owner),
                -m * boundary_increment<Real>(reconstruction, c, flags), flags);
    }
};

} // namespace krona::transport::scalar::convection::detail
