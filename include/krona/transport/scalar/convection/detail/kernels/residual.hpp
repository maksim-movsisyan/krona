#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/transport/scalar/convection/detail/face_flux.hpp>
#include <krona/transport/scalar/convection/detail/stencil_data.hpp>

namespace krona::transport::scalar::convection::detail {

/// @brief Evaluates the negative outward scalar flux from inner faces of one owned cell.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Type used to evaluate and accumulate face terms.
/// @tparam Field Rank-one view of the scalar at cell centres.
/// @tparam Output Writable rank-one view of the owned cell entries to increment.
/// @tparam Flux Callable returning the signed mass rate integrated over a face.
/// @tparam Reconstruction Policy returning a face increment relative to the upwind cell value.
/// @tparam Flags Device view holding the batch's error flags.
template<class Space, class Real, class Field, class Output, class Flux, class Reconstruction, class Flags>
struct ResidualRows {
    StencilData<Space> data;
    Field field;
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
        Real residual = 0;

        for (auto k = data.offsets(i); k < data.offsets(i + 1); ++k) {
            const auto face = data.faces(k);
            if (face < faces.begin || face >= faces.end)
                continue;

            const auto c = context<Real>(data, face, time);
            const Real m = mass_flux<Real>(flux, c, flags);
            if (m == Real{0})
                continue;

            // Both adjacent cells use the same donor and opposite outward rates.
            const auto donor = m > Real{0} ? c.owner : c.neighbour;
            const Real value = static_cast<Real>(field(static_cast<std::size_t>(donor))) +
                interior_increment<Real>(reconstruction, c, m, flags);
            const Real outward = c.owner == cell ? m : -m;
            residual -= outward * value;
        }

        scalar::detail::add_checked(output, i, residual, flags);
    }
};

/// @brief Evaluates the negative outward scalar flux from one physical boundary face.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Type used to evaluate and accumulate face terms.
/// @tparam Field Rank-one view of the scalar at cell centres.
/// @tparam Output Writable rank-one view of the owned cell entries to increment.
/// @tparam Condition Concrete scalar condition held by the boundary patch.
/// @tparam Flux Callable returning the signed mass rate integrated over a face.
/// @tparam Reconstruction Policy returning a face increment relative to the upwind cell value.
/// @tparam Inflow Callable supplying the scalar carried into the domain.
/// @tparam Flags Device view holding the batch's error flags.
template<class Space, class Real, class Field, class Output, class Condition, class Flux, class Reconstruction,
         class Inflow, class Flags>
struct ResidualBoundary {
    boundary::GeometryView<Space> geometry;
    Field field;
    Output output;
    Condition condition;
    Flux flux;
    Reconstruction reconstruction;
    Inflow inflow;
    Flags flags;
    Real time;

    /// @brief Updates the adjacent cell's entries.
    /// @param face Local index of a face on this patch.
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const auto c = geometry.template context<Real>(face, time);
        const Real m = mass_flux<Real>(flux, c, flags);
        if (m == Real{0})
            return;

        // Outflow uses the reconstructed cell-side trace. Inflow samples the boundary value.
        const std::size_t i = static_cast<std::size_t>(c.owner);
        const Real value = m > Real{0}
            ? static_cast<Real>(field(i)) + boundary_increment<Real>(reconstruction, c, flags)
            : inflow_value<Real>(inflow, c, condition, flags);
        scalar::detail::atomic_add_checked(output, i, -m * value, flags);
    }
};

} // namespace krona::transport::scalar::convection::detail
