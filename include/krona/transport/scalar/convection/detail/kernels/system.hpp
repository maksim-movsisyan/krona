#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/transport/scalar/convection/detail/face_flux.hpp>
#include <krona/transport/scalar/convection/detail/stencil_data.hpp>

namespace krona::transport::scalar::convection::detail {

/// @brief Adds upwind matrix entries and the explicit reconstruction terms for one owned row.
/// @tparam AddRhs Include the explicit reconstruction term in the RHS.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Type used to evaluate and accumulate face terms.
/// @tparam Target Writable matrix and RHS slice with its face-entry map.
/// @tparam Flux Callable returning the signed mass rate integrated over a face.
/// @tparam Reconstruction Policy returning a face increment relative to the upwind cell value.
/// @tparam Flags Device view holding the batch's error flags.
template<bool AddRhs, class Space, class Real, class Target, class Flux, class Reconstruction, class Flags>
struct AddRows {
    StencilData<Space> data;
    Target target;
    Flux flux;
    Reconstruction reconstruction;
    Flags flags;
    Real time;
    FaceRange faces; ///< Local face interval contributing to this row.

    /// @brief Computes the terms seen by this owned cell.
    /// @param cell Local index of the cell.
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        using Value = typename decltype(target.values)::non_const_value_type;
        Real diagonal = 0, rhs = 0;

        for (auto k = data.offsets(i); k < data.offsets(i + 1); ++k) {
            const auto face = data.faces(k);
            if (face < faces.begin || face >= faces.end)
                continue;

            const auto c = context<Real>(data, face, time);
            const Real m = mass_flux<Real>(flux, c, flags);
            if (m == Real{0} || !representable_flux<Value>(m, flags))
                continue;

            // The face rate points out of the owner. Flip it for the neighbour's row.
            const bool owner = c.owner == cell;
            const Real outward = owner ? m : -m;
            // Outflow adds to the diagonal. Inflow couples this row to the cell across the face.
            if (outward > Real{0}) {
                diagonal += outward;
            } else {
                const std::size_t f = static_cast<std::size_t>(face);
                const auto entry = owner ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);
                scalar::detail::add_checked(target.values, entry, outward, flags);
            }

            // Keep the donor-cell matrix and move the reconstructed part to the RHS.
            if constexpr (AddRhs && reconstructed<Reconstruction>)
                rhs -= outward * interior_increment<Real>(reconstruction, c, m, flags);
        }

        scalar::detail::add_checked(target.values, target.entries.diagonal(i), diagonal, flags);
        if constexpr (AddRhs && reconstructed<Reconstruction>)
            scalar::detail::add_checked(target.rhs, i, rhs, flags);
    }
};

/// @brief Adds a physical boundary face's contribution to its adjacent cell's system row.
/// @tparam AddRhs Include prescribed inflow and outflow reconstruction in the RHS.
/// @tparam Space Kokkos memory space of the mesh arrays.
/// @tparam Real Type used to evaluate and accumulate face terms.
/// @tparam Target Writable matrix and RHS slice with its face-entry map.
/// @tparam Condition Concrete scalar condition held by the boundary patch.
/// @tparam Flux Callable returning the signed mass rate integrated over a face.
/// @tparam Reconstruction Policy returning a face increment relative to the upwind cell value.
/// @tparam Inflow Callable supplying the scalar carried into the domain.
/// @tparam Flags Device view holding the batch's error flags.
template<bool AddRhs, class Space, class Real, class Target, class Condition, class Flux, class Reconstruction,
         class Inflow, class Flags>
struct AddBoundary {
    boundary::GeometryView<Space> geometry;
    Target target;
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
        using Value = typename decltype(target.values)::non_const_value_type;
        if (m == Real{0} || !representable_flux<Value>(m, flags))
            return;

        // Several boundary faces may update the same cell.
        const std::size_t i = static_cast<std::size_t>(c.owner);
        // Use the cell value on outflow and the prescribed value on inflow.
        if (m > Real{0}) {
            scalar::detail::atomic_add_checked(target.values, target.entries.diagonal(i), m, flags);
            if constexpr (AddRhs && reconstructed<Reconstruction>)
                scalar::detail::atomic_add_checked(target.rhs, i,
                    -m * boundary_increment<Real>(reconstruction, c, flags), flags);
        } else if constexpr (AddRhs) {
            scalar::detail::atomic_add_checked(target.rhs, i,
                -m * inflow_value<Real>(inflow, c, condition, flags), flags);
        }
    }
};

} // namespace krona::transport::scalar::convection::detail
