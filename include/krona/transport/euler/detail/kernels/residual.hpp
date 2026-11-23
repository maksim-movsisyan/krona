#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/numerics/reconstruction/types.hpp>
#include <krona/transport/detail/assembly_output.hpp>
#include <krona/transport/euler/detail/face_physics.hpp>
#include <krona/transport/face_range.hpp>

namespace krona::transport::euler::detail {

/// @brief Gathers inner-face fluxes into a single owned residual row.
/// @tparam Space Memory space of the mesh arrays.
/// @tparam R Scalar used for flux calculations by the operator.
/// @tparam A Scalar used to integrate and sum fluxes.
/// @tparam Output Writable pack in mass, momentum and energy order.
/// @tparam Physics Bound reconstruction, state conversion and flux solver.
template<class Space, class R, class A, class Output, class Physics>
struct InnerResidualRows {
    StencilData<Space> data;
    Output output;
    Physics physics;
    FaceRange range;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const {
        const auto current = physics.base(cell);
        if (!state_valid(current.state)) {
            flag(physics.flags, BadState);
            return;
        }

        Kokkos::Array<A, 5> sum{};
        for (auto k = data.offsets(cell); k < data.offsets(cell + 1); ++k) {
            const auto face = data.faces(k);
            if (face < range.begin || face >= range.end) continue;
            const std::size_t f = static_cast<std::size_t>(face);
            const auto flux = physics.inner(data, f, cell, current);
            if (!flux.valid()) continue;

            // Flux points from owner to neighbour. The residual subtracts outward flux.
            const A sign = data.owner(f) == static_cast<LocalIndex>(cell) ? A{-1} : A{1};
            transport::detail::accumulate_flux(sum, flux.flux, sign * static_cast<A>(data.area(f)));
        }
        if (!transport::detail::add_row(output, cell, sum)) flag(physics.flags, BadOutput);
    }
};

/// @brief Adds the flux of one exterior boundary face to its owner.
/// @tparam Space Memory space of the geometry.
/// @tparam R Scalar used for the face context and flux.
/// @tparam A Scalar used to integrate flux over the area.
/// @tparam Output Writable pack in mass, momentum and energy order.
/// @tparam Physics Bound reconstruction, state conversion and flux solver.
/// @tparam Model Physical boundary model.
/// @tparam Condition Condition shared by the patch.
/// @tparam Closure Policy that supplies the exterior state.
template<class Space, class R, class A, class Output, class Physics, class Model, class Condition, class Closure>
struct BoundaryResidualFaces {
    boundary::GeometryView<Space> geometry;
    Output output;
    Physics physics;
    Model model;
    Condition condition;
    Closure closure;
    numerics::reconstruction::CellRange cells;
    R time;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const auto c = geometry.template context<R>(face, time);
        const std::size_t cell = static_cast<std::size_t>(c.owner);
        if (cell < cells.begin || cell >= cells.end) return;
        const auto current = physics.base(cell);
        if (!state_valid(current.state)) {
            flag(physics.flags, BadState);
            return;
        }
        const auto flux = physics.boundary_flux(c, condition, model, closure, current);
        if (!flux.valid()) return;
        Kokkos::Array<A, 5> contribution{};
        transport::detail::accumulate_flux(contribution, flux.flux, -static_cast<A>(geometry.face_area(static_cast<std::size_t>(face))));
        // Two faces can share an owner and write the same residual component.
        for (std::size_t k = 0; k < contribution.size(); ++k)
            if (!transport::detail::atomic_add(&output(cell, k), contribution[k])) flag(physics.flags, BadOutput);
    }
};

} // namespace krona::transport::euler::detail
