#pragma once

#include <krona/transport/detail/assembly_output.hpp>
#include <krona/transport/viscous/detail/inner_face.hpp>
#include <krona/transport/viscous/detail/linearization.hpp>
#include <krona/transport/viscous/status.hpp>
#include <krona/transport/face_range.hpp>

namespace krona::transport::viscous::detail {

/// @brief Gathers inner-face contributions into one owned cell row per work item.
/// @tparam Jacobian True to assemble compact Jacobian blocks, false to accumulate the viscous residual.
/// @tparam Space Memory space of the mesh views borrowed by the stencil.
/// @tparam R Scalar type used for face states, gradients, fluxes and their derivatives.
/// @tparam A Scalar type used to multiply face contributions by area and accumulate them in each row.
/// @tparam Target Writable five-component residual pack or, for Jacobian assembly, a block view with its entry map.
/// @tparam Inputs Reader for cell primitives (p, u, v, w, T) and Cartesian gradients of u, v, w and T.
/// @tparam Properties Device-callable policy that evaluates viscosity and thermal conductivity at the face state.
/// @tparam EOS Equation of state used to convert Jacobian columns from primitive to conservative variables.
/// @tparam Flags Writable device view of error flags shared by the assembly batch.
template<bool Jacobian, class Space, class R, class A, class Target, class Inputs,
         class Properties, class EOS, class Flags>
struct InnerRows {
    StencilData<Space> stencil;
    Target target;
    Inputs inputs; // see stencil_data.hpp for example of ds
    Properties properties;
    EOS eos;
    Flags flags;
    FaceRange range;
    R bulk;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell_id) const {
        const std::size_t cell = static_cast<std::size_t>(cell_id);
        const auto state = inputs.template state<R>(cell);
        const auto gradients = inputs.template gradients<R>(cell);
        Kokkos::Array<A, 5> residual{};
        math::Matrix5<A> diagonal;
        bool touched = false;

        for (auto i = stencil.offsets(cell); i < stencil.offsets(cell + 1); ++i) {
            const auto face = stencil.faces(i);
            if (face < range.begin || face >= range.end) continue;

            const std::size_t f = static_cast<std::size_t>(face);
            const std::size_t owner = static_cast<std::size_t>(stencil.owner(f));
            const std::size_t neighbour = static_cast<std::size_t>(stencil.neighbour(f));
            const bool on_owner = owner == cell;
            const std::size_t other = on_owner ? neighbour : owner;

            const auto adjacent = inputs.template state<R>(other);
            const auto adjacent_gradients = inputs.template gradients<R>(other);

            const auto left = on_owner ? state : adjacent;
            const auto right = on_owner ? adjacent : state;
            const auto geometry = stencil.template inner_geometry<R>(f, owner, neighbour);

            FaceData<R> data;
            if (!prepare_inner(geometry, left, right, on_owner ? gradients : adjacent_gradients,
                               on_owner ? adjacent_gradients : gradients, properties, bulk, data)) {
                flag(flags, Error::InnerFace);
                continue;
            }
            const A outward_area = (on_owner ? A{1} : A{-1}) * static_cast<A>(stencil.area(f));
            
            if constexpr (Jacobian) {
                const auto stress = surface(data);
                if (!stress.valid()) {
                    flag(flags, Error::InnerFace);
                    continue;
                }
                const R own_sign = on_owner ? R{-1} : R{1};
                const R own_weight = on_owner ? geometry.owner_weight : R{1} - geometry.owner_weight;
                const R slope = own_sign / geometry.height;
                const auto own = inner_primitive_block(data, stress.value.traction, slope, own_weight);
                const auto off = inner_primitive_block(data, stress.value.traction, -slope, R{1} - own_weight);
                math::Matrix5<R> own_block, off_block;
                if (!conservative_block(own, state, eos, own_block) ||
                    !conservative_block(off, adjacent, eos, off_block)) {
                    flag(flags, Error::Jacobian);
                    continue;
                }

                const auto entry = on_owner ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);
                if (!transport::detail::add_block(target.blocks, entry, off_block, outward_area))
                    flag(flags, Error::Jacobian);
                transport::detail::accumulate_block(diagonal, own_block, outward_area);
            } else {
                const auto flux = outward_flux(data);
                if (!flux.valid()) {
                    flag(flags, Error::InnerFace);
                    continue;
                }
                transport::detail::accumulate_flux(residual, flux.value, -outward_area);
            }
            touched = true;
        }

        if (!touched) return;
        if constexpr (Jacobian) {
            if (!transport::detail::add_block(target.blocks, target.entries.diagonal(cell), diagonal, A{1}))
                flag(flags, Error::Jacobian);
        } else {
            if (!transport::detail::add_row(target, cell, residual)) flag(flags, Error::Residual);
        }
    }
};

} // namespace krona::transport::viscous::detail
