#pragma once

#include <krona/transport/detail/assembly_output.hpp>
#include <krona/transport/detail/component_pack.hpp>
#include <krona/transport/euler/detail/stencil_data.hpp>
#include <krona/transport/euler/jacobian.hpp>
#include <krona/transport/euler/state.hpp>
#include <krona/transport/euler/status.hpp>
#include <krona/transport/face_range.hpp>

namespace krona::transport::euler::detail {

/// @brief Assembles one owned block row per work item using cell-centre states.
/// @tparam Space Memory space of the mesh arrays.
/// @tparam R Scalar used to evaluate states and derivatives.
/// @tparam A Scalar used to integrate and sum derivative blocks.
/// @tparam Target Writable native blocks and the map from faces to entries.
/// @tparam Primitives Pack of pressure, velocity and temperature in real cells.
/// @tparam StateLaw Converts primitives to Riemann states.
/// @tparam Linearization Returns the owner and neighbour derivatives of an oriented flux.
/// @tparam Flags Writable batch diagnostics.
template<class Space, class R, class A, class Target, class Primitives, class StateLaw, class Linearization, class Flags>
struct InnerJacobianRows {
    StencilData<Space> data;
    Target target;
    Primitives values;
    StateLaw state_law;
    Linearization linearization;
    Flags flags;
    FaceRange range;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const {
        const auto current = prepare_state(state_law, transport::detail::read_primitive<R>(values, cell));
        if (!state_valid(current)) {
            flag(flags, BadState);
            return;
        }
        math::Matrix5<A> diagonal;
        bool touched = false;
        for (auto k = data.offsets(cell); k < data.offsets(cell + 1); ++k) {
            const auto face = data.faces(k);
            if (face < range.begin || face >= range.end) continue;
            const std::size_t f = static_cast<std::size_t>(face);
            const bool owner = data.owner(f) == static_cast<LocalIndex>(cell);
            const std::size_t other = static_cast<std::size_t>(owner ? data.neighbour(f) : data.owner(f));

            const auto adjacent = prepare_state(state_law, transport::detail::read_primitive<R>(values, other));
            if (!state_valid(adjacent)) {
                flag(flags, BadState);
                continue;
            }

            const numerics::riemann::Vector<R> normal{static_cast<R>(data.nx(f)), static_cast<R>(data.ny(f)),
                                                     static_cast<R>(data.nz(f))};

            const auto derivative = linearization(owner ? current : adjacent, owner ? adjacent : current, normal);
            const A area = static_cast<A>(data.area(f));

            // J is the derivative of outward flux, which is the negative of the assembled residual.
            const A outward = owner ? area : -area;
            const auto entry = owner ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);

            // Entries in this native block row are written by this thread only.
            const auto& off_diagonal = owner ? derivative.right : derivative.left;
            if (!transport::detail::add_block(target.blocks, entry, off_diagonal, outward))
                flag(flags, BadJacobian);
            transport::detail::accumulate_block(diagonal, owner ? derivative.left : derivative.right, outward);
            touched = true;
        }
        if (touched && !transport::detail::add_block(target.blocks, target.entries.diagonal(cell), diagonal, A{1}))
            flag(flags, BadJacobian);
    }
};

} // namespace krona::transport::euler::detail
