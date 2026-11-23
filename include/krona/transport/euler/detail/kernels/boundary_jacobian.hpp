#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/numerics/reconstruction/types.hpp>
#include <krona/transport/detail/assembly_output.hpp>
#include <krona/transport/euler/boundary_jacobian.hpp>
#include <krona/transport/euler/detail/face_physics.hpp>

namespace krona::transport::euler::detail {

/// @brief Assembles the boundary derivative into the face owner's diagonal block.
/// @tparam Space Memory space of the boundary geometry.
/// @tparam R Scalar used for the face context and flux derivative.
/// @tparam A Scalar used to integrate the derivative over the face.
/// @tparam Target Writable matrix blocks and owner diagonal map.
/// @tparam Physics Face trace, state conversion and Riemann solver used by the residual.
/// @tparam Model Physical boundary model.
/// @tparam Condition Condition shared by this patch.
/// @tparam Closure Policy that supplies the exterior state.
/// @tparam Linearization Returns the derivative of this face's flux by owner conservative variables.
template<class Space, class R, class A, class Target, class Physics, class Model, class Condition,
         class Closure, class Linearization>
struct BoundaryJacobianFaces {
    boundary::GeometryView<Space> geometry;
    Target target;
    Physics physics;
    Model model;
    Condition condition;
    Closure closure;
    Linearization linearization;
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

        // Use the same interior trace and fallback as the residual at the base point.
        const std::size_t f = static_cast<std::size_t>(face);
        const auto trace = physics.boundary_trace(f, current);

        const BoundaryLinearizationState<R> states{current.state, trace.primitive,
            physics.template prepared_face<jacobian_uses_prepared_face<Closure, Linearization>>(c.extended_index)};

        const auto derivative = [&] {
            if constexpr (requires { model.at(c, condition); })
                return linearization(c, condition, model.at(c, condition), closure, states,
                                     physics.state_law, physics.solver);
            else return linearization(c, condition, model, closure, states, physics.state_law, physics.solver);
        }();

        const auto entry = target.entries.diagonal(cell);
        if (!derivative.valid || entry >= target.blocks.extent(0)) {
            flag(physics.flags, BadJacobian);
            return;
        }
        
        // Multiple faces of a patch can contribute to the same row.
        const A area = static_cast<A>(geometry.face_area(f));
        for (std::size_t row = 0; row < 5; ++row)
            for (std::size_t column = 0; column < 5; ++column) {
                const A contribution = area * static_cast<A>(derivative.block(row, column));
                if (!transport::detail::atomic_add(&target.blocks(entry, row, column), contribution))
                    flag(physics.flags, BadJacobian);
            }
    }
};

} // namespace krona::transport::euler::detail
