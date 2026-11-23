#pragma once

#include <krona/transport/detail/assembly_output.hpp>
#include <krona/transport/viscous/boundary_jacobian.hpp>
#include <krona/transport/viscous/status.hpp>

namespace krona::transport::viscous::detail {

/// @brief Adds one boundary face's flux or derivative to the owner.
/// Boundary work items can share an owner, so the updates are atomic.
/// @tparam Jacobian True to add a diagonal Jacobian block, false to add a viscous residual contribution.
/// @tparam R Scalar type used for face geometry, primitive states, fluxes and their derivatives.
/// @tparam A Scalar type used to multiply the face contribution by its area before the atomic update.
/// @tparam Target Writable five-component residual pack or a block view with its entry map.
/// @tparam Inputs Reader for the owner's primitive state and Cartesian gradients of u, v, w and T.
/// @tparam Properties Device-callable policy that evaluates viscosity and thermal conductivity.
/// @tparam Model Physical boundary model used to resolve the face state.
/// @tparam Linearization Policy that computes the boundary flux Jacobian with respect to the owner's conservative variables.
/// @tparam Flags Writable device view of error flags shared by the assembly batch.
/// @tparam Areas Read-only view of face areas indexed by local face ID.
template<bool Jacobian, class R, class A, class Target, class Inputs,
         class Properties, class Model, class Linearization, class Flags, class Areas>
struct BoundaryFaces {
    Target target;
    Inputs inputs;
    Properties properties;
    Model model;
    Linearization linearize;
    Flags flags;
    R bulk;
    Areas areas;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const boundary::FaceContext<R>& context, const Condition& condition) const {
        const std::size_t owner = static_cast<std::size_t>(context.owner);
        const auto state = inputs.template state<R>(owner);
        const auto gradients = inputs.template gradients<R>(owner);
        if constexpr (requires { model.at(context, condition); })
            apply(context, condition, model.at(context, condition), owner, state, gradients);
        else
            apply(context, condition, model, owner, state, gradients);
    }

private:
    template<class Condition, class LocalModel>
    KOKKOS_INLINE_FUNCTION void apply(const boundary::FaceContext<R>& context, const Condition& condition,
                                      const LocalModel& local, std::size_t owner,
                                      State<R> state, Gradients<R> gradients) const {
        const A area = static_cast<A>(areas(static_cast<std::size_t>(context.face)));

        BoundaryFace<R> face;
        if (!positive(area) ||
            !prepare_boundary(context, condition, local, state, gradients, properties, bulk, face)) {
            flag(flags, Error::BoundaryFace);
            return;
        }
        
        if constexpr (Jacobian) {
            const auto result = linearize(BoundaryLinearizationInput<R>{context, state, gradients, face},
                                          condition, local, properties);
            const auto entry = target.entries.diagonal(owner);
            if (!result.valid || entry >= target.blocks.extent(0)) {
                flag(flags, Error::Jacobian);
                return;
            }

            // The mass equation has no molecular transport contribution.
            for (std::size_t row = 1; row < 5; ++row)
                for (std::size_t column = 0; column < 5; ++column)
                    if (!transport::detail::atomic_add(&target.blocks(entry, row, column),
                            area * static_cast<A>(result.block(row, column)))) flag(flags, Error::Jacobian);
        } else {
            const auto result = outward_flux(face.data);
            if (!result.valid()) {
                flag(flags, Error::BoundaryFace);
                return;
            }
            Kokkos::Array<A, 5> contribution{};
            transport::detail::accumulate_flux(contribution, result.value, -area);
            for (std::size_t c = 1; c < contribution.size(); ++c)
                if (!transport::detail::atomic_add(&target(owner, c), contribution[c])) flag(flags, Error::Residual);
        }
    }
};

} // namespace krona::transport::viscous::detail
