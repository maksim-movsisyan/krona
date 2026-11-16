#pragma once

#include "step.hpp"

#include <krona/boundary/physical/evaluate.hpp>
#include <krona/transport/euler/boundary.hpp>

namespace krona::apps::euler_steady {

/// @brief Fills physical-face samples for the inviscid gradient and limiter stencils.
/// Views is the physical PrimitiveViews adapter. Model is the driver's compressible ideal-gas model.
template<class Views, class Model>
struct FillInviscidFaceSamples {
    Views primitive;
    Model model;
    ph::Status<Exec>::flags_type flags;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const boundary::FaceContext<Real>& context,
                                           const Condition& condition) const {
        bool failed = false;
        const ph::DeviceErrors errors{flags, &failed};
        const auto inside = primitive.template read<Real>(static_cast<std::size_t>(context.owner));
        const auto evaluation = [&] {
            using Numeric = ph::condition_type_t<Condition>;
            if constexpr (transport::euler::detail::wall<Numeric>) {
                // All inviscid wall stencils use slip and extrapolate pressure and temperature.
                const auto velocity = ph::sample_wall_velocity(condition, context);
                return ph::evaluate(ph::SlipWall<Real>{math::cast<Real>(velocity)}, context, inside, model,
                                    boundary::scalar::NoFluxLaw{}, ph::Corrections<>{}, errors);
            } else {
                return ph::evaluate(condition, context, inside, model, boundary::scalar::NoFluxLaw{},
                                    ph::Corrections<>{}, errors);
            }
        }();
        if (failed) return;
        if (!storable_primitive(primitive, evaluation.face)) {
            errors.fail(ph::Error::Output, "Steady Euler: unrepresentable boundary stencil sample");
            return;
        }
        primitive.write(context.extended_index, evaluation.face);
    }
};

} // namespace krona::apps::euler_steady
