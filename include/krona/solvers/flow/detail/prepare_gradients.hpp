#pragma once

#include <krona/boundary/physical/relations.hpp>
#include <krona/fields/pack.hpp>
#include <krona/numerics/gradient/correction.hpp>
#include <krona/transport/viscous/heat_flux.hpp>

namespace krona::solvers::flow::detail {

/// @brief Publishes physical boundary samples and computes gradients on owned cells.
/// The viscous pass corrects the boundary samples once using the first owned gradient.
/// The caller publishes gradient ghosts after this function and before interface fluxes.
template<class Boundary, class Primitive, class Gradient, class Q, class G, class Properties, class Status>
void prepare_gradients(const Boundary& boundary, Primitive primitive, const Gradient& gradient, Q q, G output,
                       const Properties& properties, bool viscous, Status& status) {
    namespace ph = boundary::physical;
    namespace gr = numerics::gradient;
    if (!viscous) {
        status.begin_async();
        boundary.apply_async(primitive, status);
        status.check();
        gradient.apply_async(q, output);
        return;
    }

    const auto values = fields::pack(q);
    const transport::viscous::WallHeatFlux heat{values, properties};
    status.begin_async();
    boundary.apply_async(primitive, status, heat);
    status.check();
    gradient.apply_async(q, output);

    const auto correction = [&](int component) {
        return gr::GradientCorrection{gr::cell_gradients(output, component)};
    };
    const ph::Corrections corrections{correction(0),
        ph::VectorCorrection{correction(1), correction(2), correction(3)}, correction(4)};
    status.begin_async();
    boundary.apply_async(primitive, status, heat, corrections);
    status.check();
    gradient.apply_async(q, output);
}

} // namespace krona::solvers::flow::detail
