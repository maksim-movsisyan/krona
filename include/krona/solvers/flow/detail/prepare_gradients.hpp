#pragma once

#include <krona/boundary/physical/relations.hpp>
#include <krona/numerics/gradient/correction.hpp>
#include <krona/transport/viscous/heat_flux.hpp>

#include <array>
#include <utility>

namespace krona::solvers::flow::detail {

/// @brief Batches the gradient of every primitive component in one stencil pass.
template<class Gradient, class Primitives, class Gradients, std::size_t... C>
void apply_gradients(const Gradient& gradient, const Primitives& q, const Gradients& output,
                     std::index_sequence<C...>) {
    gradient.apply_set_async(std::array{q.template component<C>()...},
        std::array{output.template component<3 * C>()...},
        std::array{output.template component<3 * C + 1>()...},
        std::array{output.template component<3 * C + 2>()...});
}

/// @brief Evaluates the tangential correction of one primitive component at the boundary.
template<std::size_t C, class Gradients>
auto boundary_correction(const Gradients& output) {
    namespace gr = numerics::gradient;
    return gr::GradientCorrection{gr::cell_gradients(output.template component<3 * C>(),
        output.template component<3 * C + 1>(), output.template component<3 * C + 2>())};
}

/// @brief Fills physical boundary slots and computes gradients on owned cells.
/// Viscous runs update the boundary slots using the first gradient and recompute the gradient.
/// The caller exchanges gradient ghosts before assembling interface fluxes.
template<class Boundary, class Primitive, class Gradient, class Q, class G, class Properties, class Status>
void prepare_gradients(const Boundary& boundary, Primitive primitive, const Gradient& gradient,
                       const Q& q, const G& output, const Properties& properties, bool viscous, Status& status) {
    namespace ph = boundary::physical;
    using Components = std::make_index_sequence<Q::components>;
    if (!viscous) {
        status.begin_async();
        boundary.apply_async(primitive, status);
        status.check();
        apply_gradients(gradient, q, output, Components{});
        return;
    }

    const transport::viscous::WallHeatFlux heat{q, properties};
    status.begin_async();
    boundary.apply_async(primitive, status, heat);
    status.check();
    apply_gradients(gradient, q, output, Components{});

    const ph::Corrections corrections{boundary_correction<0>(output),
        ph::VectorCorrection{boundary_correction<1>(output), boundary_correction<2>(output), boundary_correction<3>(output)},
                                      boundary_correction<4>(output)};
    status.begin_async();
    boundary.apply_async(primitive, status, heat, corrections);
    status.check();
    apply_gradients(gradient, q, output, Components{});
}

} // namespace krona::solvers::flow::detail
