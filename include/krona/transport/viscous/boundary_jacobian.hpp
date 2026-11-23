#pragma once

#include <krona/transport/detail/flux_difference.hpp>
#include <krona/transport/viscous/detail/boundary_face.hpp>
#include <krona/transport/viscous/detail/linearization.hpp>

namespace krona::transport::viscous {

/// @brief Arguments shared by analytic and finite-difference boundary linearizations.
/// @tparam R Scalar used for primitive states, face physics and derivatives.
template<class R>
struct BoundaryLinearizationInput {
    boundary::FaceContext<R> context;
    detail::State<R> owner;
    detail::Gradients<R> gradients; ///< Frozen Cartesian gradients in the owner cell.
    detail::BoundaryFace<R> face;   ///< Base flux data, including frozen transport coefficients.
};

/// @brief Unit-area derivative with respect to the five owner conservative densities.
template<class R>
struct BoundaryJacobian {
    math::Matrix5<R> block;
    bool valid = false;
};

namespace detail {

template<class R>
struct FluxSample {
    flow::Flux<R> flux;
    bool usable = false;
    KOKKOS_INLINE_FUNCTION bool valid() const { return usable; }
};

} // namespace detail

/// @brief Probes the boundary flux in p,u,v,w,T while keeping material and gradients fixed.
struct FiniteDifferenceBoundary {
    template<class R, class Condition, class Model, class Properties>
    KOKKOS_INLINE_FUNCTION BoundaryJacobian<R> operator()(const BoundaryLinearizationInput<R>& input,
                                                          const Condition& condition, const Model& model,
                                                          const Properties& properties) const {
        const auto& material = input.face.data.material;
        const auto flux = [&](const Kokkos::Array<R, 5>& point) -> detail::FluxSample<R> {
            const detail::State<R> owner{point[0], {point[1], point[2], point[3]}, point[4]};
            detail::BoundaryFace<R> face;
            if (!detail::prepare_boundary(input.context, condition, model, owner, input.gradients, properties,
                                          material.bulk, face, &material)) return {};
            const auto result = detail::outward_flux(face.data);
            return {result.value, result.valid()};
        };

        const auto point = transport::detail::primitive_array(input.owner);
        const auto base = detail::outward_flux(input.face.data);
        if (!base.valid()) return {};
        const auto reference = transport::detail::flux_array(base.value);
        const R speed = model.thermodynamic(input.owner).sound_speed;
        if (!detail::positive(speed)) return {};

        math::Matrix5<R> primitive;
        const R relative_step = Kokkos::cbrt(Kokkos::epsilon_v<R>);
        for (std::size_t column = 0; column < 5; ++column) {
            R scale = Kokkos::abs(point[column]);
            if (column > 0 && column < 4) scale = Kokkos::max(scale, speed);
            if (!transport::detail::differentiate_flux_column(flux, point, reference, column,
                                                               relative_step * scale, primitive)) return {};
        }

        BoundaryJacobian<R> result;
        result.valid = detail::conservative_block(primitive, input.owner, model.eos, result.block);
        return result;
    }
};

/// @brief Uses affine wall relations for standard walls and probes the remaining conditions.
struct WallLinearization {
    template<class R, class Condition, class Model, class Properties>
    KOKKOS_INLINE_FUNCTION BoundaryJacobian<R> operator()(const BoundaryLinearizationInput<R>& input,
                                                          const Condition& condition, const Model& model, 
                                                          const Properties& properties) const {
        using Numeric = std::remove_cvref_t<decltype(
            boundary::physical::sample_condition(condition, input.context))>;
        if constexpr (detail::no_slip_wall<Numeric, R> || detail::slip_wall<Numeric, R>) {
            const auto& face = input.face.data;
            const auto surface = detail::surface(face);
            if (!surface.valid()) return {};

            math::Matrix5<R> primitive;
            const R inverse_norm = R{1} / face.normal.norm2();
            for (std::size_t k = 0; k < 3; ++k) {
                const auto e = detail::axis<R>(k);
                const auto d_gradient = input.face.normal_gradient_owner * e;
                detail::flux_column(primitive, k + 1, face, surface.value.traction,
                    detail::traction_increment(face, d_gradient * inverse_norm),
                    input.face.velocity_owner * e, R{0});
            }
            primitive(4, 4) = -face.material.molecular.thermal_conductivity * input.face.thermal_gradient_owner;

            BoundaryJacobian<R> result;
            result.valid = detail::conservative_block(primitive, input.owner, model.eos, result.block);
            return result;
        } else {
            return FiniteDifferenceBoundary{}(input, condition, model, properties);
        }
    }
};

} // namespace krona::transport::viscous
