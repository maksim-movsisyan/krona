#pragma once

#include <krona/transport/scalar/diffusion/detail/stencil_data.hpp>
#include <krona/transport/scalar/diffusion/detail/correction.hpp>
#include <krona/boundary/scalar/conditions.hpp>
#include <krona/transport/diffusion/isotropic.hpp>
#include <krona/transport/scalar/status.hpp>

namespace krona::transport::scalar::diffusion::detail {

using scalar::detail::flag;
using scalar::detail::add_checked;
using scalar::detail::atomic_add_checked;
using scalar::detail::BadDiffusion;
using scalar::detail::BadBoundary;
using scalar::detail::BadCorrection;
using scalar::detail::BadSystem;
using scalar::detail::HasAnchor;

/// @brief Identifies the built-in cell provider, which requires a geometric face weight.
template<class Coefficient>
inline constexpr bool needs_geometric_weight = false;

template<class View, class Interpolation>
inline constexpr bool needs_geometric_weight<InterpolatedCellCoefficient<View, Interpolation>> = true;

/// @brief Flux from owner to neighbour in the form a*(phi_owner - phi_neighbour) + constant.
/// @tparam Real Arithmetic type of the flux coefficients.
template<class Real>
struct InteriorFlux {
    Real conductance = 0; ///< Two-point coefficient a = Gamma * area / normal_distance.
    Real constant = 0;    ///< Explicit flux from the nonorthogonal gradient projection.
};

/// @brief Builds the two-point conductance and optional correction for a face shared by two cells.
/// @tparam Real Arithmetic type of the face flux.
/// @tparam Space Memory space of the borrowed mesh arrays.
/// @tparam Law Diffusion law with a local face coefficient provider.
/// @tparam Correction Gradient projection policy, or NoCorrection.
/// @tparam Flags Writable device view for the current assembly batch.
/// @param data Borrowed mesh and optional geometry cache.
/// @param face Local ID of an interior or interface face.
/// @param law Provider of the material coefficient at that face.
/// @param correction Provider of the gradient projection on the nonorthogonal area vector.
/// @param flags Shared batch diagnostics.
/// @param time Time passed to the face providers.
template<class Real, class Space, class Law, class Correction, class Flags>
KOKKOS_INLINE_FUNCTION InteriorFlux<Real> interior_flux(const StencilData<Space>& data, LocalIndex face,
                                                        const Law& law, const Correction& correction, Flags flags, Real time) {
    auto metrics = data.template interior<Real, corrected<Correction>>(face, time);
    const auto& c = metrics.context;
    using Coefficient = std::remove_cvref_t<decltype(law.coefficient)>;
    if constexpr (needs_geometric_weight<Coefficient>) {
        if (!(c.owner_weight >= Real{0} && c.owner_weight <= Real{1})) {
            flag(flags, BadDiffusion);
            return {};
        }
    }
    const Real coefficient = static_cast<Real>(law.coefficient(c));
    const Real a = coefficient * metrics.area_over_distance;
    if (!Kokkos::isfinite(coefficient) || !(coefficient > Real{0}) || !Kokkos::isfinite(a) || !(a > Real{0})) {
        flag(flags, BadDiffusion); return {};
    }
    Real constant = 0;
    if constexpr (corrected<Correction>) {
        if (!(c.owner_weight >= Real{0} && c.owner_weight <= Real{1})) {
            flag(flags, BadCorrection);
            return {};
        }
        const auto& t = metrics.nonorthogonal;
        if (!math::isfinite(t)) {
            flag(flags, BadCorrection);
            return {};
        }
        const Real increment = static_cast<Real>(correction.interior(metrics.context, t.x, t.y, t.z));
        constant = -coefficient*increment;
        if (!Kokkos::isfinite(increment) || !Kokkos::isfinite(constant)) {
            flag(flags, BadCorrection); return {};
        }
    }
    return {a, constant};
}

/// @brief Outward boundary flux in the form owner*phi_owner + constant.
/// @tparam Real Arithmetic type of the flux coefficients.
template<class Real>
struct AffineFlux {
    Real owner = 0;    ///< Coefficient multiplying the owner cell value.
    Real constant = 0; ///< Contribution independent of the owner cell value.
};

/// @brief Converts a boundary gradient relation into an integrated outward flux.
/// @tparam CorrectionOnly True to subtract the base flux with zero correction. The slope must stay fixed.
/// @tparam Real Arithmetic type of the face flux.
/// @tparam Condition Boundary condition that supplies the affine normal-gradient relation.
/// @tparam Law Diffusion law with a local face coefficient provider.
/// @tparam Correction Policy supplying the boundary reconstruction increment.
/// @tparam Flags Writable device view for the current assembly batch.
template<bool CorrectionOnly = false, class Real, class Condition, class Law, class Correction, class Flags>
KOKKOS_INLINE_FUNCTION AffineFlux<Real> boundary_flux(const boundary::FaceContext<Real>& c,
                                                      const Condition& condition, const Law& law, 
                                                      const Correction& correction, Flags flags) {
    const Real coefficient = static_cast<Real>(law.coefficient(c));
    if (!Kokkos::isfinite(coefficient) || !(coefficient > Real{0})) {
        flag(flags, BadDiffusion); return {};
    }
    const Real increment = static_cast<Real>(correction(c));
    if (!Kokkos::isfinite(increment)) { flag(flags, BadCorrection); return {}; }
    const auto sampled = IsotropicDiffusion{UniformCoefficient<Real>{coefficient}};
    const auto relation = boundary::scalar::make_relation(condition, c, increment, sampled);
    const Real owner = sampled.outward_flux(c, relation.gradient_owner);
    Real constant = sampled.outward_flux(c, relation.gradient_constant);
    if (!Kokkos::isfinite(owner) || owner < Real{0} || !Kokkos::isfinite(constant)) {
        flag(flags, BadBoundary); return {};
    }
    if constexpr (CorrectionOnly) {
        const auto base = boundary::scalar::make_relation(condition, c, Real{0}, sampled);
        if (!Kokkos::isfinite(base.gradient_constant) || relation.gradient_owner != base.gradient_owner) {
            flag(flags, BadCorrection); return {};
        }
        constant = sampled.outward_flux(c, relation.gradient_constant-base.gradient_constant);
    }
    const Real a = CorrectionOnly ? Real{0} : c.area*owner;
    const Real b = c.area*constant;
    if (!Kokkos::isfinite(a) || !Kokkos::isfinite(b)) { flag(flags, BadSystem); return {}; }
    if constexpr (!CorrectionOnly) { if (owner > Real{0}) flag(flags, HasAnchor); }
    return {a, b};
}

} // namespace krona::transport::scalar::diffusion::detail
