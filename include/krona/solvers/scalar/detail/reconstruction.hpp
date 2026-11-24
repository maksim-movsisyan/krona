#pragma once

#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/transport/scalar/config.hpp>

namespace krona::solvers::scalar::detail {

namespace rec = numerics::reconstruction;
namespace ts = transport::scalar;

/// @brief Passes a formula with its gradients to the visitor before launching device work.
/// @tparam Real Scalar type storing the selected MUSCL parameter.
/// @tparam Gradients Provider of the Cartesian derivative vectors.
/// @tparam Visitor Host callable receiving the configured formula.
template<class Real, class Gradients, class Visitor>
void with_reconstruction(const ts::ReconstructionOptions& options, Gradients gradients, const Visitor& visit) {
    using Method = ts::ReconstructionMethod;
    using Limiter = ts::RatioLimiter;
    if (options.method == Method::Linear) {
        visit(rec::Linear{gradients});
        return;
    }
    mpi::require(options.method == Method::MusclKappa || options.method == Method::Tvd,
                 "Scalar transport: expected a high-order reconstruction");
    const auto with_skew = [&](auto skew) {
        if (options.method == Method::MusclKappa) {
            visit(rec::MusclKappa{gradients, static_cast<Real>(options.kappa), skew});
            return;
        }
        const auto tvd = [&](auto limiter) {
            visit(rec::Tvd{gradients, limiter, rec::DarwishMoukalled{}, skew});
        };
        switch (options.tvd_limiter) {
            case Limiter::Minmod: return tvd(rec::tvd::Minmod{});
            case Limiter::Superbee: return tvd(rec::tvd::Superbee{});
            case Limiter::MonotonizedCentral: return tvd(rec::tvd::MonotonizedCentral{});
            case Limiter::VanLeer: return tvd(rec::tvd::VanLeer{});
            case Limiter::VanAlbada: return tvd(rec::tvd::VanAlbada{});
            case Limiter::Koren: return tvd(rec::tvd::Koren{});
        }
        mpi::fatal("Scalar transport: unknown TVD limiter");
    };
    if (options.skew_correction) with_skew(rec::LinearSkewCorrection{});
    else with_skew(rec::NoSkewCorrection{});
}

} // namespace krona::solvers::scalar::detail
