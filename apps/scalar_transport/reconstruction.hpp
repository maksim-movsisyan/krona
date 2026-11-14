#pragma once

#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/transport/scalar/config.hpp>

namespace krona::apps::scalar_transport {

namespace rec = numerics::reconstruction;
namespace ts = transport::scalar;

/// @brief Dispatches to one concrete scheme on the host before launching work.
template<class Real, class Visitor>
void with_reconstruction(const ts::ReconstructionOptions& options, const Visitor& visit) {
    using Method = ts::ReconstructionMethod;
    using Limiter = ts::RatioLimiter;
    if (options.method == Method::Linear) {
        visit(rec::SecondOrderUpwind{});
        return;
    }
    mpi::require(options.method == Method::MusclKappa || options.method == Method::Tvd,
                 "Scalar transport: expected a high-order reconstruction");
    const auto with_skew = [&](auto skew) {
        using Skew = decltype(skew);
        if (options.method == Method::MusclKappa) {
            visit(rec::MusclKappa<Real, Skew>{static_cast<Real>(options.kappa), skew});
            return;
        }
        const auto tvd = [&](auto limiter) {
            visit(rec::Tvd<decltype(limiter), rec::DarwishMoukalled, Skew>{limiter, {}, skew});
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

} // namespace krona::apps::scalar_transport
