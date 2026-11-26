#pragma once

#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/transport/compressible/config.hpp>

namespace krona::solvers::flow::detail {

namespace rec = numerics::reconstruction;
namespace ct = transport::compressible;

/// @brief Selects the local TVD ratio law without specializing the face kernel for each law.
struct TvdLimiter {
    ct::RatioLimiter kind;

    template<class R>
    KOKKOS_INLINE_FUNCTION R operator()(R ratio) const {
        switch (kind) {
            case ct::RatioLimiter::Minmod: return rec::tvd::Minmod{}(ratio);
            case ct::RatioLimiter::Superbee: return rec::tvd::Superbee{}(ratio);
            case ct::RatioLimiter::MonotonizedCentral: return rec::tvd::MonotonizedCentral{}(ratio);
            case ct::RatioLimiter::VanLeer: return rec::tvd::VanLeer{}(ratio);
            case ct::RatioLimiter::VanAlbada: return rec::tvd::VanAlbada{}(ratio);
            case ct::RatioLimiter::Koren: return rec::tvd::Koren{}(ratio);
        }
        return R{0};
    }
};

/// @brief Supplies the optional transverse correction for line-based schemes.
struct SkewCorrection {
    bool enabled;

    template<class R, class G, class V>
    KOKKOS_INLINE_FUNCTION R operator()(G gradient, V offset, V span, R fraction) const {
        if (enabled) return rec::LinearSkewCorrection{}.template operator()<R>(gradient, offset, span, fraction);
        return R{0};
    }
};

/// @brief Dispatches the selected interior formula on the host before assembling convection.
template<class R, class Gradients, class Visitor>
void with_high_order_formula(const ct::ReconstructionOptions& options, Gradients gradients, const Visitor& visit) {
    const SkewCorrection skew{options.skew_correction};
    switch (options.method) {
        case ct::ReconstructionMethod::Linear:
            return visit(rec::Linear{gradients});
        case ct::ReconstructionMethod::MusclKappa:
            return visit(rec::MusclKappa{gradients, static_cast<R>(options.kappa), skew});
        case ct::ReconstructionMethod::Tvd:
            return visit(rec::Tvd{gradients, TvdLimiter{options.tvd_limiter}, rec::DarwishMoukalled{}, skew});
        case ct::ReconstructionMethod::FirstOrder:
            mpi::fatal("Steady flow: expected a high-order reconstruction");
    }
    mpi::fatal("Steady flow: unknown reconstruction");
}

} // namespace krona::solvers::flow::detail
