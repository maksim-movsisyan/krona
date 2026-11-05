#pragma once
#include "config.hpp"
#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <variant>

namespace krona::apps::scalar_transport {
namespace rec = numerics::reconstruction;
namespace ts = transport::scalar;
using M0 = rec::MusclKappa<ComputeReal,rec::NoSkewCorrection>;
using M1 = rec::MusclKappa<ComputeReal,rec::LinearSkewCorrection>;
template<class Limiter> using T0 = rec::Tvd<Limiter,rec::DarwishMoukalled,rec::NoSkewCorrection>;
template<class Limiter> using T1 = rec::Tvd<Limiter,rec::DarwishMoukalled,rec::LinearSkewCorrection>;
using SchemeChoice = std::variant<rec::FirstOrder,rec::SecondOrderUpwind,M0,M1,
    T0<rec::tvd::Minmod>,T1<rec::tvd::Minmod>,T0<rec::tvd::Superbee>,T1<rec::tvd::Superbee>,
    T0<rec::tvd::MonotonizedCentral>,T1<rec::tvd::MonotonizedCentral>,T0<rec::tvd::VanLeer>,T1<rec::tvd::VanLeer>,
    T0<rec::tvd::VanAlbada>,T1<rec::tvd::VanAlbada>,T0<rec::tvd::Koren>,T1<rec::tvd::Koren>>;
inline SchemeChoice choose_scheme(const ts::ReconstructionOptions& o)
{
    switch(o.method) {
    case ts::ReconstructionMethod::Upwind: return rec::FirstOrder{};
    case ts::ReconstructionMethod::Linear: return rec::SecondOrderUpwind{};
    case ts::ReconstructionMethod::MusclKappa:
        if(o.skew_correction)return M1{static_cast<ComputeReal>(o.kappa)};
        return M0{static_cast<ComputeReal>(o.kappa)};
    case ts::ReconstructionMethod::Tvd: break;
    default: mpi::fatal("Scalar transport: unknown reconstruction method");
    }
    const auto make = [&]<class L>() -> SchemeChoice {
        if(o.skew_correction)return T1<L>{};
        return T0<L>{};
    };
    switch(o.tvd_limiter) {
    case ts::RatioLimiter::Minmod: return make.template operator()<rec::tvd::Minmod>();
    case ts::RatioLimiter::Superbee: return make.template operator()<rec::tvd::Superbee>();
    case ts::RatioLimiter::MonotonizedCentral: return make.template operator()<rec::tvd::MonotonizedCentral>();
    case ts::RatioLimiter::VanLeer: return make.template operator()<rec::tvd::VanLeer>();
    case ts::RatioLimiter::VanAlbada: return make.template operator()<rec::tvd::VanAlbada>();
    case ts::RatioLimiter::Koren: return make.template operator()<rec::tvd::Koren>();
    }
    mpi::fatal("Scalar transport: unknown reconstruction selection");
}
} // namespace krona::apps::scalar_transport
