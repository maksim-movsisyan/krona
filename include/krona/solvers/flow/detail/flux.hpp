#pragma once

#include <krona/numerics/riemann/flux.hpp>
#include <krona/numerics/riemann/hll.hpp>
#include <krona/numerics/riemann/hllc.hpp>
#include <krona/transport/compressible/config.hpp>

namespace krona::solvers::flow::detail {

/// @brief Selects the configured residual flux within a single Riemann kernel type.
struct NumericalFlux {
    transport::compressible::Flux kind;

    template<class R>
    KOKKOS_INLINE_FUNCTION numerics::riemann::Result<R> operator()(numerics::riemann::State<R> left,
        numerics::riemann::State<R> right, numerics::riemann::Vector<R> normal) const {
        namespace ri = numerics::riemann;
        using Flux = transport::compressible::Flux;
        switch (kind) {
            case Flux::Hllc: return ri::Hllc{}(left, right, normal);
            case Flux::Hll: return ri::Hll{}(left, right, normal);
            case Flux::Rusanov: return ri::Rusanov{}(left, right, normal);
        }
        return {{}, R{0}, ri::Status::InvalidState};
    }
};

} // namespace krona::solvers::flow::detail
