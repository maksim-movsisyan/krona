#pragma once

#include <krona/boundary/context.hpp>

namespace krona::boundary {

/// @brief Omits the explicit tangential reconstruction increment.
struct NoCorrection {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(const FaceContext<Real>&) const { return Real{0}; }
};

/// @brief Uses owned-cell gradients to compute (x_f - x_P - h*n) dot grad(phi).
template<class X, class Y, class Z>
struct GradientCorrection {
    X x;
    Y y;
    Z z;

    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(const FaceContext<Real>& c) const
    {
        const std::size_t i = static_cast<std::size_t>(c.owner);
        return c.tx*static_cast<Real>(x(i)) + c.ty*static_cast<Real>(y(i)) +
               c.tz*static_cast<Real>(z(i));
    }
};

} // namespace krona::boundary
