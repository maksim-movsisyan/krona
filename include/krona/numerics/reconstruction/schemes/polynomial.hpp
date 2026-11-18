#pragma once

#include <krona/numerics/reconstruction/geometry.hpp>

namespace krona::numerics::reconstruction {

/// @brief Uses the base cell value at every point.
struct FirstOrder {
    static constexpr Geometry geometry = Geometry::Cell;

    template <class R, class Side, class Values>
    KOKKOS_INLINE_FUNCTION R increment(const Side&, const Values&, std::size_t) const {
        return R{0};
    }
};

/// @brief Adds the local linear Taylor increment at a specified point.
/// @tparam Gradients Provider returning {dq/dx, dq/dy, dq/dz} at (cell, component).
template <class Gradients>
struct Linear {
    static constexpr Geometry geometry = Geometry::Point;
    Gradients gradients;

    template <class Visitor>
    void visit_inputs(Visitor&& visit) const {
        visit("gradient", gradients);
    }

    template <class R, class Side, class Values>
    KOKKOS_INLINE_FUNCTION R increment(const Side& side, const Values&, std::size_t c) const {
        return dot_as<R>(gradients(side.cell, c), side.offset);
    }
};

/// @brief Mean-preserving quadratic Taylor polynomial, evaluated at one point.
/// @tparam Gradients Cartesian first derivatives at (cell, component).
/// @tparam Hessians Symmetric second derivatives at (cell, component).
/// @tparam Moments Symmetric volume averages of r*r^T at (cell), with r measured from the volume centroid.
template <class Gradients, class Hessians, class Moments>
struct Quadratic {
    static constexpr Geometry geometry = Geometry::Point;
    Gradients gradients;
    Hessians hessians;
    Moments moments;

    template <class Visitor>
    void visit_inputs(Visitor&& visit) const {
        visit("gradient", gradients);
        visit("Hessian", hessians);
        visit("moment", moments);
    }

    template <class R, class Side, class Values>
    KOKKOS_INLINE_FUNCTION R increment(const Side& side, const Values&, std::size_t c) const {
        // Subtract each central moment so the quadratic increment averages to zero in the cell.
        const auto h = hessians(side.cell, c);
        const auto m = moments(side.cell);
        const R x = static_cast<R>(side.offset.x), y = static_cast<R>(side.offset.y),
                z = static_cast<R>(side.offset.z);
        const R curvature = static_cast<R>(h.xx) * (x * x - static_cast<R>(m.xx)) +
                            static_cast<R>(h.yy) * (y * y - static_cast<R>(m.yy)) +
                            static_cast<R>(h.zz) * (z * z - static_cast<R>(m.zz)) +
                            R{2} * (static_cast<R>(h.xy) * (x * y - static_cast<R>(m.xy)) +
                                    static_cast<R>(h.xz) * (x * z - static_cast<R>(m.xz)) +
                                    static_cast<R>(h.yz) * (y * z - static_cast<R>(m.yz)));
        return dot_as<R>(gradients(side.cell, c), side.offset) + R{0.5} * curvature;
    }
};

} // namespace krona::numerics::reconstruction
