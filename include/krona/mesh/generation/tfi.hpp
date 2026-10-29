#pragma once

#include <array>
#include <cmath>
#include <limits>

#include "krona/mesh/generation/config.hpp"

namespace krona::mesh::generation {

class TFI {
public:
    // Derivatives of the trilinear map, normalized to avoid a dimensional cutoff.
    [[nodiscard]] static bool positive_jacobian(const std::array<Vec3, 8>& p,
                                               double u, double v, double w) noexcept {
        const Vec3 du = (p[1]-p[0])*((1-v)*(1-w)) + (p[2]-p[3])*(v*(1-w)) +
                        (p[5]-p[4])*((1-v)*w) + (p[6]-p[7])*(v*w);
        const Vec3 dv = (p[3]-p[0])*((1-u)*(1-w)) + (p[2]-p[1])*(u*(1-w)) +
                        (p[7]-p[4])*((1-u)*w) + (p[6]-p[5])*(u*w);
        const Vec3 dw = (p[4]-p[0])*((1-u)*(1-v)) + (p[5]-p[1])*(u*(1-v)) +
                        (p[7]-p[3])*((1-u)*v) + (p[6]-p[2])*(u*v);
        const double a = std::hypot(du.x, du.y, du.z);
        const double b = std::hypot(dv.x, dv.y, dv.z);
        const double c = std::hypot(dw.x, dw.y, dw.z);
        if (!(a > 0 && b > 0 && c > 0) || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
            return false;
        const double determinant = (du/a).cross(dv/b).dot(dw/c);
        return std::isfinite(determinant) && determinant > 64 * std::numeric_limits<double>::epsilon();
    }

    // 1D Linear edge interpolation between two points: s in [0, 1]
    [[nodiscard]] static constexpr Vec3 interpolate_edge(const Vec3& p0, const Vec3& p1, double s) noexcept {
        return p0 * (1.0 - s) + p1 * s;
    }

    // 2D Bilinear face interpolation on a quad: u, v in [0, 1]
    // Vertices order: 0: (0,0), 1: (1,0), 2: (1,1), 3: (0,1)
    [[nodiscard]] static constexpr Vec3 interpolate_quad(const std::array<Vec3, 4>& corners,
                                                         double u,
                                                         double v) noexcept {
        const double n0 = (1.0 - u) * (1.0 - v);
        const double n1 = u * (1.0 - v);
        const double n2 = u * v;
        const double n3 = (1.0 - u) * v;

        return corners[0] * n0 + corners[1] * n1 + corners[2] * n2 + corners[3] * n3;
    }

    // 3D Trilinear volume interpolation inside a HEXA_8 block
    // Parametric coordinates: xi, eta, zeta in [0, 1]
    // Vertices must follow CGNS/VTK HEXA_8 indexing:
    //   Bottom plane (zeta = 0): 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0)
    //   Top plane    (zeta = 1): 4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    [[nodiscard]] static constexpr Vec3 interpolate_hex(const std::array<Vec3, 8>& corners,
                                                        double xi,
                                                        double eta,
                                                        double zeta) noexcept {
        const double one_minus_xi   = 1.0 - xi;
        const double one_minus_eta  = 1.0 - eta;
        const double one_minus_zeta = 1.0 - zeta;

        // Shape functions for HEXA_8
        const double n0 = one_minus_xi * one_minus_eta * one_minus_zeta;
        const double n1 = xi           * one_minus_eta * one_minus_zeta;
        const double n2 = xi           * eta           * one_minus_zeta;
        const double n3 = one_minus_xi * eta           * one_minus_zeta;

        const double n4 = one_minus_xi * one_minus_eta * zeta;
        const double n5 = xi           * one_minus_eta * zeta;
        const double n6 = xi           * eta           * zeta;
        const double n7 = one_minus_xi * eta           * zeta;

        return corners[0] * n0 + corners[1] * n1 + corners[2] * n2 + corners[3] * n3 +
               corners[4] * n4 + corners[5] * n5 + corners[6] * n6 + corners[7] * n7;
    }
};

} // namespace krona::mesh::generation 