#pragma once

#include <krona/numerics/riemann/state_jacobians.hpp>

namespace krona::transport::euler {

/// @brief Borrowed writable native blocks and face-to-entry maps for the same graph.
template<class Blocks, class Entries>
struct JacobianView {
    Blocks blocks;
    Entries entries;
};

/// @brief Flux-density derivatives in the conservative variable order.
template<class R>
struct FaceJacobians {
    numerics::riemann::Matrix5<R> left, right;
};

/// @brief First-order Rusanov linearization with the face's signal speed held fixed.
template<class EOS>
struct FrozenRusanov {
    EOS eos;

    template<class R>
    KOKKOS_INLINE_FUNCTION FaceJacobians<R> operator()(numerics::riemann::State<R> left,
        numerics::riemann::State<R> right, numerics::riemann::Vector<R> unit_normal) const {
        using namespace numerics::riemann;
        const R speed = Kokkos::max(Kokkos::abs(left.velocity.dot(unit_normal)) + left.sound_speed,
                                    Kokkos::abs(right.velocity.dot(unit_normal)) + right.sound_speed);
        auto l = physical_flux_jacobian(left, unit_normal, eos);
        auto r = physical_flux_jacobian(right, unit_normal, eos);
        for (std::size_t row = 0; row < 5; ++row) {
            l(row, row) += speed;
            r(row, row) -= speed;
        }
        for (std::size_t k = 0; k < l.values.size(); ++k) {
            l.values[k] *= R{0.5};
            r.values[k] *= R{0.5};
        }
        return {l, r};
    }
};

} // namespace krona::transport::euler
