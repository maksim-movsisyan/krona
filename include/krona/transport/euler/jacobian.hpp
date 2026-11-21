#pragma once

#include <krona/numerics/riemann/state_jacobians.hpp>

namespace krona::transport::euler {

/// @brief Borrows the writable matrix blocks and the locations where face contributions go.
/// @tparam Blocks Rank-three native view indexed by entry, row variable and column variable.
/// @tparam Entries Device map produced by the face assembly plan for those blocks.
template<class Blocks, class Entries>
struct JacobianView {
    Blocks blocks;   ///< Writable 5x5 blocks; assembly increments their existing coefficients.
    Entries entries; ///< Diagonal entries and off-diagonal entries for each face side.
};

/// @brief Derivatives of the oriented flux per unit area in conservative variable order.
/// @tparam R Scalar type of the derivative entries.
template<class R>
struct FaceJacobians {
    numerics::riemann::Matrix5<R> left;  ///< Derivative with respect to owner state.
    numerics::riemann::Matrix5<R> right; ///< Derivative with respect to neighbour state.
};

/// @brief Approximates the inner-face Jacobian using the two cell-centre states.
/// Treats the Rusanov signal speed as constant while differentiating.
/// @tparam EOS Provides the thermodynamic derivatives for the physical flux Jacobian.
template<class EOS>
struct FrozenRusanov {
    EOS eos;

    /// @brief Returns the two Jacobian blocks for a face with owner-to-neighbour normal.
    /// @tparam R Scalar used for the states and derivatives.
    /// @param left State at the owner centre.
    /// @param right State at the neighbour centre.
    /// @param unit_normal Unit normal from owner to neighbour.
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
