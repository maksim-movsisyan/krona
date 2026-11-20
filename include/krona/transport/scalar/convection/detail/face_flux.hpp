#pragma once

#include <krona/transport/scalar/convection/reconstruction.hpp>
#include <krona/transport/scalar/status.hpp>

namespace krona::transport::scalar::convection::detail {

using scalar::detail::flag;
using scalar::detail::BadMassFlux;
using scalar::detail::BadInflow;
using scalar::detail::BadReconstruction;
using scalar::detail::BadSystem;

/// @brief Reads the mass flow rate in the face's owner-outward direction.
/// @tparam Real Type used for face flux arithmetic.
/// @tparam Flux Callable returning the signed rate through a face.
/// @tparam Context Inner or boundary face context accepted by the provider.
/// @tparam Flags Device view holding the batch's error flags.
/// @param flux Provider of the integrated rate.
/// @param c Face and time at which to sample it.
/// @param flags Batch to mark if the rate is nonfinite.
template<class Real, class Flux, class Context, class Flags>
KOKKOS_INLINE_FUNCTION Real mass_flux(const Flux& flux, const Context& c, Flags flags) {
    const Real m = static_cast<Real>(flux(c));
    if (!Kokkos::isfinite(m)) {
        flag(flags, BadMassFlux);
        return Real{0};
    }
    return m;
}

/// @brief Checks that a nonzero rate can be stored as a nonzero matrix coefficient.
/// @tparam Value Scalar type of the matrix entries.
/// @tparam Real Arithmetic type of the rate.
/// @tparam Flags Device view holding the batch's error flags.
/// @param m Nonzero mass flow rate.
/// @param flags Batch to mark if conversion overflows or rounds the rate to zero.
template<class Value, class Real, class Flags>
KOKKOS_INLINE_FUNCTION bool representable_flux(Real m, Flags flags) {
    const auto value = static_cast<Value>(m);
    if (!Kokkos::isfinite(value) || value == Value{0}) {
        flag(flags, BadSystem);
        return false;
    }
    return true;
}

/// @brief Reads the reconstruction increment on the upwind side of an inner face.
/// @tparam Real Type used for the rate and the increment.
/// @tparam R Face reconstruction policy or NoReconstruction.
/// @tparam Context Inner face context accepted by the policy.
/// @tparam Flags Device view holding the batch's error flags.
/// @param reconstruction Provider of increments relative to the donor cell.
/// @param c Face whose donor trace is needed.
/// @param m Nonzero mass rate in the face's owner-outward direction.
/// @param flags Batch to mark if the increment is nonfinite.
template<class Real, class R, class Context, class Flags>
KOKKOS_INLINE_FUNCTION Real interior_increment(const R& reconstruction, const Context& c, Real m, Flags flags) {
    if constexpr (!reconstructed<R>) {
        return Real{0};
    } else {
        const Real delta = static_cast<Real>(reconstruction.interior(c, m > Real{0}));
        if (!Kokkos::isfinite(delta)) {
            flag(flags, BadReconstruction);
            return Real{0};
        }
        return delta;
    }
}

/// @brief Reads the reconstruction increment from the adjacent cell to an outflow face.
/// @tparam Real Type used for the increment.
/// @tparam R Face reconstruction policy or NoReconstruction.
/// @tparam Context Boundary face context accepted by the policy.
/// @tparam Flags Device view holding the batch's error flags.
/// @param reconstruction Provider of the outflow trace increment.
/// @param c Boundary face whose cell-side trace is needed.
/// @param flags Batch to mark if the increment is nonfinite.
template<class Real, class R, class Context, class Flags>
KOKKOS_INLINE_FUNCTION Real boundary_increment(const R& reconstruction, const Context& c, Flags flags) {
    if constexpr (!reconstructed<R>) {
        return Real{0};
    } else {
        const Real delta = static_cast<Real>(reconstruction.boundary(c));
        if (!Kokkos::isfinite(delta)) {
            flag(flags, BadReconstruction);
            return Real{0};
        }
        return delta;
    }
}

/// @brief Samples the scalar carried into the mesh by a boundary face.
/// @tparam Real Arithmetic type of the sampled value.
/// @tparam Inflow Callable supplying the scalar for an inward flow.
/// @tparam Context Boundary face context accepted by the policy.
/// @tparam Condition Scalar condition assigned to the face.
/// @tparam Flags Device view holding the batch's error flags.
/// @param inflow Provider of the incoming scalar value.
/// @param c Face and time at which to sample it.
/// @param condition Condition from the face's boundary patch.
/// @param flags Batch to mark if the policy returns a nonfinite value.
template<class Real, class Inflow, class Context, class Condition, class Flags>
KOKKOS_INLINE_FUNCTION Real inflow_value(const Inflow& inflow, const Context& c, const Condition& condition,
                                         Flags flags) {
    static_assert(!std::is_arithmetic_v<Inflow>, "Pass time after reconstruction and inflow policies");
    const Real value = static_cast<Real>(inflow(c, condition));
    if (!Kokkos::isfinite(value)) {
        flag(flags, BadInflow);
        return Real{0};
    }
    return value;
}

} // namespace krona::transport::scalar::convection::detail
