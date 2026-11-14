#pragma once

#include <krona/boundary/scalar/conditions.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::transport::scalar::convection {

/// @brief Integrated mass/time, signed along each rank's canonical owner-outward normal.
/// Interface copies must describe the same physical flux with their respective normal signs.
template<class View>
struct FaceMassFlux {
    View values;
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return values(static_cast<std::size_t>(c.face)); }
};

template<class Real = ComputeReal>
struct UniformMassFlux {
    Real value = 0;
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context&) const { return value; }
};

/// @brief Only FixedValue supplies default inflow data. Other conditions return invalid data
/// if actually encountered on inflow; zero-flux/outflow faces never access this policy.
struct FixedValueInflow {
    template<class Real, class Condition>
    KOKKOS_INLINE_FUNCTION Real operator()(const boundary::FaceContext<Real>& c, const Condition& condition) const {
        if constexpr (boundary::scalar::is_fixed_value<Condition>)
            return static_cast<Real>(boundary::profile::sample(condition.value, c));
        else return Kokkos::quiet_NaN_v<Real>;
    }
};

/// @brief No high-order increment; no reconstruction field/gradient input is needed for assembly.
struct NoReconstruction {};

template<class Real>
struct InteriorFaceContext {
    LocalIndex face = 0, owner = 0, neighbour = 0;
    Real x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0;
    Real area = 0, time = 0;
};

} // namespace krona::transport::scalar::convection
