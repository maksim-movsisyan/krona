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
    template<class Real, class Value>
    KOKKOS_INLINE_FUNCTION Real operator()(const boundary::FaceContext<Real>&,
        const boundary::scalar::FixedValue<Value>& condition) const
    { return static_cast<Real>(condition.value); }
    template<class Real, class Condition>
    KOKKOS_INLINE_FUNCTION Real operator()(const boundary::FaceContext<Real>&, const Condition&) const
    { return Kokkos::quiet_NaN_v<Real>; }
};

/// @brief Device descriptor of one compact patch profile, selected/validated by InflowProfiles.
/// Borrows a rank-one View's pointer/stride; the condition is deliberately ignored.
/// The source allocation must outlive submitted work. No ownership or per-face patch lookup.
template<class Value>
struct PatchInflowValues {
    const Value* values = nullptr;
    std::size_t stride = 1;
    LocalIndex face_begin = 0;
    template<class Real, class Condition>
    KOKKOS_INLINE_FUNCTION Value operator()(const boundary::FaceContext<Real>& c, const Condition&) const
    { return values[static_cast<std::size_t>(c.face-face_begin)*stride]; }
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
