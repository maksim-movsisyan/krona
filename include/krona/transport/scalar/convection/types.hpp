#pragma once

#include <krona/boundary/scalar/conditions.hpp>
#include <krona/transport/scalar/face_range.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::transport::scalar::convection {

using scalar::FaceRange;

/// @brief Reads the signed mass flow rate already integrated over each face.
/// Interface copies must describe the same physical flux with their respective normal signs.
/// @tparam View Rank-one device view indexed by local face, including physical boundary faces.
template<class View>
struct FaceMassFlux {
    View values; ///< Mass per unit time, positive along the face normal.

    /// @brief Looks up the rate using the face index in either kind of context.
    /// @tparam Context InteriorFaceContext or boundary::FaceContext.
    /// @param c Context whose face index selects the entry.
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return values(static_cast<std::size_t>(c.face)); }
};

/// @brief Supplies the same integrated mass flow rate in every face's own orientation.
/// @tparam Real Floating-point type of the prescribed rate.
template<class Real = ComputeReal>
struct UniformMassFlux {
    Real value = 0; ///< Mass per unit time, positive along the face normal.

    /// @brief Returns the stored rate without reading the face geometry.
    /// @tparam Context InteriorFaceContext or boundary::FaceContext.
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context&) const { return value; }
};

/// @brief Samples the prescribed scalar when mass enters through a boundary face.
/// A condition without a fixed value returns NaN so the diagnostic batch reports missing inflow data.
struct FixedValueInflow {
    /// @brief Reads the fixed scalar value or samples its profile at this face and time.
    /// @tparam Real Arithmetic type used by the transport kernel.
    /// @tparam Condition Scalar condition assigned to the face.
    /// @param c Boundary face and time used to sample a profile.
    /// @param condition Condition providing the inflow value.
    template<class Real, class Condition>
    KOKKOS_INLINE_FUNCTION Real operator()(const boundary::FaceContext<Real>& c, const Condition& condition) const {
        if constexpr (boundary::scalar::is_fixed_value<Condition>)
            return static_cast<Real>(boundary::profile::sample(condition.value, c));
        else return Kokkos::quiet_NaN_v<Real>;
    }
};

/// @brief Uses the upwind cell value directly, with no high-order increment.
struct NoReconstruction {};

/// @brief Describes a face between two local cells, including a possible MPI ghost.
/// @tparam Real Floating-point type of the geometry and evaluation time.
template<class Real>
struct InteriorFaceContext {
    LocalIndex face = 0;         ///< Local face index.
    LocalIndex owner = 0;        ///< Owned cell from which the face normal points out.
    LocalIndex neighbour = 0;    ///< Cell on the other side, possibly an MPI ghost.
    Real x = 0, y = 0, z = 0;    ///< Coordinates of the face centroid.
    Real nx = 0, ny = 0, nz = 0; ///< Unit normal pointing from owner to neighbour.
    Real area = 0;               ///< Face area, available to the mass flux provider.
    Real time = 0;               ///< Time at which the face data are evaluated.
};

} // namespace krona::transport::scalar::convection
