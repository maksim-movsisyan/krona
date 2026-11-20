#pragma once

#include <krona/execution/types.hpp>

#include <type_traits>

namespace krona::transport {

/// @brief Supplies the same scalar diffusion coefficient on every face.
/// @tparam Real Floating-point type used to store the coefficient.
template<class Real = ComputeReal>
struct UniformCoefficient {
    Real value = 1; ///< Constant coefficient in the units of the diffusion law.

    /// @brief Returns the constant for any face.
    /// @tparam Context Any face context type. The coefficient uses only the stored value.
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context&) const { return value; }
};

/// @brief Uses the owner-cell coefficient as the value on an exterior face.
/// @tparam View Rank-one Kokkos view indexed by local cell ID. Strided views are supported.
template<class View>
struct OwnerCellCoefficient {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values; ///< Cell coefficients accessible from the execution space of the caller.

    /// @brief Reads the coefficient from the cell adjacent to the boundary face.
    /// @tparam Context Boundary face context with an owner index.
    /// @param c Exterior face. If a neighbour index is present, it must be negative.
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const {
        if constexpr (requires { c.neighbour; }) { KOKKOS_ASSERT(c.neighbour < 0); }
        return values(static_cast<std::size_t>(c.owner));
    }
};

/// @brief Reads a coefficient that the caller has already computed for each face.
/// @tparam View Rank-one Kokkos view indexed by local face ID. Strided views are supported.
template<class View>
struct FaceCoefficient {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values; ///< Face coefficients accessible from the execution space of the caller.

    /// @brief Returns the stored coefficient for the given face.
    /// @tparam Context Face context with a local face index.
    /// @param c Face whose coefficient to read.
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return values(static_cast<std::size_t>(c.face)); }
};

/// @brief Linearly interpolates the two cell coefficients to the face.
struct ArithmeticCoefficient {
    /// @brief Forms the weighted arithmetic mean using the mesh interpolation weight.
    /// @tparam Real Floating-point type of the coefficients and the weight.
    /// @param owner Coefficient in the owner cell.
    /// @param neighbour Coefficient in the neighbour cell.
    /// @param owner_weight Weight of the owner value in [0, 1].
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(Real owner, Real neighbour, Real owner_weight) const
    { return owner_weight*owner + (Real{1}-owner_weight)*neighbour; }
};

/// @brief Uses harmonic interpolation so that the resistances on the two sides add in series.
struct HarmonicCoefficient {
    /// @brief Returns the effective coefficient between the two cell centres.
    /// @tparam Real Floating-point type of the coefficients and the weight.
    /// @param owner Positive diffusion coefficient in the owner cell.
    /// @param neighbour Positive diffusion coefficient in the neighbour cell.
    /// @param owner_weight Neighbour-side fraction of the total distance, in [0, 1].
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real operator()(Real owner, Real neighbour, Real owner_weight) const
    {
        // Return invalid data to the operator's asynchronous validation instead of hiding it.
        if (!Kokkos::isfinite(owner) || !(owner > Real{0})) return owner;
        if (!Kokkos::isfinite(neighbour) || !(neighbour > Real{0})) return neighbour;
        // Each resistance uses its own distance fraction. The owner's is 1 - owner_weight.
        return Real{1}/((Real{1}-owner_weight)/owner + owner_weight/neighbour);
    }
};

/// @brief Builds a face coefficient from cell values, interpolating wherever there are two cells.
/// @tparam View Rank-one Kokkos view indexed by local cell ID, including current ghost values.
/// @tparam Interpolation Callable taking the owner value, neighbour value and owner weight.
template<class View, class Interpolation = HarmonicCoefficient>
struct InterpolatedCellCoefficient {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;                   ///< Cell coefficients accessible from the caller's execution space.
    Interpolation interpolation{}; ///< Rule used on faces with a neighbour.

    /// @brief Interpolates two positive cell values or returns the owner value on an exterior face.
    /// @tparam Context Face context with owner and distance. Interior faces also supply neighbour and owner_weight.
    /// @param c Face to sample. The type of distance helps select the arithmetic precision.
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const {
        using Real = std::common_type_t<typename View::non_const_value_type, decltype(c.distance)>;
        const Real owner = static_cast<Real>(values(static_cast<std::size_t>(c.owner)));
        if constexpr (requires { c.neighbour; c.owner_weight; }) {
            if (c.neighbour >= 0) {
                const Real neighbour = static_cast<Real>(values(static_cast<std::size_t>(c.neighbour)));
                // Pass invalid cell data back before interpolation can average it into a usable value.
                if (!Kokkos::isfinite(owner) || !(owner > Real{0})) return owner;
                if (!Kokkos::isfinite(neighbour) || !(neighbour > Real{0})) return neighbour;
                return interpolation(owner, neighbour, static_cast<Real>(c.owner_weight));
            }
        }
        return owner;
    }
};

/// @brief Relates the normal derivative of a scalar to its diffusive flux per unit area.
/// The minus sign in q_n = -coefficient * d(phi)/dn makes diffusion flow down the gradient.
/// @tparam Coefficient Callable that returns the local coefficient for a face context.
template<class Coefficient>
struct IsotropicDiffusion {
    Coefficient coefficient; ///< Provider captured by value. It may read a view or evaluate a model.

    /// @brief Converts a prescribed diffusive flux per unit area to a normal derivative.
    /// @tparam Context Face data required by the coefficient provider.
    /// @tparam Value Floating-point type of the supplied flux.
    /// @param c Face at which to sample the finite, strictly positive coefficient.
    /// @param outward_flux Signed flux per unit area in the direction of the face normal.
    template<class Context, class Value>
    KOKKOS_INLINE_FUNCTION auto normal_gradient(const Context& c, Value outward_flux) const {
        using Real = std::common_type_t<decltype(coefficient(c)), Value>;
        static_assert(std::is_floating_point_v<Real>);
        const Real value = static_cast<Real>(coefficient(c));
        KOKKOS_ASSERT(Kokkos::isfinite(value) && value > Real{0});
        return -static_cast<Real>(outward_flux)/value;
    }

    /// @brief Computes the signed diffusive flux per unit area along the face normal.
    /// @tparam Context Face data required by the coefficient provider.
    /// @tparam Value Floating-point type of the supplied derivative.
    /// @param c Face at which to sample the finite, nonnegative coefficient.
    /// @param normal_gradient_value Derivative along the face normal, outward on a boundary.
    template<class Context, class Value>
    KOKKOS_INLINE_FUNCTION auto outward_flux(const Context& c, Value normal_gradient_value) const {
        using Real = std::common_type_t<decltype(coefficient(c)), Value>;
        static_assert(std::is_floating_point_v<Real>);
        const Real value = static_cast<Real>(coefficient(c));
        KOKKOS_ASSERT(Kokkos::isfinite(value) && value >= Real{0});
        return -value*static_cast<Real>(normal_gradient_value);
    }
};

} // namespace krona::transport
