#pragma once

#include <krona/numerics/reconstruction/face_profiles.hpp>
#include <krona/numerics/reconstruction/schemes/polynomial.hpp>
#include <krona/transport/detail/component_pack.hpp>

namespace krona::transport::euler {

/// @brief Adapts face profiles of p, u, v, w and T to the Euler flux operator.
/// @tparam Profiles Bound formulas that can evaluate both traces and the physical-face interior trace.
template <class Profiles>
struct Reconstruction {
    Profiles profiles; ///< Bound cell primitives, stencil and formulas.
    using inner_formula = decltype(Profiles::inner);
    using boundary_formula = decltype(Profiles::boundary);

    /// @brief Returns the primitive pack shared by both formulas.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION const auto& values() const {
        return profiles.values;
    }

    /// @brief Evaluates both traces at the centroid of an inner face.
    /// @tparam Real Scalar used to evaluate the formulas.
    /// @param f Local face index in the inner range.
    template <class Real>
    KOKKOS_INLINE_FUNCTION auto interior(std::size_t f) const {
        return profiles.template inner_values<Real>(f);
    }

    /// @brief Evaluates the owner's trace at an exterior boundary face centroid.
    /// @tparam Real Scalar used to evaluate the formula.
    /// @param f Local face index in the boundary range.
    template <class Real>
    KOKKOS_INLINE_FUNCTION auto boundary(std::size_t f) const {
        return profiles.template boundary_values<Real>(f);
    }

    /// @brief Checks the primitive field and the sampled inputs of the requested formulas.
    /// @tparam Exec Execution space that will read the reconstruction inputs.
    /// @param cells Number of packed cell primitives, including MPI ghosts.
    /// @param extended Minimum extent for a boundary launch. Includes face slots if the closure needs them.
    /// @param with_boundary Whether to check the boundary formula inputs as well.
    template <class Exec>
    void validate(std::size_t cells, std::size_t extended, bool with_boundary) const {
        transport::detail::validate_pack<Exec, 5, false>(values(), with_boundary ? extended : cells);
        profiles.template validate<Exec>(cells, with_boundary ? extended : cells, with_boundary,
                                         "Euler: undersized reconstruction provider");
    }
};

} // namespace krona::transport::euler
