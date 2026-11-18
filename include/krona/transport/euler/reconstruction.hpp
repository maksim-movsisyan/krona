#pragma once

#include <krona/numerics/reconstruction/face_profiles.hpp>
#include <krona/numerics/reconstruction/schemes/polynomial.hpp>
#include <krona/transport/euler/bindings.hpp>

namespace krona::transport::euler {

/// @brief Adapts face profiles of p, u, v, w and T to the Euler flux operator.
/// @tparam Profiles Bound formulas that can evaluate both traces and the physical-face interior trace.
template <class Profiles>
struct Reconstruction {
    Profiles profiles;
    using inner_formula = decltype(profiles.inner);
    using boundary_formula = decltype(profiles.boundary);

    [[nodiscard]] KOKKOS_INLINE_FUNCTION const auto& values() const {
        return profiles.values;
    }

    template <class Real>
    KOKKOS_INLINE_FUNCTION auto interior(std::size_t f) const {
        return profiles.template inner_values<Real>(f);
    }

    template <class Real>
    KOKKOS_INLINE_FUNCTION auto boundary(std::size_t f) const {
        return profiles.template boundary_values<Real>(f);
    }

    /// @brief Checks the primitive field and the sampled inputs of the requested formulas.
    /// @tparam Exec Execution space that will read the reconstruction inputs.
    template <class Exec>
    void validate(std::size_t cells, std::size_t extended, bool with_boundary) const {
        detail::validate_pack<Exec, false>(values(), with_boundary ? extended : cells);
        profiles.template validate<Exec>(cells, with_boundary ? extended : cells, with_boundary,
                                         "Euler: undersized reconstruction provider");
    }
};

} // namespace krona::transport::euler
