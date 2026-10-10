#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>

namespace krona::transport::scalar {

/// @brief Constant source density s in -div(D*grad(phi)) = s, before cell-volume integration.
template<class Real = ComputeReal>
struct UniformSource {
    Real value = 0;
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};

/// @brief Cell-indexed source density; only the owned prefix is read.
template<class View>
struct CellSource {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell) const { return values(cell); }
};

/// @brief Two-point approximation; disabling the check does not add a nonorthogonal correction.
struct DiffusionOptions {
    bool require_orthogonal = true;
    double orthogonality_tolerance = 1.0e-6;
};

/// @brief Global physical balance of the supplied field, not the internal solver vector.
struct BalanceResult {
    double source_integral = 0;
    double outward_boundary_flux = 0;
    double defect = 0; // outward_boundary_flux - source_integral
};

} // namespace krona::transport::scalar
