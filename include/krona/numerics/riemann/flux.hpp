#pragma once

/// @file
/// @brief Riemann solvers, state adapters and helpers for integrating flux over a face.

#include <krona/numerics/riemann/detail/validation.hpp>
#include <krona/numerics/riemann/eos.hpp>
#include <krona/numerics/riemann/hllc.hpp>
#include <krona/numerics/riemann/roe.hpp>

namespace krona::numerics::riemann {

/// @brief Computes a face-integrated flux from an oriented area vector.
/// The vector length gives the area. Only the flux is scaled by that area.
/// @tparam Solver Callable such as Hllc<> that returns a flux per unit area.
/// @tparam R Scalar type of the states, geometry and result.
/// @param solver Solver to apply to the normalized face direction.
/// @param left State on the side the area vector points away from.
/// @param right State on the side the area vector points towards.
/// @param area_vector Face area times the unit normal from left to right.
template<class Solver,class R>
KOKKOS_INLINE_FUNCTION Result<R> integrated_flux(const Solver& solver,State<R> left,State<R> right,Vector<R> area_vector) {
    const R area=Kokkos::sqrt(area_vector.norm2());
    if(!detail::finite(area_vector)||!Kokkos::isfinite(area)||!(area>R{0}))
        return detail::failure<R>(Status::InvalidNormal);
    auto result=solver(left,right,area_vector*(R{1}/area));
    if(result.valid()){
        result.flux=result.flux*area;
        if(!detail::finite(result.flux))return detail::failure<R>(Status::NonfiniteFlux);
    }
    return result;
}

/// @brief Computes a face-integrated flux when the unit normal and area are already available.
/// The flux is multiplied by the area. The signal speed remains in velocity units.
/// @tparam Solver Callable such as Hllc<> that returns a flux per unit area.
/// @tparam R Scalar type of the states, geometry and result.
/// @param solver Solver to apply to the supplied unit normal.
/// @param left State on the side the normal points away from.
/// @param right State on the side the normal points towards.
/// @param normal Unit normal pointing from left to right.
/// @param area Positive face area.
template<class Solver, class R>
KOKKOS_INLINE_FUNCTION Result<R> integrated_flux(
    const Solver& solver, State<R> left, State<R> right, Vector<R> normal, R area) {
    if (!Kokkos::isfinite(area) || !(area > R{0}))
        return detail::failure<R>(Status::InvalidNormal);

    auto result = solver(left, right, normal);
    if (result.valid()) {
        result.flux = result.flux * area;
        if (!detail::finite(result.flux))
            return detail::failure<R>(Status::NonfiniteFlux);
    }
    return result;
}

} // namespace krona::numerics::riemann
