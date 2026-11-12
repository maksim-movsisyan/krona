#pragma once

#include <krona/numerics/reconstruction/schemes.hpp>
#include <krona/numerics/reconstruction/detail/contracts.hpp>

#include <utility>

namespace krona::numerics::reconstruction {
namespace detail {
template<class R, class Scheme, class Side, class D, std::size_t... C>
KOKKOS_INLINE_FUNCTION auto side_values(Scheme scheme, const Side& side, const D& d, std::index_sequence<C...>)
{ return Kokkos::Array<R, D::components>{(static_cast<R>(d.values(side.cell, C))+scheme.template increment<R>(side, d, C))...}; }

template<class R, class Scheme, class Face, class D, std::size_t... C>
KOKKOS_INLINE_FUNCTION auto face_values(Scheme scheme, const Face& face, const D& d, std::index_sequence<C...>) {
    return FaceValues<R, D::components>{{(static_cast<R>(d.values(face.left.cell, C))+scheme.template increment<R>(face.left, d, C))...},
                                       {(static_cast<R>(d.values(face.right.cell, C))+scheme.template increment<R>(face.right, d, C))...}};
}
} // namespace detail

/// @brief Reconstructs one interior trace inside the caller's device kernel.
/// Inputs must be finite, accessible and alive; no allocation, launch, fence or MPI.
template<class Real = ComputeReal, class Scheme, class Side, class D>
KOKKOS_INLINE_FUNCTION auto evaluate_side(Scheme scheme, const Side& side, const D& d) {
    static_assert(std::is_floating_point_v<Real>);
    detail::check_input_components<Scheme, D>();
    return detail::side_values<Real>(scheme, side, d, std::make_index_sequence<D::components>{});
}

/// @brief Reconstructs both traces in component-pack order; does not select an upwind side.
template<class Real = ComputeReal, class Scheme, class Face, class D>
KOKKOS_INLINE_FUNCTION auto evaluate(Scheme scheme, const Face& face, const D& d) {
    static_assert(std::is_floating_point_v<Real>);
    detail::check_input_components<Scheme, D>();
    return detail::face_values<Real>(scheme, face, d, std::make_index_sequence<D::components>{});
}

} // namespace krona::numerics::reconstruction
