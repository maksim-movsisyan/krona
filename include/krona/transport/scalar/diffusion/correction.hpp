#pragma once

#include <krona/boundary/correction.hpp>
#include <krona/transport/scalar/detail/views.hpp>

#include <type_traits>

namespace krona::transport::scalar::diffusion {

using NoCorrection = boundary::NoCorrection;
 
/// @brief Borrows three native rank-one gradient components, including strided field components.
/// Owned and real MPI ghost gradients must be current before any corrected interior operation.
template<class X, class Y, class Z>
struct CellGradients {
    X x; Y y; Z z;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell) const
    {
        using R = std::common_type_t<typename X::non_const_value_type,
            typename Y::non_const_value_type, typename Z::non_const_value_type>;
        return Kokkos::Array<R, 3>{static_cast<R>(x(cell)), static_cast<R>(y(cell)), static_cast<R>(z(cell))};
    }
};

/// @brief Explicit nonorthogonal correction from an externally supplied gradient(cell)[axis].
/// Interior gradients use actual owner interpolation weights; no midpoint fallback is supplied.
/// operator()(boundary context) is also a scalar-boundary reconstruction-increment provider.
template<class Gradients>
struct NonOrthogonalCorrection {
    Gradients gradients;

    template<class Context, class R>
    KOKKOS_INLINE_FUNCTION R interior(const Context& c, R tx, R ty, R tz) const
    {
        const auto p = gradients(static_cast<std::size_t>(c.owner));
        const auto n = gradients(static_cast<std::size_t>(c.neighbour));
        const R w = static_cast<R>(c.owner_weight);
        return tx*(w*static_cast<R>(p[0])+(R{1}-w)*static_cast<R>(n[0])) +
               ty*(w*static_cast<R>(p[1])+(R{1}-w)*static_cast<R>(n[1])) +
               tz*(w*static_cast<R>(p[2])+(R{1}-w)*static_cast<R>(n[2]));
    }

    template<class R>
    KOKKOS_INLINE_FUNCTION R operator()(const boundary::FaceContext<R>& c) const
    {
        const auto g = gradients(static_cast<std::size_t>(c.owner));
        return c.tx*static_cast<R>(g[0])+c.ty*static_cast<R>(g[1])+c.tz*static_cast<R>(g[2]);
    }
};

namespace detail {
template<class C> inline constexpr bool corrected = !std::is_same_v<C, NoCorrection>;

template<class Exec, class G> void validate_gradients(const G&, std::size_t) {}
template<class Exec, class X, class Y, class Z>
void validate_gradients(const CellGradients<X, Y, Z>& g, std::size_t count)
{
    scalar::detail::validate_view<Exec>(g.x, count);
    scalar::detail::validate_view<Exec>(g.y, count);
    scalar::detail::validate_view<Exec>(g.z, count);
}
} // namespace detail

} // namespace krona::transport::scalar::diffusion
