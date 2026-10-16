#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/mpi/error.hpp>

#include <type_traits>

namespace krona::boundary::physical {

/// @brief Adds the explicit tangential temperature reconstruction from owned-cell gradients.
template<class X, class Y, class Z>
struct TemperatureGradientCorrection {
    X x; Y y; Z z;

    template<class Real>
    KOKKOS_INLINE_FUNCTION Context<Real> operator()(Context<Real> c) const
    {
        const auto i = static_cast<std::size_t>(c.owner);
        const auto& t = c.tangential_displacement;
        using Accumulator = std::common_type_t<Real, GeometryReal, typename X::non_const_value_type,
                                              typename Y::non_const_value_type, typename Z::non_const_value_type>;
        c.temperature_tangential_increment = static_cast<Real>(
            static_cast<Accumulator>(t.x) * static_cast<Accumulator>(x(i)) +
            static_cast<Accumulator>(t.y) * static_cast<Accumulator>(y(i)) +
            static_cast<Accumulator>(t.z) * static_cast<Accumulator>(z(i)));
        return c;
    }
    template<class Exec>
    void validate(std::size_t count) const
    { validate_view<Exec>(x, count); validate_view<Exec>(y, count); validate_view<Exec>(z, count); }
private:
    template<class Exec, class View>
    static void validate_view(const View& v, std::size_t count)
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
        static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
        mpi::require(v.extent(0) >= count && (count == 0 || (v.data() && v.stride(0) > 0)),
                     "Physical boundary: temperature gradient extent or storage is invalid");
    }
};

} // namespace krona::boundary::physical
