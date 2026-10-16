#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>
#include <type_traits>

namespace krona::boundary::physical {

struct NoTemperature {};

/// @brief Allocation-free density descriptor for a constant-density field adapter.
template<class Real>
struct ConstantDensity {
    Real value = 1;
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};

namespace detail {
template<class Exec, class View>
void require_view(const View& v, std::size_t count)
{
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
    static_assert(!std::is_const_v<typename View::value_type>, "Boundary field must be writable");
    static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
    mpi::require(v.extent(0) >= count && (count == 0 || (v.data() && v.stride(0) > 0)),
                 "Physical boundary: field extent or storage is invalid");
}
template<class Real>
void require_density(ConstantDensity<Real> density)
{ mpi::require(std::isfinite(density.value) && density.value > Real{0}, "Physical boundary: invalid constant density"); }
}

/// @brief Borrowed rank-one fields; component subviews may have LayoutStride.
template<class Real, class Density, class X, class Y, class Z, class Pressure, class Thermal = NoTemperature>
struct PrimitiveFields {
    static constexpr bool has_temperature = !std::is_same_v<Thermal, NoTemperature>;
    static constexpr bool constant_density = std::is_same_v<Density, ConstantDensity<Real>>;
    using state_type = State<Real, has_temperature>;
    Density density;
    X velocity_x; Y velocity_y; Z velocity_z;
    Pressure pressure;
    Thermal temperature;

    KOKKOS_INLINE_FUNCTION state_type operator()(std::size_t i) const
    {
        state_type q{static_cast<Real>(density(i)), static_cast<Real>(pressure(i)),
                     {static_cast<Real>(velocity_x(i)), static_cast<Real>(velocity_y(i)),
                      static_cast<Real>(velocity_z(i))}};
        if constexpr (has_temperature) q.temperature = static_cast<Real>(temperature(i));
        return q;
    }

    KOKKOS_INLINE_FUNCTION void store(std::size_t i, const state_type& q) const
    {
        if constexpr (Kokkos::is_view<Density>::value)
            density(i) = static_cast<typename Density::non_const_value_type>(q.density);
        velocity_x(i) = static_cast<typename X::non_const_value_type>(q.velocity.x);
        velocity_y(i) = static_cast<typename Y::non_const_value_type>(q.velocity.y);
        velocity_z(i) = static_cast<typename Z::non_const_value_type>(q.velocity.z);
        pressure(i) = static_cast<typename Pressure::non_const_value_type>(q.pressure);
        if constexpr (has_temperature)
            temperature(i) = static_cast<typename Thermal::non_const_value_type>(q.temperature);
    }

    template<class Exec>
    void validate(std::size_t count) const
    {
        if constexpr (Kokkos::is_view<Density>::value) detail::require_view<Exec>(density, count);
        else {
            static_assert(std::is_same_v<Density, ConstantDensity<Real>>);
            detail::require_density(density);
        }
        detail::require_view<Exec>(velocity_x, count); detail::require_view<Exec>(velocity_y, count);
        detail::require_view<Exec>(velocity_z, count); detail::require_view<Exec>(pressure, count);
        if constexpr (has_temperature) detail::require_view<Exec>(temperature, count);
    }
};

template<class Real = ComputeReal, class D, class X, class Y, class Z, class P, class T = NoTemperature>
[[nodiscard]] auto make_fields(D density, X x, Y y, Z z, P pressure, T temperature = {})
{ return PrimitiveFields<Real, D, X, Y, Z, P, T>{density, x, y, z, pressure, temperature}; }

template<class Real = ComputeReal, class X, class Y, class Z, class P, class T = NoTemperature>
[[nodiscard]] auto make_incompressible_fields(Real density, X x, Y y, Z z, P pressure, T temperature = {})
{ return make_fields<Real>(ConstantDensity<Real>{density}, x, y, z, pressure, temperature); }

} // namespace krona::boundary::physical
