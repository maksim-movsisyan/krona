#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/mpi/error.hpp>
#include <type_traits>

namespace krona::boundary::physical {

struct NoTemperature {};

namespace detail {

template<class Exec, bool Writable, class V>
void validate_component(V v, std::size_t count)
{
    static_assert(Kokkos::is_view<V>::value && V::rank == 1);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename V::memory_space>::accessible);
    static_assert(std::is_floating_point_v<typename V::non_const_value_type>);
    if constexpr (Writable) static_assert(!std::is_const_v<typename V::value_type>);
    mpi::require(v.extent(0) >= count && (v.extent(0) == 0 || (v.data() && v.stride(0) > 0)),
                 "Physical boundary: invalid or undersized primitive component");
}

}

/// Borrowed rank-one components; works with split velocity arrays and strided AoS subviews.
template<class P, class U, class V, class W, class T = NoTemperature>
struct PrimitiveViews {
    static constexpr bool thermal = !std::is_same_v<T, NoTemperature>;
    P pressure; U u; V v; W w; T temperature;

    template<class Real>
    KOKKOS_INLINE_FUNCTION Primitive<Real, thermal> read(std::size_t i) const
    {
        Primitive<Real, thermal> s;
        s.pressure = static_cast<Real>(pressure(i));
        s.velocity = {static_cast<Real>(u(i)),static_cast<Real>(v(i)),static_cast<Real>(w(i))};
        if constexpr (thermal) s.temperature = static_cast<Real>(temperature(i));
        return s;
    }
    template<class Real>
    KOKKOS_INLINE_FUNCTION void write(std::size_t i, const Primitive<Real, thermal>& s) const
    {
        pressure(i) = static_cast<typename P::non_const_value_type>(s.pressure);
        u(i) = static_cast<typename U::non_const_value_type>(s.velocity.x);
        v(i) = static_cast<typename V::non_const_value_type>(s.velocity.y);
        w(i) = static_cast<typename W::non_const_value_type>(s.velocity.z);
        if constexpr (thermal) temperature(i) = static_cast<typename T::non_const_value_type>(s.temperature);
    }
    template<class Exec, bool Writable = false>
    void validate(std::size_t count) const
    {
        detail::validate_component<Exec,Writable>(pressure,count);
        detail::validate_component<Exec,Writable>(u,count);
        detail::validate_component<Exec,Writable>(v,count);
        detail::validate_component<Exec,Writable>(w,count);
        if constexpr (thermal) detail::validate_component<Exec,Writable>(temperature,count);
    }
};

template<class P,class U,class V,class W>
auto primitive_components(P p,U u,V v,W w) { return PrimitiveViews<P,U,V,W>{p,u,v,w,{}}; }
// note: all of them are views
template<class P,class U,class V,class W,class T>
auto primitive_components(P p,U u,V v,W w,T t) { return PrimitiveViews<P,U,V,W,T>{p,u,v,w,t}; }

template<class P,class Velocity>
auto primitive_views(P p,Velocity velocity)
{
    static_assert(P::components == 1 && Velocity::components == 3);
    return primitive_components(p.component(0),velocity.component(0),velocity.component(1),velocity.component(2));
}
template<class P,class Velocity,class T>
auto primitive_views(P p,Velocity velocity,T temperature)
{
    static_assert(P::components == 1 && Velocity::components == 3 && T::components == 1);
    return primitive_components(p.component(0),velocity.component(0),velocity.component(1),velocity.component(2),temperature.component(0));
}

template<class Views>
struct OwnerReader {
    Views fields;
    template<class Exec>
    void validate(std::size_t count) const { fields.template validate<Exec>(count); }
    template<class Real,class Condition>
    KOKKOS_INLINE_FUNCTION auto operator()(const FaceContext<Real>& c,const Condition&) const
    { return fields.template read<Real>(static_cast<std::size_t>(c.owner)); }
};

} // namespace krona::boundary::physical
