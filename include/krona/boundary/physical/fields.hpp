#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/fields/component.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <krona/mpi/error.hpp>
#include <type_traits>

namespace krona::boundary::physical {

/// @brief Marks a primitive-field binding that contains just pressure and velocity.
struct NoTemperature {};

namespace detail {

/// @brief Checks whether one component can supply the required cells or boundary slots.
/// @tparam Exec Execution space that will use the component.
/// @tparam Writable True when a kernel will write this component.
/// @tparam V Rank-one floating-point Kokkos view.
/// @param v Component view, including its stride.
/// @param count Minimum number of entries the caller needs.
template<class Exec, bool Writable, class V>
void validate_component(V v, std::size_t count) {
    static_assert(Kokkos::is_view<V>::value && V::rank == 1);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename V::memory_space>::accessible);
    static_assert(std::is_floating_point_v<typename V::non_const_value_type>);
    if constexpr (Writable) static_assert(!std::is_const_v<typename V::value_type>);
    mpi::require(v.extent(0)>=count,"Physical boundary: invalid or undersized primitive component");
    (void)fields::descriptor_from_view(v);
}

/// @brief Checks that a finite value fits the destination type before a storage conversion.
/// @tparam To Floating-point type stored by the field.
/// @tparam From Arithmetic type of the value being stored.
/// @param value Value produced by the boundary closure.
template<class To,class From>
KOKKOS_INLINE_FUNCTION bool representable(From value) {
    using R=std::common_type_t<To,From>;
    constexpr R limit = static_cast<R>(Kokkos::finite_max_v<To>);
    return Kokkos::isfinite(value)&&static_cast<R>(value)<=limit&&static_cast<R>(value)>=-limit;
}

}

/// @brief Groups the views of one primitive field, with support for split or interleaved storage.
/// @tparam P Pressure component view.
/// @tparam U X velocity component view.
/// @tparam V Y velocity component view.
/// @tparam W Z velocity component view.
/// @tparam T Temperature component view or NoTemperature for mechanical fields.
template<class P, class U, class V, class W, class T = NoTemperature>
struct PrimitiveViews {
    static constexpr bool thermal = !std::is_same_v<T, NoTemperature>;
    P pressure; U u; V v; W w; T temperature;

    /// @brief Loads the primitive state at one index into the requested arithmetic type.
    /// @tparam Real Floating-point type used by the boundary calculation.
    /// @param i Local cell or extended-field index.
    template<class Real>
    KOKKOS_INLINE_FUNCTION Primitive<Real, thermal> read(std::size_t i) const {
        Primitive<Real, thermal> s;
        s.pressure = static_cast<Real>(pressure(i));
        s.velocity = {static_cast<Real>(u(i)),static_cast<Real>(v(i)),static_cast<Real>(w(i))};
        if constexpr (thermal) s.temperature = static_cast<Real>(temperature(i));
        return s;
    }

    /// @brief Checks each value before writing, so a rejected face leaves all its components intact.
    /// @tparam Real Floating-point type of the computed state.
    /// @param s State to check against all destination component types.
    template<class Real>
    KOKKOS_INLINE_FUNCTION bool can_store(const Primitive<Real,thermal>& s) const {
        bool ok=detail::representable<typename P::non_const_value_type>(s.pressure)&&
            detail::representable<typename U::non_const_value_type>(s.velocity.x)&&
            detail::representable<typename V::non_const_value_type>(s.velocity.y)&&
            detail::representable<typename W::non_const_value_type>(s.velocity.z);
        if constexpr (thermal) ok=ok&&detail::representable<typename T::non_const_value_type>(s.temperature);
        return ok;
    }

    /// @brief Writes one state after the caller checks can_store().
    /// @tparam Real Floating-point type of the computed state.
    /// @param i Extended-field index to write.
    /// @param s State to convert into the component storage types.
    template<class Real>
    KOKKOS_INLINE_FUNCTION void write(std::size_t i, const Primitive<Real, thermal>& s) const {
        pressure(i) = static_cast<typename P::non_const_value_type>(s.pressure);
        u(i) = static_cast<typename U::non_const_value_type>(s.velocity.x);
        v(i) = static_cast<typename V::non_const_value_type>(s.velocity.y);
        w(i) = static_cast<typename W::non_const_value_type>(s.velocity.z);
        if constexpr (thermal) temperature(i) = static_cast<typename T::non_const_value_type>(s.temperature);
    }

    /// @brief Validates every component view on the host.
    /// @tparam Exec Execution space that will use the views.
    /// @tparam Writable True for a binding that will receive face values.
    /// @param count Minimum extent required in each component.
    template<class Exec, bool Writable = false>
    void validate(std::size_t count) const {
        detail::validate_component<Exec,Writable>(pressure,count);
        detail::validate_component<Exec,Writable>(u,count);
        detail::validate_component<Exec,Writable>(v,count);
        detail::validate_component<Exec,Writable>(w,count);
        if constexpr (thermal) detail::validate_component<Exec,Writable>(temperature,count);
    }
};

/// @brief Binds rank-one views of pressure and the three Cartesian velocity components.
template<class P,class U,class V,class W>
auto primitive_components(P p,U u,V v,W w) { return PrimitiveViews<P,U,V,W>{p,u,v,w,{}}; }

/// @brief Binds rank-one views of pressure, Cartesian velocity and temperature.
template<class P,class U,class V,class W,class T>
auto primitive_components(P p,U u,V v,W w,T t) { return PrimitiveViews<P,U,V,W,T>{p,u,v,w,t}; }

/// @brief Extracts the components from a scalar pressure field and a three-component velocity field.
template<class P,class Velocity>
auto primitive_views(P p,Velocity velocity) {
    static_assert(P::components == 1 && Velocity::components == 3);
    return primitive_components(p.component(0),velocity.component(0),velocity.component(1),velocity.component(2));
}
/// @brief Extracts those same components and adds a scalar temperature field.
template<class P,class Velocity,class T>
auto primitive_views(P p,Velocity velocity,T temperature) {
    static_assert(P::components == 1 && Velocity::components == 3 && T::components == 1);
    return primitive_components(p.component(0),velocity.component(0),velocity.component(1),velocity.component(2),temperature.component(0));
}

/// @brief Adapts primitive fields so the manager can use the owner cell as its interior state.
/// @tparam Views Primitive component binding with a read() method.
template<class Views>
struct OwnerReader {
    Views fields;
    
    /// @brief Checks enough field entries are readable for all owned cells.
    /// @tparam Exec Execution space that will read the fields.
    /// @param count Number of owned cells.
    template<class Exec>
    void validate(std::size_t count) const { fields.template validate<Exec>(count); }

    /// @brief Reads the adjacent owned cell's primitive state.
    /// @tparam Real Floating-point type of the face context and state.
    /// @tparam Condition Current patch condition, accepted to match the manager's reader interface.
    /// @param c Face context supplying the owner cell ID.
    template<class Real,class Condition>
    KOKKOS_INLINE_FUNCTION auto operator()(const FaceContext<Real>& c,const Condition&) const
    { return fields.template read<Real>(static_cast<std::size_t>(c.owner)); }
};

} // namespace krona::boundary::physical
