#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/boundary/profiles.hpp>

#include <type_traits>

namespace krona::boundary::physical {

/// @brief Parameter source producing one ordinary numeric condition at the current face/time.
/// The wrapper stores its provider only; numeric condition objects are constructed on demand.
template<class Condition, class Provider>
struct Profiled {
    using condition_type = Condition;
    Provider parameters;
    template<class Exec>
    void validate(PatchRange range) const { profile::validate<Exec>(parameters, range); }
};
template<class C> struct ConditionType { using type = C; static constexpr bool profiled = false; };
template<class C, class P> struct ConditionType<Profiled<C, P>> { using type = C; static constexpr bool profiled = true; };
template<class C> using condition_type_t = typename ConditionType<std::remove_cvref_t<C>>::type;
template<class C> inline constexpr bool is_profiled = ConditionType<std::remove_cvref_t<C>>::profiled;

/// @brief A custom provider returns exactly Condition from provider(context).
template<class Condition, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto profiled(Provider provider)
{ return Profiled<Condition, Provider>{provider}; }

template<class C, class Context>
KOKKOS_INLINE_FUNCTION auto sample_condition(const C& condition, const Context& context)
{
    if constexpr (is_profiled<C>) {
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(condition.parameters(context))>, condition_type_t<C>>,
                      "Physical profile: provider must return its declared condition type");
        return condition.parameters(context);
    } else return condition;
}

/// @brief Mechanical wall projection; factory profiles do not read thermal parameters here.
template<class C,class Context>
KOKKOS_INLINE_FUNCTION auto sample_wall_velocity(const C& condition,const Context& context)
{
    if constexpr (is_profiled<C> && requires { condition.parameters.wall_velocity(context); })
        return condition.parameters.wall_velocity(context);
    else return sample_condition(condition,context).velocity;
}

template<class C> inline constexpr bool requires_flux_law=false;
template<class R> inline constexpr bool requires_flux_law<HeatFluxWall<R>> =true;
template<class C,class P> inline constexpr bool requires_flux_law<Profiled<C,P>> =requires_flux_law<C>;

namespace detail {
template<class R, class P, class C>
KOKKOS_INLINE_FUNCTION R scalar_parameter(const P& p, const C& c)
{ return static_cast<R>(profile::sample(p, c)); }
template<class R, class P, class C>
KOKKOS_INLINE_FUNCTION Vector<R> vector_parameter(const P& p, const C& c)
{ return math::cast<R>(profile::sample(p, c)); }

template<class R, template<class> class Wall, class V>
struct MechanicalParameters {
    V velocity;
    template<class C> KOKKOS_INLINE_FUNCTION auto wall_velocity(const C& c) const { return vector_parameter<R>(velocity,c); }
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return Wall<R>{vector_parameter<R>(velocity, c)}; }
    template<class Exec> void validate(PatchRange range) const { profile::validate<Exec>(velocity, range); }
};
template<class R, template<class> class Wall, class P, class V>
struct ThermalWallParameters {
    P parameter; V velocity;
    template<class C> KOKKOS_INLINE_FUNCTION auto wall_velocity(const C& c) const { return vector_parameter<R>(velocity,c); }
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return Wall<R>{scalar_parameter<R>(parameter,c),vector_parameter<R>(velocity,c)}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(parameter,range); profile::validate<Exec>(velocity,range); }
};
template<class R, class V, class T>
struct VelocityParameters {
    V velocity; T temperature;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return VelocityInlet<R>{vector_parameter<R>(velocity,c),scalar_parameter<R>(temperature,c)}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(velocity,range); profile::validate<Exec>(temperature,range); }
};
template<class R, class P, class T, class D>
struct TotalParameters {
    P pressure; T temperature; D direction; bool allow_choking;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return TotalConditionsInlet<R>{scalar_parameter<R>(pressure,c),scalar_parameter<R>(temperature,c),
                                     vector_parameter<R>(direction,c),allow_choking}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(pressure,range); profile::validate<Exec>(temperature,range); profile::validate<Exec>(direction,range); }
};
template<class R, class P, class T, class V, class M, class D>
struct SupersonicParameters {
    P pressure; T temperature; V velocity; M mach; D direction;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return SupersonicInlet<R>{scalar_parameter<R>(pressure,c),scalar_parameter<R>(temperature,c),
        vector_parameter<R>(velocity,c),scalar_parameter<R>(mach,c),vector_parameter<R>(direction,c)}; }
    template<class Exec> void validate(PatchRange range) const
    {
        profile::validate<Exec>(pressure,range); profile::validate<Exec>(temperature,range);
        profile::validate<Exec>(velocity,range); profile::validate<Exec>(mach,range); profile::validate<Exec>(direction,range);
    }
};
template<class R, class P, class B>
struct OutletParameters {
    P pressure; B backflow;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return PressureOutlet<R>{scalar_parameter<R>(pressure,c),sample_condition(backflow,c)}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(pressure,range); profile::validate<Exec>(backflow,range); }
};
template<class R, class Reservoir>
struct ReservoirParameters {
    Reservoir reservoir;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { Backflow<R> result; result.mode=BackflowMode::TotalConditions; result.reservoir=sample_condition(reservoir,c); return result; }
    template<class Exec> void validate(PatchRange range) const { profile::validate<Exec>(reservoir,range); }
};
template<class R, class D, class T>
struct ReverseParameters {
    D direction; T temperature;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    {
        Backflow<R> result; result.mode=BackflowMode::DirectionTemperature;
        result.direction=vector_parameter<R>(direction,c); result.temperature=scalar_parameter<R>(temperature,c); return result;
    }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(direction,range); profile::validate<Exec>(temperature,range); }
};
} // namespace detail

template<class R=ComputeReal, class V>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto slip_wall(V velocity)
{ return profiled<SlipWall<R>>(detail::MechanicalParameters<R,SlipWall,V>{velocity}); }
template<class R=ComputeReal, class V=Vector<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto adiabatic_wall(V velocity={})
{ return profiled<AdiabaticWall<R>>(detail::MechanicalParameters<R,AdiabaticWall,V>{velocity}); }
template<class R=ComputeReal, class T, class V=Vector<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto isothermal_wall(T temperature, V velocity={})
{ return profiled<IsothermalWall<R>>(detail::ThermalWallParameters<R,IsothermalWall,T,V>{temperature,velocity}); }
template<class R=ComputeReal, class F, class V=Vector<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto heat_flux_wall(F flux, V velocity={})
{ return profiled<HeatFluxWall<R>>(detail::ThermalWallParameters<R,HeatFluxWall,F,V>{flux,velocity}); }
template<class R=ComputeReal, class V, class T=R>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto velocity_inlet(V velocity, T temperature={})
{ return profiled<VelocityInlet<R>>(detail::VelocityParameters<R,V,T>{velocity,temperature}); }
template<class R=ComputeReal, class P, class T, class D>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto total_conditions_inlet(P pressure,T temperature,D direction,bool allow_choking=false)
{ return profiled<TotalConditionsInlet<R>>(detail::TotalParameters<R,P,T,D>{pressure,temperature,direction,allow_choking}); }
template<class R=ComputeReal, class P, class T, class V>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto supersonic_inlet(P pressure,T temperature,V velocity)
{ return profiled<SupersonicInlet<R>>(detail::SupersonicParameters<R,P,T,V,R,Vector<R>>{pressure,temperature,velocity,R{0},{}}); }
template<class R=ComputeReal, class P, class T, class M, class D>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto supersonic_mach_inlet(P pressure,T temperature,M mach,D direction)
{ return profiled<SupersonicInlet<R>>(detail::SupersonicParameters<R,P,T,Vector<R>,M,D>{pressure,temperature,{},mach,direction}); }
template<class R=ComputeReal, class P, class B=Backflow<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pressure_outlet(P pressure,B backflow={})
{ return profiled<PressureOutlet<R>>(detail::OutletParameters<R,P,B>{pressure,backflow}); }
template<class R=ComputeReal, class Reservoir>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto total_backflow(Reservoir reservoir)
{ return profiled<Backflow<R>>(detail::ReservoirParameters<R,Reservoir>{reservoir}); }
template<class R=ComputeReal, class D, class T=R>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto directional_backflow(D direction,T temperature={})
{ return profiled<Backflow<R>>(detail::ReverseParameters<R,D,T>{direction,temperature}); }

} // namespace krona::boundary::physical
