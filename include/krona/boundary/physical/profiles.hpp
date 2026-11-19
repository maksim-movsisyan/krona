#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/boundary/profiles.hpp>

#include <type_traits>

namespace krona::boundary::physical {

/// @brief Stores a provider that produces a complete numeric condition at the current face and time.
/// @tparam Condition Numeric condition returned by the provider.
/// @tparam Provider Device-callable object accepting the face context.
template<class Condition, class Provider>
struct Profiled {
    using condition_type = Condition;
    Provider parameters; ///< Returns the complete Condition when called with the face context.

    /// @brief Asks the provider to check any patch metadata it borrows.
    /// @tparam Exec Execution space that will run the provider.
    /// @param range Local face range where the condition will be used.
    template<class Exec>
    void validate(PatchRange range) const { profile::validate<Exec>(parameters, range); }
};

/// @brief Exposes the numeric condition type for both direct and profiled conditions.
/// @tparam C Concrete type stored on the patch.
template<class C> struct ConditionType { using type = C; static constexpr bool profiled = false; };
template<class C, class P> struct ConditionType<Profiled<C, P>> { using type = C; static constexpr bool profiled = true; };
template<class C> using condition_type_t = typename ConditionType<std::remove_cvref_t<C>>::type;
template<class C> inline constexpr bool is_profiled = ConditionType<std::remove_cvref_t<C>>::profiled;

/// @brief Binds a custom provider that supplies a complete numeric condition for a face.
/// @tparam Condition Exact numeric type returned by the provider.
/// @tparam Provider Device-callable object supplying the condition.
/// @param provider Provider to store in the wrapper.
template<class Condition, class Provider>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto profiled(Provider provider)
{ return Profiled<Condition, Provider>{provider}; }

/// @brief Returns the condition itself or samples the provider to build its numeric form.
/// @tparam C Direct condition or Profiled wrapper.
/// @tparam Context Face context accepted by the provider.
/// @param condition Condition assigned to the patch.
/// @param context Face and time to evaluate.
template<class C, class Context>
KOKKOS_INLINE_FUNCTION auto sample_condition(const C& condition, const Context& context) {
    if constexpr (is_profiled<C>) {
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(condition.parameters(context))>, condition_type_t<C>>,
                      "Physical profile: provider must return its declared condition type");
        return condition.parameters(context);
    } else return condition;
}

/// @brief Reads the wall velocity for the inviscid mirror. Factory profiles sample just the velocity.
/// @tparam C Direct wall condition or a profiled wall.
/// @tparam Context Face context accepted by the velocity provider.
/// @param condition Wall condition assigned to the patch.
/// @param context Face and time to evaluate.
template<class C,class Context>
KOKKOS_INLINE_FUNCTION auto sample_wall_velocity(const C& condition,const Context& context) {
    if constexpr (is_profiled<C> && requires { condition.parameters.wall_velocity(context); })
        return condition.parameters.wall_velocity(context);
    else return sample_condition(condition,context).velocity;
}

/// @brief Marks a condition that requires an explicit heat-flux law before a manager launch.
/// @tparam C Direct condition type or Profiled wrapper.
template<class C> inline constexpr bool requires_flux_law=false;
template<class R> inline constexpr bool requires_flux_law<HeatFluxWall<R>> =true;
template<class C,class P> inline constexpr bool requires_flux_law<Profiled<C,P>> =requires_flux_law<C>;

namespace detail {
// Each provider can be a plain constant, a function of the current face and
// time, or a per-face array. Adapt the sampled value to the numeric condition.
template<class R, class P, class C>
KOKKOS_INLINE_FUNCTION R scalar_parameter(const P& p, const C& c)
{ return static_cast<R>(profile::sample(p, c)); }
template<class R, class P, class C>
KOKKOS_INLINE_FUNCTION Vector<R> vector_parameter(const P& p, const C& c)
{ return math::cast<R>(profile::sample(p, c)); }

/// @brief Samples a wall condition whose only parameter is its velocity.
/// @tparam R Arithmetic type of the wall condition.
/// @tparam Wall Numeric wall type, templated on its arithmetic type.
/// @tparam V Provider of Cartesian wall velocity.
template<class R, template<class> class Wall, class V>
struct MechanicalParameters {
    V velocity;
    template<class C> KOKKOS_INLINE_FUNCTION auto wall_velocity(const C& c) const { return vector_parameter<R>(velocity,c); }
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return Wall<R>{vector_parameter<R>(velocity, c)}; }
    template<class Exec> void validate(PatchRange range) const { profile::validate<Exec>(velocity, range); }
};

/// @brief Samples wall velocity and the thermal datum while exposing a velocity-only path for Euler.
/// @tparam R Arithmetic type of the wall condition.
/// @tparam Wall Numeric wall type.
/// @tparam P Provider of wall temperature or heat-flux density.
/// @tparam V Provider of Cartesian wall velocity.
template<class R, template<class> class Wall, class P, class V>
struct ThermalWallParameters {
    P parameter; V velocity;
    template<class C> KOKKOS_INLINE_FUNCTION auto wall_velocity(const C& c) const { return vector_parameter<R>(velocity,c); }
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return Wall<R>{scalar_parameter<R>(parameter,c),vector_parameter<R>(velocity,c)}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(parameter,range); profile::validate<Exec>(velocity,range); }
};

/// @brief Samples the velocity and optional temperature of an incompressible inlet.
template<class R, class V, class T>
struct VelocityParameters {
    V velocity; T temperature;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return VelocityInlet<R>{vector_parameter<R>(velocity,c),scalar_parameter<R>(temperature,c)}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(velocity,range); profile::validate<Exec>(temperature,range); }
};

/// @brief Samples stagnation conditions and flow direction from the inlet reservoir.
template<class R, class P, class T, class D>
struct TotalParameters {
    P pressure; T temperature; D direction; bool allow_choking;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return TotalConditionsInlet<R>{scalar_parameter<R>(pressure,c),scalar_parameter<R>(temperature,c),
                                     vector_parameter<R>(direction,c),allow_choking}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(pressure,range); profile::validate<Exec>(temperature,range); profile::validate<Exec>(direction,range); }
};

/// @brief Samples static pressure, temperature and the selected description of supersonic velocity.
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

/// @brief Samples the outlet pressure and its optional backflow condition.
template<class R, class P, class B>
struct OutletParameters {
    P pressure; B backflow;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return PressureOutlet<R>{scalar_parameter<R>(pressure,c),sample_condition(backflow,c)}; }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(pressure,range); profile::validate<Exec>(backflow,range); }
};

/// @brief Samples a compressible reservoir and wraps it as total-condition backflow.
template<class R, class Reservoir>
struct ReservoirParameters {
    Reservoir reservoir;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { Backflow<R> result; result.mode=BackflowMode::TotalConditions; result.reservoir=sample_condition(reservoir,c); return result; }
    template<class Exec> void validate(PatchRange range) const { profile::validate<Exec>(reservoir,range); }
};

/// @brief Samples the direction and temperature of incompressible backflow.
template<class R, class D, class T>
struct ReverseParameters {
    D direction; T temperature;
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const {
        Backflow<R> result; result.mode=BackflowMode::DirectionTemperature;
        result.direction=vector_parameter<R>(direction,c); result.temperature=scalar_parameter<R>(temperature,c); return result;
    }
    template<class Exec> void validate(PatchRange range) const
    { profile::validate<Exec>(direction,range); profile::validate<Exec>(temperature,range); }
};
} // namespace detail

/// @brief Builds a slip wall with a constant or profiled wall velocity.
/// @param velocity Supplies the Cartesian wall velocity.
template<class R=ComputeReal, class V>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto slip_wall(V velocity)
{ return profiled<SlipWall<R>>(detail::MechanicalParameters<R,SlipWall,V>{velocity}); }

/// @brief Builds an adiabatic no-slip wall, stationary by default.
/// @param velocity Supplies the Cartesian wall velocity.
template<class R=ComputeReal, class V=Vector<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto adiabatic_wall(V velocity={})
{ return profiled<AdiabaticWall<R>>(detail::MechanicalParameters<R,AdiabaticWall,V>{velocity}); }

/// @brief Builds an isothermal no-slip wall with independently specified profiles.
/// @param temperature Supplies the positive wall temperature.
/// @param velocity Supplies the Cartesian wall velocity.
template<class R=ComputeReal, class T, class V=Vector<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto isothermal_wall(T temperature, V velocity={})
{ return profiled<IsothermalWall<R>>(detail::ThermalWallParameters<R,IsothermalWall,T,V>{temperature,velocity}); }

/// @brief Builds a no-slip wall with a prescribed outward heat flux.
/// @param flux Supplies heat flux per unit area.
/// @param velocity Supplies the Cartesian wall velocity.
template<class R=ComputeReal, class F, class V=Vector<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto heat_flux_wall(F flux, V velocity={})
{ return profiled<HeatFluxWall<R>>(detail::ThermalWallParameters<R,HeatFluxWall,F,V>{flux,velocity}); }

/// @brief Builds an incompressible inlet with specified velocity and optional temperature.
/// @param velocity Supplies the Cartesian inlet velocity.
/// @param temperature Supplies the temperature for a thermal model.
template<class R=ComputeReal, class V, class T=R>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto velocity_inlet(V velocity, T temperature={})
{ return profiled<VelocityInlet<R>>(detail::VelocityParameters<R,V,T>{velocity,temperature}); }

/// @brief Builds a characteristic inlet from individually supplied reservoir parameters.
/// @param pressure Supplies the stagnation pressure.
/// @param temperature Supplies the stagnation temperature.
/// @param direction Supplies the inward direction.
/// @param allow_choking Allows the characteristic policy to choose a sonic boundary state.
template<class R=ComputeReal, class P, class T, class D>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto total_conditions_inlet(P pressure,T temperature,D direction,bool allow_choking=false)
{ return profiled<TotalConditionsInlet<R>>(detail::TotalParameters<R,P,T,D>{pressure,temperature,direction,allow_choking}); }

/// @brief Builds a supersonic inlet with explicit Cartesian velocity.
/// @param pressure Supplies static pressure.
/// @param temperature Supplies static temperature.
/// @param velocity Supplies the full inlet velocity.
template<class R=ComputeReal, class P, class T, class V>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto supersonic_inlet(P pressure,T temperature,V velocity)
{ return profiled<SupersonicInlet<R>>(detail::SupersonicParameters<R,P,T,V,R,Vector<R>>{pressure,temperature,velocity,R{0},{}}); }

/// @brief Builds a supersonic inlet whose speed is set by Mach number and sound speed.
/// @param pressure Supplies static pressure.
/// @param temperature Supplies static temperature.
/// @param mach Supplies the prescribed Mach number.
/// @param direction Supplies the inward direction.
template<class R=ComputeReal, class P, class T, class M, class D>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto supersonic_mach_inlet(P pressure,T temperature,M mach,D direction)
{ return profiled<SupersonicInlet<R>>(detail::SupersonicParameters<R,P,T,Vector<R>,M,D>{pressure,temperature,{},mach,direction}); }

/// @brief Builds an outlet with a prescribed pressure and an optional reverse-flow policy.
/// @param pressure Supplies the target static pressure.
/// @param backflow Numeric or profiled data for flow entering the domain.
template<class R=ComputeReal, class P, class B=Backflow<R>>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pressure_outlet(P pressure,B backflow={})
{ return profiled<PressureOutlet<R>>(detail::OutletParameters<R,P,B>{pressure,backflow}); }

/// @brief Selects compressible total-condition backflow for use with pressure_outlet().
/// @param reservoir Numeric or profiled total-condition inlet.
template<class R=ComputeReal, class Reservoir>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto total_backflow(Reservoir reservoir)
{ return profiled<Backflow<R>>(detail::ReservoirParameters<R,Reservoir>{reservoir}); }

/// @brief Selects incompressible backflow with prescribed direction and optional temperature.
/// @param direction Supplies the inward direction.
/// @param temperature Supplies temperature for a thermal model.
template<class R=ComputeReal, class D, class T=R>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto directional_backflow(D direction,T temperature={})
{ return profiled<Backflow<R>>(detail::ReverseParameters<R,D,T>{direction,temperature}); }

} // namespace krona::boundary::physical
