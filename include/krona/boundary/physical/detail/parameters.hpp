#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/boundary/physical/status.hpp>

namespace krona::boundary::physical::detail {

template<class Errors>
KOKKOS_INLINE_FUNCTION bool require_device(bool ok, const Errors& errors, Error code, const char* message)
{ if (!ok) errors.fail(code,message); return ok; }
template<class R, class Errors>
KOKKOS_INLINE_FUNCTION bool finite_parameter(R value, const Errors& errors)
{ return require_device(Kokkos::isfinite(value),errors,Error::Parameters,"Physical boundary: parameter must be finite"); }
template<class R, class Errors>
KOKKOS_INLINE_FUNCTION bool positive_parameter(R value, const Errors& errors)
{ return require_device(Kokkos::isfinite(value)&&value>R{0},errors,Error::Parameters,"Physical boundary: require a finite positive parameter"); }
template<class R, class Errors>
KOKKOS_INLINE_FUNCTION bool finite_parameter(Vector<R> value, const Errors& errors)
{ return require_device(math::isfinite(value),errors,Error::Parameters,"Physical boundary: vector must be finite"); }
template<class R, class Errors>
KOKKOS_INLINE_FUNCTION bool direction_parameter(Vector<R>& value, const Errors& errors, bool normalize) {
    if (!finite_parameter(value,errors)) return false;
    const R length2=value.norm2();
    if (!require_device(Kokkos::isfinite(length2)&&length2>R{0},errors,Error::Parameters,
                        "Physical boundary: direction must be nonzero")) return false;
    if (normalize) value=value*(R{1}/Kokkos::sqrt(length2));
    return true;
}

template<class C, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(C&, const Model&, const Errors&, bool) { return true; }
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(SlipWall<R>& c,const Model&,const Errors& e,bool)
{ return finite_parameter(c.velocity,e); }
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(AdiabaticWall<R>& c,const Model&,const Errors& e,bool)
{ return finite_parameter(c.velocity,e); }
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(IsothermalWall<R>& c,const Model&,const Errors& e,bool) {
    return require_device(Model::thermal,e,Error::Unsupported,"Physical boundary: isothermal wall requires a temperature field")&&
           finite_parameter(c.velocity,e)&&positive_parameter(c.temperature,e);
}
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(HeatFluxWall<R>& c,const Model&,const Errors& e,bool) {
    return require_device(Model::thermal,e,Error::Unsupported,"Physical boundary: heat-flux wall requires a temperature field")&&
           finite_parameter(c.velocity,e)&&finite_parameter(c.flux,e);
}
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(TotalConditionsInlet<R>& c,const Model&,const Errors& e,bool normalize) {
    return require_device(Model::compressible&&Model::has_characteristics,e,Error::Unsupported,
        "Physical boundary: total inlet requires a compatible compressible characteristic model")&&
        positive_parameter(c.total_pressure,e)&&positive_parameter(c.total_temperature,e)&&direction_parameter(c.direction,e,normalize);
}
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(VelocityInlet<R>& c,const Model&,const Errors& e,bool) {
    if (!require_device(!Model::compressible,e,Error::Unsupported,
                        "Physical boundary: VelocityInlet currently targets incompressible flow")||!finite_parameter(c.velocity,e)) return false;
    if constexpr (Model::thermal) return positive_parameter(c.temperature,e);
    return true;
}
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(SupersonicInlet<R>& c,const Model&,const Errors& e,bool normalize) {
    if (!require_device(Model::compressible,e,Error::Unsupported,"Physical boundary: supersonic inlet requires compressible flow")||
        !positive_parameter(c.pressure,e)||!positive_parameter(c.temperature,e)||!finite_parameter(c.velocity,e)||
        !require_device(Kokkos::isfinite(c.mach)&&c.mach>=R{0},e,Error::Parameters,"Physical boundary: invalid prescribed Mach number")) return false;
    return c.mach==R{0}||direction_parameter(c.direction,e,normalize);
}
template<class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(SupersonicOutlet&,const Model&,const Errors& e,bool)
{ return require_device(Model::compressible,e,Error::Unsupported,"Physical boundary: supersonic outlet requires compressible flow"); }
template<class R, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool prepare_device(PressureOutlet<R>& c,const Model& model,const Errors& e,bool normalize) {
    if (!finite_parameter(c.pressure,e)) return false;
    if (!require_device(c.backflow.mode==BackflowMode::Reject||c.backflow.mode==BackflowMode::TotalConditions||
                        c.backflow.mode==BackflowMode::DirectionTemperature,e,Error::Parameters,"Physical boundary: invalid backflow mode")) return false;
    if constexpr (Model::compressible) {
        if (!positive_parameter(c.pressure,e)||!require_device(Model::has_characteristics,e,Error::Unsupported,
                "Physical boundary: pressure outlet requires a characteristic policy")||
            !require_device(c.backflow.mode!=BackflowMode::DirectionTemperature,e,Error::Unsupported,
                "Physical boundary: compressible backflow requires total reservoir conditions")) return false;
    } else if (!require_device(c.backflow.mode!=BackflowMode::TotalConditions,e,Error::Unsupported,
                "Physical boundary: incompressible backflow uses DIRECTION_TEMPERATURE")) return false;
    if (c.backflow.mode==BackflowMode::TotalConditions) return prepare_device(c.backflow.reservoir,model,e,normalize);
    if (c.backflow.mode==BackflowMode::DirectionTemperature) {
        if (!direction_parameter(c.backflow.direction,e,normalize)) return false;
        if constexpr (Model::thermal) return positive_parameter(c.backflow.temperature,e);
    }
    return true;
}

template<class R, bool Thermal, class Model, class Errors>
KOKKOS_INLINE_FUNCTION bool valid_state(const Primitive<R,Thermal>& s,const Model&,const Errors& e,Error code) {
    bool ok=Kokkos::isfinite(s.pressure)&&math::isfinite(s.velocity);
    if constexpr (Model::compressible) ok=ok&&s.pressure>R{0};
    if constexpr (Thermal) ok=ok&&Kokkos::isfinite(s.temperature)&&s.temperature>R{0};
    return require_device(ok,e,code,"Physical boundary: invalid primitive state (finite values, positive thermal T and compressible p required)");
}

} // namespace krona::boundary::physical::detail
