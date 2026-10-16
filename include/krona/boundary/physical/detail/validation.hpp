#pragma once

#include <krona/boundary/physical/models.hpp>
#include <krona/mpi/error.hpp>
#include <cmath>

namespace krona::boundary::physical::detail {

template<class Real>
void finite_vector(Vector<Real> v)
{ mpi::require(std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z),"Physical boundary: vector must be finite"); }
template<class Real>
Vector<Real> unit_vector(Vector<Real> v)
{
    finite_vector(v);
    const Real norm = std::sqrt(v.norm2());
    mpi::require(std::isfinite(norm)&&norm>Real{0},"Physical boundary: direction must be nonzero");
    return v*(Real{1}/norm);
}
template<class Real>
void positive(Real v)
{ mpi::require(std::isfinite(v)&&v>Real{0},"Physical boundary: require a finite positive parameter"); }

template<class C,class Model>
C prepare(C c,const Model& model)
{
    if constexpr (requires { c.validate(model); }) c.validate(model);
    return c;
}
template<class Real,class Model>
SlipWall<Real> prepare(SlipWall<Real> c,const Model&) { finite_vector(c.velocity); return c; }
template<class Real,class Model>
AdiabaticWall<Real> prepare(AdiabaticWall<Real> c,const Model&) { finite_vector(c.velocity); return c; }
template<class Real,class Model>
IsothermalWall<Real> prepare(IsothermalWall<Real> c,const Model&)
{
    mpi::require(Model::thermal,"Physical boundary: isothermal wall requires a temperature field");
    finite_vector(c.velocity); positive(c.temperature); return c;
}
template<class Real,class Model>
HeatFluxWall<Real> prepare(HeatFluxWall<Real> c,const Model&)
{
    mpi::require(Model::thermal,"Physical boundary: heat-flux wall requires a temperature field");
    finite_vector(c.velocity); mpi::require(std::isfinite(c.flux),"Physical boundary: flux must be finite"); return c;
}
template<class Real,class Model>
TotalConditionsInlet<Real> prepare(TotalConditionsInlet<Real> c,const Model&)
{
    mpi::require(Model::compressible&&Model::has_characteristics,
        "Physical boundary: total inlet requires a compatible compressible characteristic model");
    positive(c.total_pressure); positive(c.total_temperature); c.direction=unit_vector(c.direction); return c;
}
template<class Real,class Model>
VelocityInlet<Real> prepare(VelocityInlet<Real> c,const Model&)
{
    mpi::require(!Model::compressible,"Physical boundary: VelocityInlet currently targets incompressible flow");
    finite_vector(c.velocity); if constexpr (Model::thermal) positive(c.temperature); return c;
}
template<class Real,class Model>
SupersonicInlet<Real> prepare(SupersonicInlet<Real> c,const Model&)
{
    mpi::require(Model::compressible,"Physical boundary: supersonic inlet requires compressible flow");
    positive(c.pressure); positive(c.temperature); finite_vector(c.velocity);
    mpi::require(std::isfinite(c.mach)&&c.mach>=Real{0},"Physical boundary: invalid prescribed Mach number");
    if (c.mach>Real{0}) c.direction=unit_vector(c.direction);
    return c;
}
template<class Model>
SupersonicOutlet prepare(SupersonicOutlet c,const Model&)
{ mpi::require(Model::compressible,"Physical boundary: supersonic outlet requires compressible flow"); return c; }
template<class Real,class Model>
PressureOutlet<Real> prepare(PressureOutlet<Real> c,const Model& model)
{
    mpi::require(std::isfinite(c.pressure),"Physical boundary: pressure must be finite");
    if constexpr (Model::compressible) {
        positive(c.pressure);
        mpi::require(Model::has_characteristics,"Physical boundary: pressure outlet requires a characteristic policy");
        mpi::require(c.backflow.mode!=BackflowMode::DirectionTemperature,
            "Physical boundary: compressible backflow requires total reservoir conditions");
    } else mpi::require(c.backflow.mode!=BackflowMode::TotalConditions,
        "Physical boundary: incompressible backflow uses DIRECTION_TEMPERATURE");
    if (c.backflow.mode==BackflowMode::TotalConditions) c.backflow.reservoir=prepare(c.backflow.reservoir,model);
    if (c.backflow.mode==BackflowMode::DirectionTemperature) {
        c.backflow.direction=unit_vector(c.backflow.direction);
        if constexpr (Model::thermal) positive(c.backflow.temperature);
    }
    return c;
}

template<class Real,class Other>
Vector<Real> vector_cast(Vector<Other> v)
{ return {static_cast<Real>(v.x),static_cast<Real>(v.y),static_cast<Real>(v.z)}; }
template<class Real> Symmetry condition_cast(Symmetry c) { return c; }
template<class Real> SupersonicOutlet condition_cast(SupersonicOutlet c) { return c; }
template<class Real,class O> SlipWall<Real> condition_cast(SlipWall<O> c) { return {vector_cast<Real>(c.velocity)}; }
template<class Real,class O> AdiabaticWall<Real> condition_cast(AdiabaticWall<O> c) { return {vector_cast<Real>(c.velocity)}; }
template<class Real,class O> IsothermalWall<Real> condition_cast(IsothermalWall<O> c)
{ return {static_cast<Real>(c.temperature),vector_cast<Real>(c.velocity)}; }
template<class Real,class O> HeatFluxWall<Real> condition_cast(HeatFluxWall<O> c)
{ return {static_cast<Real>(c.flux),vector_cast<Real>(c.velocity)}; }
template<class Real,class O> TotalConditionsInlet<Real> condition_cast(TotalConditionsInlet<O> c)
{ return {static_cast<Real>(c.total_pressure),static_cast<Real>(c.total_temperature),vector_cast<Real>(c.direction),c.allow_choking}; }
template<class Real,class O> SupersonicInlet<Real> condition_cast(SupersonicInlet<O> c)
{ return {static_cast<Real>(c.pressure),static_cast<Real>(c.temperature),vector_cast<Real>(c.velocity),
    static_cast<Real>(c.mach),vector_cast<Real>(c.direction)}; }
template<class Real,class O> VelocityInlet<Real> condition_cast(VelocityInlet<O> c)
{ return {vector_cast<Real>(c.velocity),static_cast<Real>(c.temperature)}; }
template<class Real,class O> PressureOutlet<Real> condition_cast(PressureOutlet<O> c)
{
    Backflow<Real> back{c.backflow.mode,condition_cast<Real>(c.backflow.reservoir),
        vector_cast<Real>(c.backflow.direction),static_cast<Real>(c.backflow.temperature)};
    return {static_cast<Real>(c.pressure),back};
}

} // namespace krona::boundary::physical::detail
