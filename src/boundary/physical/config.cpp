#include <krona/boundary/physical/config.hpp>
#include <krona/mpi/error.hpp>

#if __has_include(<toml++/toml.hpp>)
#include <toml++/toml.hpp>
#elif __has_include(<toml++/toml.h>)
#include <toml++/toml.h>
#elif __has_include(<toml.hpp>)
#include <toml.hpp>
#else
#error "Physical boundary configuration requires toml++"
#endif

#include <cmath>
#include <initializer_list>
#include <unordered_set>

namespace krona::boundary::physical {
namespace {
[[noreturn]] void error(const std::string& message) { mpi::fatal("Physical boundary configuration: "+message); }

std::string string(const toml::table& t,std::string_view key)
{
    auto s=t[key].value<std::string>();
    if (!s) error("missing string '"+std::string{key}+"'");
    return *s;
}
double number(const toml::table& t,std::string_view key)
{
    const auto node=t[key]; const auto x=node.value<double>();
    if (!node || !node.is_number() || !x || !std::isfinite(*x))
        error("'"+std::string{key}+"' must be a finite number");
    return *x;
}
double number(const toml::table& t,std::string_view canonical,std::string_view alias)
{
    if (t.contains(canonical)&&t.contains(alias)) error("duplicate parameter aliases for '"+std::string{canonical}+"'");
    return number(t,t.contains(canonical)?canonical:alias);
}
Vector<double> vector(const toml::table& t,std::string_view key,bool required=true)
{
    if (!t.contains(key)&&!required) return {};
    const auto* a=t.get_as<toml::array>(key);
    if (!a || a->size()!=3) error("'"+std::string{key}+"' must contain three numbers");
    double values[3]{};
    for (std::size_t i=0;i<3;++i) {
        const auto& node=(*a)[i]; const auto x=node.value<double>();
        if (!node.is_number() || !x || !std::isfinite(*x)) error("nonfinite/non-numeric vector entry");
        values[i]=*x;
    }
    return {values[0],values[1],values[2]};
}
bool boolean(const toml::table& t,std::string_view key,bool fallback=false)
{
    if (!t.contains(key)) return fallback;
    const auto x=t[key].value<bool>();
    if (!x) error("'"+std::string{key}+"' must be boolean");
    return *x;
}
void keys(const toml::table& t,std::initializer_list<std::string_view> allowed,bool patch)
{
    for (const auto& [key,node]:t) {
        (void)node;
        const auto k=key.str();
        if (patch && (k=="patch_id"||k=="name"||k=="family"||k=="source_type"||
                      k=="global_face_count"||k=="scalar")) continue;
        bool found=false;
        for (const auto a:allowed) found=found||a==k;
        if (!found) error("unexpected key '"+std::string{k}+"'");
    }
}
TotalConditionsInlet<double> total_inlet(const toml::table& t)
{ return {number(t,"total_pressure","p0"),number(t,"total_temperature","T0"),vector(t,"direction"),boolean(t,"allow_choking")}; }

Backflow<double> backflow(const toml::table& parent)
{
    Backflow<double> result;
    if (!parent.contains("backflow")) return result;
    const auto* t=parent.get_as<toml::table>("backflow");
    if (!t) error("backflow must be a table");
    const auto type=string(*t,"type");
    if (type=="REJECT") keys(*t,{"type"},false);
    else if (type=="TOTAL_CONDITIONS_INLET") {
        keys(*t,{"type","total_pressure","p0","total_temperature","T0","direction","allow_choking"},false);
        result.mode=BackflowMode::TotalConditions; result.reservoir=total_inlet(*t);
    } else if (type=="DIRECTION_TEMPERATURE") {
        keys(*t,{"type","direction","temperature"},false);
        result.mode=BackflowMode::DirectionTemperature;
        result.direction=vector(*t,"direction");
        if (t->contains("temperature")) result.temperature=number(*t,"temperature");
    } else error("unsupported backflow type '"+type+"'");
    return result;
}

ConfigCondition condition(const toml::table& t)
{
    if (const auto* scalar=t.get_as<toml::table>("scalar"))
        if (scalar->contains("temperature"))
            error("scalar.temperature conflicts with the physical boundary temperature channel");
    const auto type=string(t,"type");
    if (type=="SYMMETRY") { keys(t,{"type"},true); return Symmetry{}; }
    if (type=="SLIP_WALL") {
        keys(t,{"type","velocity"},true); return SlipWall<double>{vector(t,"velocity",false)};
    }
    if (type=="ADIABATIC_WALL"||type=="WALL_ADIABATIC") {
        keys(t,{"type","velocity"},true); return AdiabaticWall<double>{vector(t,"velocity",false)};
    }
    if (type=="ISOTHERMAL_WALL"||type=="WALL_ISOTHERMAL") {
        keys(t,{"type","temperature","T_wall","velocity"},true);
        return IsothermalWall<double>{number(t,"temperature","T_wall"),vector(t,"velocity",false)};
    }
    if (type=="HEAT_FLUX_WALL") {
        keys(t,{"type","flux","velocity"},true);
        return HeatFluxWall<double>{number(t,"flux"),vector(t,"velocity",false)};
    }
    if (type=="PRESSURE_OUTLET"||type=="OUTLET_BACK_PRESSURE") {
        keys(t,{"type","pressure","p_back","backflow"},true);
        return PressureOutlet<double>{number(t,"pressure","p_back"),backflow(t)};
    }
    if (type=="TOTAL_CONDITIONS_INLET"||type=="INLET_TOTAL_TP") {
        keys(t,{"type","total_pressure","p0","total_temperature","T0","direction","allow_choking"},true);
        return total_inlet(t);
    }
    if (type=="SUPERSONIC_INLET"||type=="INLET_SUPERSONIC") {
        keys(t,{"type","pressure","p","temperature","T","velocity","mach","M","direction"},true);
        SupersonicInlet<double> result;
        result.pressure=number(t,"pressure","p"); result.temperature=number(t,"temperature","T");
        if (t.contains("velocity")) {
            if (t.contains("mach")||t.contains("M")||t.contains("direction"))
                error("supersonic inlet accepts velocity OR Mach and direction");
            result.velocity=vector(t,"velocity");
        } else { result.mach=number(t,"mach","M"); result.direction=vector(t,"direction");
            if (!(result.mach>0)) error("prescribed Mach number must be positive"); }
        return result;
    }
    if (type=="SUPERSONIC_OUTLET"||type=="OUTLET_SUPERSONIC") {
        keys(t,{"type"},true); return SupersonicOutlet{};
    }
    if (type=="VELOCITY_INLET") {
        keys(t,{"type","velocity","temperature"},true);
        return VelocityInlet<double>{vector(t,"velocity"),t.contains("temperature")?number(t,"temperature"):0.0};
    }
    error("unsupported physical type '"+type+"'");
}

Configuration decode(const toml::table& root)
{
    Configuration result;
    const auto* entries=root.get_as<toml::array>("boundary_condition");
    if (!entries) error("expected [[boundary_condition]] entries or boundary_condition = []");
    std::unordered_set<std::int64_t> seen;
    result.patches.reserve(entries->size());
    for (const auto& entry:*entries) {
        const auto* t=entry.as_table();
        if (!t) error("boundary_condition entries must be tables");
        const auto id=(*t)["patch_id"].value<std::int64_t>();
        if (!id || *id<0) error("patch_id must be a nonnegative integer");
        if (!seen.insert(*id).second) error("duplicate patch_id "+std::to_string(*id));
        result.patches.push_back({*id,string(*t,"name"),condition(*t)});
    }
    if (root.contains("unlabelled")) {
        const auto* t=root.get_as<toml::table>("unlabelled");
        if (!t) error("unlabelled must be a table");
        // A scalar-only unlabelled table does not silently select a physical condition.
        if (t->contains("type")) result.unlabelled=condition(*t);
    }
    return result;
}

Configuration parsed(std::string_view text)
{
#if TOML_EXCEPTIONS
    return decode(toml::parse(text));
#else
    auto result=toml::parse(text);
    if (!result) { std::ostringstream message; message<<result.error(); error(message.str()); }
    return decode(result.table());
#endif
}
}

Configuration parse_config(std::string_view text) { return mpi::guarded([&] { return parsed(text); }); }
Configuration read_config(const std::filesystem::path& file)
{
    return mpi::guarded([&] {
#if TOML_EXCEPTIONS
        return decode(toml::parse_file(file.string()));
#else
        auto result=toml::parse_file(file.string());
        if (!result) { std::ostringstream message; message<<result.error(); error(message.str()); }
        return decode(result.table());
#endif
    });
}

} // namespace krona::boundary::physical
