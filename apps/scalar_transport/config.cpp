#include "config.hpp"
#include <krona/mpi/error.hpp>
#if __has_include(<toml++/toml.hpp>)
#include <toml++/toml.hpp>
#elif __has_include(<toml++/toml.h>)
#include <toml++/toml.h>
#elif __has_include(<toml.hpp>)
#include <toml.hpp>
#else
#error "Scalar transport configuration requires toml++"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace krona::apps::scalar_transport {
namespace {
[[noreturn]] void error(const std::string& message)
{ mpi::fatal("Scalar transport configuration: " + message); }
void keys(const toml::table& t, std::initializer_list<std::string_view> allowed, std::string_view where)
{
    for (const auto& [key, node] : t) {
        (void)node;
        if (std::find(allowed.begin(), allowed.end(), key.str()) == allowed.end())
            error(std::string{where} + ": unexpected key '" + std::string{key.str()} + "'");
    }
}
const toml::table& table(const toml::table& t, std::string_view key, bool required = false)
{
    static const toml::table empty;
    const auto* node = t.get(key);
    if (!node) {
        if (required) error("missing table '" + std::string{key} + "'");
        return empty;
    }
    if (!node->is_table()) error("expected table '" + std::string{key} + "'");
    return *node->as_table();
}
std::string string(const toml::table& t, std::string_view key, std::string fallback = {})
{
    const auto* node = t.get(key);
    if (!node) return fallback;
    const auto value = node->value<std::string>();
    if (!value || value->empty() || value->find('\0') != std::string::npos)
        error("expected nonempty string '" + std::string{key} + "'");
    return *value;
}
double number(const toml::table& t, std::string_view key, double fallback)
{
    const auto* node = t.get(key);
    if (!node) return fallback;
    const auto value = node->value<double>();
    if (!node->is_number() || !value || !std::isfinite(*value))
        error("expected finite number '" + std::string{key} + "'");
    return *value;
}
bool boolean(const toml::table& t, std::string_view key, bool fallback)
{
    const auto* node = t.get(key);
    if (!node) return fallback;
    const auto value = node->value<bool>();
    if (!value) error("expected boolean '" + std::string{key} + "'");
    return *value;
}
int integer(const toml::table& t, std::string_view key, int fallback)
{
    const auto* node = t.get(key);
    if (!node) return fallback;
    const auto value = node->value<std::int64_t>();
    if (!node->is_integer() || !value || *value < 1 || *value > std::numeric_limits<int>::max())
        error("expected positive int '" + std::string{key} + "'");
    return static_cast<int>(*value);
}
std::filesystem::path resolve(const std::filesystem::path& base,const std::string& path)
{ return (base/std::filesystem::path{path}).lexically_normal(); }
Configuration decode(const toml::table& root,const std::filesystem::path& file,std::string_view text)
{
    keys(root,{"version","application","mesh","boundary","time","output","scalars"},"root");
    if (integer(root,"version",1) != 1) error("unsupported version");
    const auto& app=table(root,"application"); keys(app,{"field"},"application");
    const auto field=string(app,"field","scalar");
    Configuration c;
    static_cast<transport::scalar::Configuration&>(c)=transport::scalar::parse_config(text,field);
    const auto base=file.parent_path();
    const auto& mesh=table(root,"mesh",true); keys(mesh,{"file"},"mesh");
    const auto mesh_file=string(mesh,"file"); if(mesh_file.empty())error("mesh.file is required");
    c.mesh_file=resolve(base,mesh_file);
    const auto& bc=table(root,"boundary",true); keys(bc,{"file"},"boundary");
    const auto bc_file=string(bc,"file"); if(bc_file.empty())error("boundary.file is required");
    c.boundary_file=resolve(base,bc_file);
    const auto& t=table(root,"time"); keys(t,{"mode","dt","end_time","maximum_steps"},"time");
    const auto mode=string(t,"mode","steady");
    if(mode=="steady")c.mode=Mode::Steady;
    else if(mode=="transient")c.mode=Mode::Transient;
    else error("time.mode must be steady or transient");
    c.dt=number(t,"dt",c.dt);c.end_time=number(t,"end_time",c.end_time);
    c.maximum_steps=integer(t,"maximum_steps",c.maximum_steps);
    if(!(c.dt>0) || !(c.end_time>0))error("positive dt/end_time required");
    const double steps=c.end_time/c.dt;
    if(c.mode==Mode::Transient && (!std::isfinite(steps) || steps>static_cast<double>(c.maximum_steps)*(1+16*std::numeric_limits<double>::epsilon())))
        error("maximum_steps is too small for end_time/dt");
    bool moving=false;for(double v:c.velocity)moving=moving||v!=0;
    if(c.mode==Mode::Steady && !c.diffusion_enabled && !(c.convection&&moving))error("steady mode needs an active spatial operator");
    const auto& out=table(root,"output");keys(out,{"enabled","directory","interval"},"output");
    c.output_enabled=boolean(out,"enabled",true);c.output_interval=integer(out,"interval",1);
    c.output_directory=resolve(base,string(out,"directory","output"));
    if(c.verification && c.mode==Mode::Steady) {
        bool stationary=false;
        const auto& reference=c.reference?*c.reference:c.initial;
        if(std::holds_alternative<ConstantInitial>(reference))stationary=c.source==0;
        if(const auto* l=std::get_if<LinearInitial>(&reference)) {
            double transport=0;for(std::size_t d=0;d<3;++d)transport+=(c.convection?c.density*c.velocity[d]:0)*l->gradient[d];
            stationary=std::abs(c.source-transport)<=64*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(c.source),std::abs(transport)});
        }
        if(!stationary)error("steady verification needs a stationary constant/linear reference");
    }
    return c;
}
} // namespace
Configuration parse_config(std::string_view text,const std::filesystem::path& case_file)
{
    return mpi::guarded([&] {
#if TOML_EXCEPTIONS
        return decode(toml::parse(text),case_file,text);
#else
        auto result=toml::parse(text);
        if(!result){std::ostringstream message;message<<result.error();error(message.str());}
        return decode(result.table(),case_file,text);
#endif
    });
}
} // namespace krona::apps::scalar_transport
