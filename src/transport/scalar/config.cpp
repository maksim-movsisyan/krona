#include <krona/transport/scalar/config.hpp>
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

namespace krona::transport::scalar {
namespace {
[[noreturn]] void error(const std::string& message)
{ mpi::fatal("Scalar configuration: " + message); }
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
std::array<double, 3> vector(const toml::table& t, std::string_view key, std::array<double, 3> fallback)
{
    const auto* node = t.get(key);
    if (!node) return fallback;
    const auto* a = node->as_array();
    if (!a || a->size() != 3) error("expected three numbers in '" + std::string{key} + "'");
    for (std::size_t i = 0; i < 3; ++i) {
        const auto value = (*a)[i].value<double>();
        if (!(*a)[i].is_number() || !value || !std::isfinite(*value))
            error("expected finite vector '" + std::string{key} + "'");
        fallback[i] = *value;
    }
    return fallback;
}
template<class E>
E choice(const toml::table& t, std::string_view key, std::string fallback,
         std::initializer_list<std::pair<std::string_view, E>> choices)
{
    const auto value = string(t, key, std::move(fallback));
    for (const auto& [name, code] : choices) if (value == name) return code;
    error("unsupported " + std::string{key} + " '" + value + "'");
}
void tolerances(double relative, double absolute)
{
    if (relative < 0 || relative >= 1 || absolute < 0 || (relative == 0 && absolute == 0))
        error("tolerances require 0 <= relative < 1, absolute >= 0, at least one nonzero");
}
Configuration decode(const toml::table& root, std::string_view field)
{
    if (field.empty()) error("field name must not be empty");
    const auto& all = table(root, "scalars", true);
    const auto& t = table(all, field, true);
    keys(t, {"physics", "numerics", "time", "linear_solver", "outer_iteration", "initial_condition", "reference", "source", "verification"}, field);
    Configuration c; c.field = field;
    // Defaults reflect both algebra and field storage precision.
    const double precision = std::max(static_cast<double>(std::numeric_limits<SolverReal>::epsilon()),
                                     static_cast<double>(std::numeric_limits<StorageReal>::epsilon()));
    c.linear.relative_tolerance = std::max(1.e-8, 20*precision);
    c.outer.relative_tolerance = std::max(1.e-6, 100*precision);
    const auto& p = table(t, "physics");
    keys(p, {"density", "velocity", "diffusion_coefficient", "convection", "diffusion"}, "physics");
    c.density = number(p, "density", c.density); 
    c.velocity = vector(p, "velocity", c.velocity);
    c.diffusion = number(p, "diffusion_coefficient", c.diffusion);
    c.convection = boolean(p, "convection", c.convection); 
    c.diffusion_enabled = boolean(p, "diffusion", c.diffusion_enabled);
    if (!(c.density > 0) || c.diffusion < 0 || (c.diffusion_enabled && !(c.diffusion > 0)))
        error("positive density and positive enabled diffusion coefficient are required");
    const auto& n = table(t, "numerics"); keys(n, {"gradient", "reconstruction", "diffusion"}, "numerics");
    const auto& g = table(n, "gradient"); keys(g, {"method", "interpolation", "pivot_tolerance"}, "gradient");
    const auto gm = string(g, "method", "least_squares_face");
    c.gradient_enabled = gm != "none";
    using GS = numerics::gradient::Scheme;
    if (gm == "green_gauss_cell") c.gradient.scheme = GS::GreenGaussCell;
    else if (gm == "green_gauss_face") c.gradient.scheme = GS::GreenGaussFace;
    else if (gm == "least_squares_face" || gm == "none") c.gradient.scheme = GS::LeastSquaresFace;
    else error("unsupported gradient method '" + gm + "'");
    using GI = numerics::gradient::Interpolation;
    c.gradient.interpolation = choice<GI>(g, "interpolation", "arithmetic", {{"arithmetic",GI::Arithmetic},{"auxiliary_weights",GI::AuxiliaryWeights}});
    c.gradient.pivot_tolerance = number(g, "pivot_tolerance", c.gradient.pivot_tolerance);
    if (!(c.gradient.pivot_tolerance > 0 && c.gradient.pivot_tolerance < 1)) error("invalid gradient pivot_tolerance");
    const auto& r = table(n, "reconstruction");
    keys(r, {"method", "tvd_limiter", "kappa", "skew_correction", "cell_limiter", "regularization"}, "reconstruction");
    using RM = ReconstructionMethod;
    c.reconstruction.method = choice<RM>(r,"method","upwind",{{"upwind",RM::Upwind},{"linear",RM::Linear},{"muscl_kappa",RM::MusclKappa},{"tvd",RM::Tvd}});
    using RL = RatioLimiter;
    c.reconstruction.tvd_limiter = choice<RL>(r,"tvd_limiter","van_leer",{{"minmod",RL::Minmod},{"superbee",RL::Superbee},{"monotonized_central",RL::MonotonizedCentral},{"van_leer",RL::VanLeer},{"van_albada",RL::VanAlbada},{"koren",RL::Koren}});
    c.reconstruction.kappa = number(r,"kappa",c.reconstruction.kappa);
    c.reconstruction.skew_correction = boolean(r,"skew_correction",true);
    using CL = CellLimiter;
    c.reconstruction.cell_limiter = choice<CL>(r,"cell_limiter","none",{{"none",CL::None},{"barth_jespersen",CL::BarthJespersen},{"venkatakrishnan",CL::Venkatakrishnan}});
    c.reconstruction.regularization = number(r,"regularization",0);
    if (c.reconstruction.kappa < -1 || c.reconstruction.kappa > 1 || c.reconstruction.regularization < 0)
        error("kappa must be in [-1,1]; regularization must be nonnegative");
    if (r.contains("kappa") && c.reconstruction.method != RM::MusclKappa) error("kappa requires muscl_kappa");
    if (r.contains("skew_correction") && c.reconstruction.method != RM::MusclKappa && c.reconstruction.method != RM::Tvd)
        error("skew_correction is selected only for muscl_kappa/tvd; linear uses its full Taylor displacement");
    if (r.contains("tvd_limiter") && c.reconstruction.method != RM::Tvd) error("tvd_limiter requires tvd");
    if (r.contains("regularization") && c.reconstruction.cell_limiter != CL::Venkatakrishnan) error("regularization requires venkatakrishnan");
    if (c.reconstruction.cell_limiter != CL::None && c.reconstruction.method == RM::Upwind) error("upwind does not use a cell limiter");
    if (!c.convection && c.reconstruction.method != RM::Upwind) error("reconstruction requires enabled convection");
    const auto& d = table(n,"diffusion");
    keys(d,{"nonorthogonal_correction","require_orthogonal","orthogonality_tolerance"},"diffusion");
    c.nonorthogonal_correction = boolean(d,"nonorthogonal_correction",false);
    c.require_orthogonal = boolean(d,"require_orthogonal",!c.nonorthogonal_correction);
    c.orthogonality_tolerance = number(d,"orthogonality_tolerance",c.orthogonality_tolerance);
    if (c.orthogonality_tolerance < 0 || c.orthogonality_tolerance >= 1) error("invalid orthogonality_tolerance");
    if (c.nonorthogonal_correction && !c.diffusion_enabled) error("diffusion correction requires enabled diffusion");
    if (c.needs_gradients() && !c.gradient_enabled) error("reconstruction/correction requires a gradient method");
    const auto& ti = table(t,"time"); keys(ti,{"scheme"},"scalar time");
    c.time_scheme = choice<TimeScheme>(ti,"scheme","backward_euler",{{"backward_euler",TimeScheme::BackwardEuler},{"bdf2",TimeScheme::Bdf2}});
    const auto& o = table(t,"outer_iteration");
    keys(o,{"maximum_iterations","relative_tolerance","absolute_tolerance","relaxation"},"outer_iteration");
    c.outer.maximum_iterations = integer(o,"maximum_iterations",c.outer.maximum_iterations);
    c.outer.relative_tolerance = number(o,"relative_tolerance",c.outer.relative_tolerance);
    c.outer.absolute_tolerance = number(o,"absolute_tolerance",0);
    c.outer.relaxation = number(o,"relaxation",1);
    tolerances(c.outer.relative_tolerance,c.outer.absolute_tolerance);
    if (!(c.outer.relaxation > 0 && c.outer.relaxation <= 1)) error("relaxation must be in (0,1]");
    const auto& l = table(t,"linear_solver");
    keys(l,{"method","preconditioner","relative_tolerance","absolute_tolerance","maximum_iterations","restart_length","verify_residual"},"linear_solver");
    c.linear.method=string(l,"method",c.linear.method); c.linear.preconditioner=string(l,"preconditioner",c.linear.preconditioner);
    c.linear.relative_tolerance=number(l,"relative_tolerance",c.linear.relative_tolerance);
    c.linear.absolute_tolerance=number(l,"absolute_tolerance",0);
    c.linear.maximum_iterations=integer(l,"maximum_iterations",c.linear.maximum_iterations);
    c.linear.restart_length=integer(l,"restart_length",c.linear.restart_length);
    c.linear.verify_residual=boolean(l,"verify_residual",false);
    tolerances(c.linear.relative_tolerance,c.linear.absolute_tolerance);
    const auto& s = table(t,"source"); keys(s,{"type","value"},"source");
    if (string(s,"type","constant") != "constant") error("only constant sources are built in");
    c.source=number(s,"value",0);
    const auto read_initial=[&](const toml::table& ic) -> InitialCondition {
        const auto type = string(ic,"type","constant");
        if (type == "constant") {
            keys(ic,{"type","value"},"initial_condition"); return ConstantInitial{number(ic,"value",0)};
        } else if (type == "linear") {
            keys(ic,{"type","value","gradient","origin"},"initial_condition");
            return LinearInitial{number(ic,"value",0),vector(ic,"gradient",{}),vector(ic,"origin",{})};
        } else if (type == "gaussian") {
            keys(ic,{"type","background","amplitude","centre","width"},"initial_condition");
            GaussianInitial a;
            a.background=number(ic,"background",0); a.amplitude=number(ic,"amplitude",1);
            a.centre=vector(ic,"centre",a.centre); a.width=number(ic,"width",a.width);
            if (!(a.width > 0) || !std::isfinite(a.width*a.width) || !(a.width*a.width > 0)) error("invalid Gaussian width");
            return a;
        } else error("built-in initial types are constant, linear, gaussian");
    };
    c.initial=read_initial(table(t,"initial_condition"));
    if(t.contains("reference"))c.reference=read_initial(table(t,"reference"));
    const auto& v = table(t,"verification"); keys(v,{"enabled","maximum_l2_error"},"verification");
    c.verification=boolean(v,"enabled",false);
    if (v.contains("maximum_l2_error")) {
        c.maximum_l2_error=number(v,"maximum_l2_error",0);
        if (!c.verification || *c.maximum_l2_error < 0) error("maximum_l2_error needs verification and a nonnegative bound");
    }
    return c;
}
} // namespace
Configuration parse_config(std::string_view text, std::string_view field)
{
    return mpi::guarded([&] {
#if TOML_EXCEPTIONS
        return decode(toml::parse(text),field);
#else
        auto result=toml::parse(text);
        if (!result) { std::ostringstream message; message << result.error(); error(message.str()); }
        return decode(result.table(),field);
#endif
    });
}
Configuration read_config(const std::filesystem::path& file, std::string_view field)
{
    return mpi::guarded([&] {
#if TOML_EXCEPTIONS
        return decode(toml::parse_file(file.string()), field);
#else
        auto parsed = toml::parse_file(file.string());
        if (!parsed) parse_error(parsed.error());
        return decode(parsed.table(), field);
#endif
    });
}
} // namespace krona::transport::scalar
