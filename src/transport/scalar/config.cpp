#include <krona/transport/scalar/config.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/toml_config/toml.hpp>

#include <algorithm>
#include <limits>

namespace krona::transport::scalar {
namespace {

void validate_tolerances(const toml_config::TomlReader& reader, double relative, double absolute) {
    if (relative < 0 || relative >= 1 || absolute < 0 || (relative == 0 && absolute == 0))
        reader.error("tolerances require 0 <= relative < 1, absolute >= 0 and at least one nonzero value");
}

void read_numerics(Configuration& c, const toml::table& n, const toml_config::TomlReader& reader) {
    reader.check_keys(n, {"gradient", "reconstruction", "diffusion"}, "numerics");

    const auto& g = reader.table(n, "gradient");
    reader.check_keys(g, {"method", "interpolation", "pivot_tolerance"}, "gradient");
    using GS = numerics::gradient::Scheme;
    using GI = numerics::gradient::Interpolation;
    const auto method = reader.nonempty_string(g, "method", "least_squares_face");
    c.gradient_enabled = method != "none";
    if (c.gradient_enabled)
        c.gradient.scheme = reader.choice<GS>(g, "method", "least_squares_face", {
            {"green_gauss_cell", GS::GreenGaussCell}, {"green_gauss_face", GS::GreenGaussFace},
            {"least_squares_face", GS::LeastSquaresFace}});
    c.gradient.interpolation = reader.choice<GI>(g, "interpolation", "arithmetic", {
        {"arithmetic", GI::Arithmetic}, {"auxiliary_weights", GI::AuxiliaryWeights}});
    c.gradient.pivot_tolerance = reader.number(g, "pivot_tolerance", c.gradient.pivot_tolerance);
    if (!(c.gradient.pivot_tolerance > 0 && c.gradient.pivot_tolerance < 1))
        reader.error("gradient.pivot_tolerance must be in (0, 1)");

    const auto& r = reader.table(n, "reconstruction");
    reader.check_keys(r, {"method", "tvd_limiter", "kappa", "skew_correction", "cell_limiter", "regularization"},
                      "reconstruction");
    using RM = ReconstructionMethod;
    using RL = RatioLimiter;
    using CL = CellLimiter;
    auto& options = c.reconstruction;
    options.method = reader.choice<RM>(r, "method", "upwind", {
        {"upwind", RM::Upwind}, {"linear", RM::Linear}, {"muscl_kappa", RM::MusclKappa}, {"tvd", RM::Tvd}});
    options.tvd_limiter = reader.choice<RL>(r, "tvd_limiter", "van_leer", {
        {"minmod", RL::Minmod}, {"superbee", RL::Superbee}, {"monotonized_central", RL::MonotonizedCentral},
        {"van_leer", RL::VanLeer}, {"van_albada", RL::VanAlbada}, {"koren", RL::Koren}});
    options.kappa = reader.number(r, "kappa", options.kappa);
    options.skew_correction = reader.boolean(r, "skew_correction", options.skew_correction);
    options.cell_limiter = reader.choice<CL>(r, "cell_limiter", "none", {
        {"none", CL::None}, {"barth_jespersen", CL::BarthJespersen}, {"venkatakrishnan", CL::Venkatakrishnan}});
    options.regularization = reader.number(r, "regularization", options.regularization);
    if (options.kappa < -1 || options.kappa > 1) reader.error("reconstruction.kappa must be in [-1, 1]");
    if (options.regularization < 0) reader.error("reconstruction.regularization must be nonnegative");
    if (r.contains("kappa") && options.method != RM::MusclKappa)
        reader.error("reconstruction.kappa requires muscl_kappa");
    if (r.contains("skew_correction") && options.method != RM::MusclKappa && options.method != RM::Tvd)
        reader.error("reconstruction.skew_correction requires muscl_kappa or tvd");
    if (r.contains("tvd_limiter") && options.method != RM::Tvd) 
        reader.error("tvd_limiter requires tvd");
    if (r.contains("regularization") && options.cell_limiter != CL::Venkatakrishnan)
        reader.error("reconstruction.regularization requires venkatakrishnan");
    if (options.cell_limiter != CL::None && options.method == RM::Upwind)
        reader.error("a cell limiter requires a high-order reconstruction");
    if (!c.convection_enabled && options.method != RM::Upwind) 
        reader.error("reconstruction requires enabled convection");

    const auto& d = reader.table(n, "diffusion");
    reader.check_keys(d, {"nonorthogonal_correction", "require_orthogonal", "orthogonality_tolerance"}, "diffusion");
    c.nonorthogonal_correction = reader.boolean(d, "nonorthogonal_correction", c.nonorthogonal_correction);
    c.require_orthogonal = reader.boolean(d, "require_orthogonal", !c.nonorthogonal_correction);
    c.orthogonality_tolerance = reader.number(d, "orthogonality_tolerance", c.orthogonality_tolerance);
    if (c.orthogonality_tolerance < 0 || c.orthogonality_tolerance >= 1)
        reader.error("diffusion.orthogonality_tolerance must be in [0, 1)");
    if (c.nonorthogonal_correction && !c.diffusion_enabled)
        reader.error("diffusion correction requires enabled diffusion");
    if (c.needs_gradients() && !c.gradient_enabled)
        reader.error("reconstruction and diffusion correction require a gradient method");
}

Configuration decode(const toml::table& root, std::string_view field, const toml_config::TomlReader& reader) {
    if (field.empty()) reader.error("field name must not be empty");
    const auto& all = reader.table(root, "scalars", true);
    const auto& t = reader.table(all, field, true);
    reader.check_keys(t, {"physics", "numerics", "time", "linear_solver", "outer_iteration", "source"}, field);

    Configuration c;
    c.field = field;
    // Scale the default stopping tolerances to the least precise participant.
    const double precision = std::max(static_cast<double>(std::numeric_limits<SolverReal>::epsilon()),
                                      static_cast<double>(std::numeric_limits<StorageReal>::epsilon()));
    c.linear.relative_tolerance = std::max(c.linear.relative_tolerance, 20*precision);
    c.outer.relative_tolerance = std::max(c.outer.relative_tolerance, 100*precision);

    const auto& p = reader.table(t, "physics");
    reader.check_keys(p, {"density", "diffusion_coefficient", "convection", "diffusion"}, "physics");
    c.density = reader.number(p, "density", c.density);
    c.diffusion = reader.number(p, "diffusion_coefficient", c.diffusion);
    c.convection_enabled = reader.boolean(p, "convection", c.convection_enabled);
    c.diffusion_enabled = reader.boolean(p, "diffusion", c.diffusion_enabled);
    if (!(c.density > 0) || c.diffusion < 0 || (c.diffusion_enabled && !(c.diffusion > 0)))
        reader.error("density and enabled diffusion coefficient must be positive");

    read_numerics(c, reader.table(t, "numerics"), reader);

    const auto& ti = reader.table(t, "time");
    reader.check_keys(ti, {"scheme", "dt", "steps"}, "time");
    c.time_scheme = reader.choice<TimeScheme>(ti, "scheme", "backward_euler", {
        {"steady", TimeScheme::Steady}, {"backward_euler", TimeScheme::BackwardEuler}, {"bdf2", TimeScheme::Bdf2}});
    c.time_step = reader.number(ti, "dt", c.time_step);
    c.steps = reader.positive_integer(ti, "steps", c.steps);
    if (!(c.time_step > 0)) reader.error("time.dt must be positive");
    if (c.time_scheme == TimeScheme::Steady) {
        if (ti.contains("dt") || ti.contains("steps")) reader.error("steady time scheme takes neither dt nor steps");
        if (!c.convection_enabled && !c.diffusion_enabled) reader.error("steady transport requires convection or diffusion");
    }

    const auto& o = reader.table(t, "outer_iteration");
    reader.check_keys(o, {"maximum_iterations", "relative_tolerance", "absolute_tolerance", "relaxation"},
                      "outer_iteration");
    c.outer.maximum_iterations = reader.positive_integer(o, "maximum_iterations", c.outer.maximum_iterations);
    c.outer.relative_tolerance = reader.number(o, "relative_tolerance", c.outer.relative_tolerance);
    c.outer.absolute_tolerance = reader.number(o, "absolute_tolerance", c.outer.absolute_tolerance);
    c.outer.relaxation = reader.number(o, "relaxation", c.outer.relaxation);
    validate_tolerances(reader, c.outer.relative_tolerance, c.outer.absolute_tolerance);
    if (!(c.outer.relaxation > 0 && c.outer.relaxation <= 1)) reader.error("relaxation must be in (0, 1]");

    const auto& l = reader.table(t, "linear_solver");
    reader.check_keys(l, {"method", "preconditioner", "relative_tolerance", "absolute_tolerance",
                          "maximum_iterations", "restart_length", "verify_residual"}, "linear_solver");
    c.linear.method = reader.nonempty_string(l, "method", c.linear.method);
    c.linear.preconditioner = reader.nonempty_string(l, "preconditioner", c.linear.preconditioner);
    c.linear.relative_tolerance = reader.number(l, "relative_tolerance", c.linear.relative_tolerance);
    c.linear.absolute_tolerance = reader.number(l, "absolute_tolerance", c.linear.absolute_tolerance);
    c.linear.maximum_iterations = reader.positive_integer(l, "maximum_iterations", c.linear.maximum_iterations);
    c.linear.restart_length = reader.positive_integer(l, "restart_length", c.linear.restart_length);
    c.linear.verify_residual = reader.boolean(l, "verify_residual", c.linear.verify_residual);
    validate_tolerances(reader, c.linear.relative_tolerance, c.linear.absolute_tolerance);

    const auto& s = reader.table(t, "source");
    reader.check_keys(s, {"type", "value"}, "source");
    if (reader.nonempty_string(s, "type", "constant") != "constant") reader.error("source.type must be constant");
    c.source = reader.number(s, "value", c.source);
    return c;
}

} // namespace

Configuration parse_config(std::string_view text, std::string_view field) {
    return mpi::guarded([&] {
        const toml_config::TomlReader reader{"Scalar transport configuration"};
        return decode(reader.parse_text(text), field, reader);
    });
}

Configuration read_config(const std::filesystem::path& file, std::string_view field, mpi::CommView comm) {
    return parse_config(toml_config::read_broadcast(file, comm), field);
}

} // namespace krona::transport::scalar