#include <krona/transport/compressible/config.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/toml_config/toml.hpp>

namespace krona::transport::compressible {
namespace {

using toml_config::TomlReader;

void check_tolerances(const TomlReader& reader, double absolute, double relative, std::string_view table) {
    if (absolute < 0 || relative < 0 || relative >= 1 || (absolute == 0 && relative == 0))
        reader.error(std::string{table} + " tolerances require absolute >= 0, relative in [0, 1), and a positive tolerance");
}

MolecularOptions decode_molecular(const toml::table& table, const TomlReader& reader) {
    reader.check_keys(table, {"enabled", "viscosity", "conductivity", "dynamic_viscosity", "bulk_viscosity",
        "thermal_conductivity", "prandtl", "reference_viscosity", "reference_temperature",
        "sutherland_temperature"}, "flow.transport");
    MolecularOptions result;
    result.viscosity = reader.choice<ViscosityModel>(table, "viscosity", "constant",
        {{"constant", ViscosityModel::Constant}, {"sutherland", ViscosityModel::Sutherland}});
    result.conductivity = reader.choice<ConductivityModel>(table, "conductivity", "constant_prandtl",
        {{"constant", ConductivityModel::Constant}, {"constant_prandtl", ConductivityModel::ConstantPrandtl}});
    result.dynamic_viscosity = reader.number(table, "dynamic_viscosity", result.dynamic_viscosity);
    result.bulk_viscosity = reader.number(table, "bulk_viscosity", result.bulk_viscosity);
    result.thermal_conductivity = reader.number(table, "thermal_conductivity", result.thermal_conductivity);
    result.prandtl = reader.number(table, "prandtl", result.prandtl);
    result.reference_viscosity = reader.number(table, "reference_viscosity", result.reference_viscosity);
    result.reference_temperature = reader.number(table, "reference_temperature", result.reference_temperature);
    result.sutherland_temperature = reader.number(table, "sutherland_temperature", result.sutherland_temperature);
    if (result.dynamic_viscosity < 0 || result.bulk_viscosity < 0 || result.thermal_conductivity < 0 ||
        result.prandtl <= 0 || result.reference_viscosity <= 0 || result.reference_temperature <= 0 ||
        result.sutherland_temperature < 0)
        reader.error("flow.transport has an invalid molecular coefficient or reference state");
    return result;
}

/// @brief Reads the local flux and reconstruction options.
void read_numerics(Configuration& config, const toml::table& table, const TomlReader& reader) {
    reader.check_keys(table, {"flux", "reconstruction", "limiter", "kappa", "tvd_limiter", "skew_correction"},
                      "flow.numerics");
    config.flux = reader.choice<Flux>(table, "flux", "hllc",
        {{"hllc", Flux::Hllc}, {"hll", Flux::Hll}, {"rusanov", Flux::Rusanov}});

    using Method = ReconstructionMethod;
    auto& options = config.reconstruction;
    options.method = reader.choice<Method>(table, "reconstruction", "first_order",
        {{"first_order", Method::FirstOrder}, {"linear", Method::Linear},
         {"muscl_kappa", Method::MusclKappa}, {"tvd", Method::Tvd}});
    options.cell_limiter = reader.choice<CellLimiter>(table, "limiter", "none",
        {{"none", CellLimiter::None}, {"barth_jespersen", CellLimiter::BarthJespersen}});
    options.tvd_limiter = reader.choice<RatioLimiter>(table, "tvd_limiter", "van_leer",
        {{"minmod", RatioLimiter::Minmod}, {"superbee", RatioLimiter::Superbee},
         {"monotonized_central", RatioLimiter::MonotonizedCentral}, {"van_leer", RatioLimiter::VanLeer},
         {"van_albada", RatioLimiter::VanAlbada}, {"koren", RatioLimiter::Koren}});
    options.kappa = reader.number(table, "kappa", options.kappa);
    options.skew_correction = reader.boolean(table, "skew_correction", options.skew_correction);

    if (options.kappa < -1 || options.kappa > 1)
        reader.error("flow.numerics.kappa must be in [-1, 1]");
    if (table.contains("kappa") && options.method != Method::MusclKappa)
        reader.error("flow.numerics.kappa requires muscl_kappa");
    if (table.contains("tvd_limiter") && options.method != Method::Tvd)
        reader.error("flow.numerics.tvd_limiter requires tvd");
    if (table.contains("skew_correction") && options.method != Method::MusclKappa && options.method != Method::Tvd)
        reader.error("flow.numerics.skew_correction requires muscl_kappa or tvd");
    if (options.cell_limiter != CellLimiter::None && !options.needs_gradients())
        reader.error("flow.numerics.limiter requires a high-order reconstruction");
}

/// @brief Decodes the common flow configuration independently of the executable.
Configuration decode(const toml::table& root, const TomlReader& reader) {
    const std::string section = "flow";
    const auto& flow = reader.table(root, section, true);
    reader.check_keys(flow, {"physics", "initial", "numerics", "pseudo_time", "linear_solver",
                                        "output", "transport"}, section);

    Configuration config;
    const auto& transport = reader.table(flow, "transport");
    config.viscous_enabled = reader.boolean(transport, "enabled", config.viscous_enabled);
    config.molecular = decode_molecular(transport, reader);

    const auto& physics = reader.table(flow, "physics");
    reader.check_keys(physics, {"gamma", "gas_constant"}, section + ".physics");
    config.gamma = reader.number(physics, "gamma", config.gamma);
    config.gas_constant = reader.number(physics, "gas_constant", config.gas_constant);
    if (!(config.gamma > 1) || !(config.gas_constant > 0))
        reader.error(section + ".physics requires gamma > 1 and gas_constant > 0");

    const auto& initial = reader.table(flow, "initial", true);
    reader.check_keys(initial, {"pressure", "temperature", "velocity"}, section + ".initial");
    config.initial.pressure = reader.number(initial, "pressure");
    config.initial.temperature = reader.number(initial, "temperature");
    config.initial.velocity = reader.vector(initial, "velocity", config.initial.velocity);
    if (!(config.initial.pressure > 0) || !(config.initial.temperature > 0))
        reader.error(section + ".initial pressure and temperature must be positive");

    read_numerics(config, reader.table(flow, "numerics"), reader);

    const auto& pseudo = reader.table(flow, "pseudo_time");
    reader.check_keys(pseudo, {"cfl", "relaxation", "absolute_tolerance", "relative_tolerance",
                              "maximum_iterations"}, section + ".pseudo_time");
    auto& step = config.pseudo_time;
    step.cfl = reader.number(pseudo, "cfl", step.cfl);
    step.relaxation = reader.number(pseudo, "relaxation", step.relaxation);
    step.absolute_tolerance = reader.number(pseudo, "absolute_tolerance", step.absolute_tolerance);
    step.relative_tolerance = reader.number(pseudo, "relative_tolerance", step.relative_tolerance);
    step.maximum_iterations = reader.positive_integer(pseudo, "maximum_iterations", step.maximum_iterations);
    if (!(step.cfl > 0) || !(step.relaxation > 0 && step.relaxation <= 1))
        reader.error(section + ".pseudo_time requires cfl > 0 and relaxation in (0, 1]");
    check_tolerances(reader, step.absolute_tolerance, step.relative_tolerance, section + ".pseudo_time");

    const auto& linear = reader.table(flow, "linear_solver");
    reader.check_keys(linear, {"method", "preconditioner", "absolute_tolerance", "relative_tolerance",
                               "maximum_iterations", "restart_length", "verify_residual"}, section + ".linear_solver");
    auto& solve = config.linear;
    solve.method = reader.nonempty_string(linear, "method", solve.method);
    solve.preconditioner = reader.nonempty_string(linear, "preconditioner", solve.preconditioner);
    solve.absolute_tolerance = reader.number(linear, "absolute_tolerance", solve.absolute_tolerance);
    solve.relative_tolerance = reader.number(linear, "relative_tolerance", solve.relative_tolerance);
    solve.maximum_iterations = reader.positive_integer(linear, "maximum_iterations", solve.maximum_iterations);
    solve.restart_length = reader.positive_integer(linear, "restart_length", solve.restart_length);
    solve.verify_residual = reader.boolean(linear, "verify_residual", solve.verify_residual);
    check_tolerances(reader, solve.absolute_tolerance, solve.relative_tolerance, section + ".linear_solver");

    const auto& output = reader.table(flow, "output");
    reader.check_keys(output, {"directory"}, section + ".output");
    if (output.contains("directory")) config.output_directory = reader.nonempty_string(output, "directory");
    return config;
}

} // namespace

Configuration parse_config(std::string_view text) {
    return mpi::guarded([&] {
        const TomlReader reader{"flow"};
        return decode(reader.parse_text(text), reader);
    });
}

Configuration read_config(const std::filesystem::path& file, mpi::CommView comm) {
    return parse_config(toml_config::read_broadcast(file, comm));
}

} // namespace krona::transport::compressible
