#include "config.hpp"

#include <krona/toml_config/file.hpp>
#include <krona/toml_config/toml.hpp>

namespace krona::apps::euler_steady {
namespace {

using toml_config::TomlReader;

void check_tolerances(const TomlReader& reader, double absolute, double relative, std::string_view table) {
    if (absolute < 0 || relative < 0 || relative >= 1 || (absolute == 0 && relative == 0))
        reader.error(std::string{table} + " tolerances require absolute >= 0, relative in [0, 1), and a positive tolerance");
}

Configuration decode(const toml::table& root, const TomlReader& reader) {
    const auto& euler = reader.table(root, "euler", true);
    reader.check_keys(root, {"euler"});
    reader.check_keys(euler, {"physics", "initial", "numerics", "pseudo_time", "linear_solver", "output"}, "euler");
    Configuration config;

    const auto& physics = reader.table(euler, "physics");
    reader.check_keys(physics, {"gamma", "gas_constant"}, "euler.physics");
    config.gamma = reader.number(physics, "gamma", config.gamma);
    config.gas_constant = reader.number(physics, "gas_constant", config.gas_constant);
    if (!(config.gamma > 1) || !(config.gas_constant > 0))
        reader.error("euler.physics requires gamma > 1 and gas_constant > 0");

    const auto& initial = reader.table(euler, "initial", true);
    reader.check_keys(initial, {"pressure", "temperature", "velocity"}, "euler.initial");
    config.initial.pressure = reader.number(initial, "pressure");
    config.initial.temperature = reader.number(initial, "temperature");
    config.initial.velocity = reader.vector(initial, "velocity", config.initial.velocity);
    if (!(config.initial.pressure > 0) || !(config.initial.temperature > 0))
        reader.error("euler.initial pressure and temperature must be positive");

    const auto& numerics = reader.table(euler, "numerics");
    reader.check_keys(numerics, {"flux", "reconstruction", "limiter"}, "euler.numerics");
    config.flux = reader.choice<Flux>(numerics, "flux", "hllc",
        {{"hllc", Flux::Hllc}, {"hll", Flux::Hll}, {"rusanov", Flux::Rusanov}});
    config.reconstruction = reader.choice<ReconstructionOrder>(numerics, "reconstruction", "first_order",
        {{"first_order", ReconstructionOrder::First}, {"second_order", ReconstructionOrder::Second}});
    config.limiter = reader.choice<Limiter>(numerics, "limiter", "none",
        {{"none", Limiter::None}, {"barth_jespersen", Limiter::BarthJespersen}});

    const auto& pseudo = reader.table(euler, "pseudo_time");
    reader.check_keys(pseudo, {"cfl", "relaxation", "absolute_tolerance", "relative_tolerance",
                              "maximum_iterations"}, "euler.pseudo_time");
    auto& step = config.pseudo_time;
    step.cfl = reader.number(pseudo, "cfl", step.cfl);
    step.relaxation = reader.number(pseudo, "relaxation", step.relaxation);
    step.absolute_tolerance = reader.number(pseudo, "absolute_tolerance", step.absolute_tolerance);
    step.relative_tolerance = reader.number(pseudo, "relative_tolerance", step.relative_tolerance);
    step.maximum_iterations = reader.positive_integer(pseudo, "maximum_iterations", step.maximum_iterations);
    if (!(step.cfl > 0) || !(step.relaxation > 0 && step.relaxation <= 1))
        reader.error("euler.pseudo_time requires cfl > 0 and relaxation in (0, 1]");
    check_tolerances(reader, step.absolute_tolerance, step.relative_tolerance, "euler.pseudo_time");

    const auto& linear = reader.table(euler, "linear_solver");
    reader.check_keys(linear, {"method", "preconditioner", "absolute_tolerance", "relative_tolerance",
                               "maximum_iterations", "restart_length", "verify_residual"}, "euler.linear_solver");
    auto& solve = config.linear;
    solve.method = reader.nonempty_string(linear, "method", solve.method);
    solve.preconditioner = reader.nonempty_string(linear, "preconditioner", solve.preconditioner);
    solve.absolute_tolerance = reader.number(linear, "absolute_tolerance", solve.absolute_tolerance);
    solve.relative_tolerance = reader.number(linear, "relative_tolerance", solve.relative_tolerance);
    solve.maximum_iterations = reader.positive_integer(linear, "maximum_iterations", solve.maximum_iterations);
    solve.restart_length = reader.positive_integer(linear, "restart_length", solve.restart_length);
    solve.verify_residual = reader.boolean(linear, "verify_residual", solve.verify_residual);
    check_tolerances(reader, solve.absolute_tolerance, solve.relative_tolerance, "euler.linear_solver");

    const auto& output = reader.table(euler, "output");
    reader.check_keys(output, {"directory"}, "euler.output");
    if (output.contains("directory")) config.output_directory = reader.nonempty_string(output, "directory");
    return config;
}

} // namespace

Configuration parse_config(std::string_view text) {
    return mpi::guarded([&] {
        const TomlReader reader{"Steady Euler"};
        return decode(reader.parse_text(text), reader);
    });
}

Configuration read_config(const std::filesystem::path& file, mpi::CommView comm) {
    return parse_config(toml_config::read_broadcast(file, comm));
}

} // namespace krona::apps::euler_steady
