#include <krona/boundary/physical/config.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/toml_config/toml.hpp>

#include <cmath>
#include <initializer_list>
#include <unordered_set>
#include <utility>

namespace krona::boundary::physical {
namespace {

using toml_config::TomlReader;
using Parameter = ProfileParameter;

class ConditionReader {
public:
    explicit ConditionReader(const TomlReader& reader) : reader_(reader) {}

    [[nodiscard]] ConfigCondition parse(const toml::table& table) {
        auto numeric = condition(table);
        if (profiles_.empty())
            return std::visit([](auto c) -> ConfigCondition { return c; }, numeric);
        return TabulatedConditionConfig{std::move(numeric), std::move(profiles_)};
    }

private:
    const TomlReader& reader_;
    std::vector<ParameterProfile> profiles_;

    double scalar(const toml::table& table, std::string_view key, Parameter parameter) {
        if (!table[key].is_string()) return reader_.number(table, key);
        profiles_.push_back({parameter, 0, std::filesystem::path{reader_.nonempty_string(table, key)}});
        return 0;
    }

    Vector<double> vector(const toml::table& table, std::string_view key, Parameter parameter, bool required = true) {
        if (!required && !table.contains(key)) return {};
        const auto& entries = reader_.array(table, key);
        if (entries.size() != 3) reader_.error("'" + std::string{key} + "' must contain three components");
        Vector<double> result{};
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& node = entries[i];
            if (const auto* text = node.as_string()) {
                const auto& path = text->get();
                if (path.empty() || path.find('\0') != std::string::npos)
                    reader_.error("'" + std::string{key} + "' profile path must be nonempty without embedded NULs");
                profiles_.push_back({parameter, i, std::filesystem::path{path}});
            } else {
                const auto value = node.value<double>();
                if (!node.is_number() || !value || !std::isfinite(*value))
                    reader_.error("'" + std::string{key} + "' components must be finite numbers or CSV paths");
                result[i] = *value;
            }
        }
        return result;
    }

    void keys(const toml::table& table, std::initializer_list<std::string_view> allowed) const {
        reader_.check_keys(table, allowed, {}, {"patch_id", "name", "family", "source_type", "global_face_count", "scalar"});
    }

    TotalConditionsInlet<double> total_inlet(const toml::table& table) {
        return {scalar(table, reader_.alias(table, "total_pressure", "p0"), Parameter::TotalPressure),
                scalar(table, reader_.alias(table, "total_temperature", "T0"), Parameter::TotalTemperature),
                vector(table, "direction", Parameter::Direction),
                reader_.boolean(table, "allow_choking", false)};
    }

    Backflow<double> backflow(const toml::table& parent) {
        Backflow<double> result;
        if (!parent.contains("backflow")) return result;
        const auto& table = reader_.table(parent, "backflow", true);
        const auto type = reader_.string(table, "type");
        if (type == "REJECT") {
            reader_.check_keys(table, {"type"});
        } else if (type == "TOTAL_CONDITIONS_INLET") {
            reader_.check_keys(table, {"type", "total_pressure", "p0", "total_temperature", "T0", "direction", "allow_choking"});
            result.mode = BackflowMode::TotalConditions;
            result.reservoir = total_inlet(table);
        } else if (type == "DIRECTION_TEMPERATURE") {
            reader_.check_keys(table, {"type", "direction", "temperature"});
            result.mode = BackflowMode::DirectionTemperature;
            result.direction = vector(table, "direction", Parameter::Direction);
            if (table.contains("temperature"))
                result.temperature = scalar(table, "temperature", Parameter::Temperature);
        } else {
            reader_.error("unsupported backflow type '" + type + "'");
        }
        return result;
    }

    detail::ConfigVariant<> condition(const toml::table& table) {
        if (reader_.table(table, "scalar").contains("temperature"))
            reader_.error("scalar.temperature conflicts with the physical boundary temperature channel");
        const auto type = reader_.string(table, "type");
        if (type == "SYMMETRY") {
            keys(table, {"type"});
            return Symmetry{};
        }
        if (type == "SLIP_WALL") {
            keys(table, {"type", "velocity"});
            return SlipWall<double>{vector(table, "velocity", Parameter::Velocity, false)};
        }
        if (type == "ADIABATIC_WALL" || type == "WALL_ADIABATIC") {
            keys(table, {"type", "velocity"});
            return AdiabaticWall<double>{vector(table, "velocity", Parameter::Velocity, false)};
        }
        if (type == "ISOTHERMAL_WALL" || type == "WALL_ISOTHERMAL") {
            keys(table, {"type", "temperature", "T_wall", "velocity"});
            return IsothermalWall<double>{scalar(table, reader_.alias(table, "temperature", "T_wall"), Parameter::Temperature),
                                          vector(table, "velocity", Parameter::Velocity, false)};
        }
        if (type == "HEAT_FLUX_WALL") {
            keys(table, {"type", "flux", "velocity"});
            return HeatFluxWall<double>{scalar(table, "flux", Parameter::Flux),
                                        vector(table, "velocity", Parameter::Velocity, false)};
        }
        if (type == "PRESSURE_OUTLET" || type == "OUTLET_BACK_PRESSURE") {
            keys(table, {"type", "pressure", "p_back", "backflow"});
            return PressureOutlet<double>{scalar(table, reader_.alias(table, "pressure", "p_back"), Parameter::Pressure),
                                          backflow(table)};
        }
        if (type == "TOTAL_CONDITIONS_INLET" || type == "INLET_TOTAL_TP") {
            keys(table, {"type", "total_pressure", "p0", "total_temperature", "T0", "direction", "allow_choking"});
            return total_inlet(table);
        }
        if (type == "SUPERSONIC_INLET" || type == "INLET_SUPERSONIC") {
            keys(table, {"type", "pressure", "p", "temperature", "T", "velocity", "mach", "M", "direction"});
            SupersonicInlet<double> result;
            result.pressure = scalar(table, reader_.alias(table, "pressure", "p"), Parameter::Pressure);
            result.temperature = scalar(table, reader_.alias(table, "temperature", "T"), Parameter::Temperature);
            if (table.contains("velocity")) {
                if (table.contains("mach") || table.contains("M") || table.contains("direction"))
                    reader_.error("supersonic inlet accepts velocity OR Mach and direction");
                result.velocity = vector(table, "velocity", Parameter::Velocity);
            } else {
                const auto mach = reader_.alias(table, "mach", "M");
                result.mach = scalar(table, mach, Parameter::Mach);
                if (!table[mach].is_string() && !(result.mach > 0))
                    reader_.error("prescribed Mach number must be positive");
                result.direction = vector(table, "direction", Parameter::Direction);
            }
            return result;
        }
        if (type == "SUPERSONIC_OUTLET" || type == "OUTLET_SUPERSONIC") {
            keys(table, {"type"});
            return SupersonicOutlet{};
        }
        if (type == "VELOCITY_INLET") {
            keys(table, {"type", "velocity", "temperature"});
            return VelocityInlet<double>{vector(table, "velocity", Parameter::Velocity),
                table.contains("temperature") ? scalar(table, "temperature", Parameter::Temperature) : 0.0};
        }
        reader_.error("unsupported physical type '" + type + "'");
    }
};

Configuration decode(const toml::table& root, const TomlReader& reader) {
    Configuration result;
    const auto& entries = reader.array(root, "boundary_condition");
    std::unordered_set<std::int64_t> seen;
    result.patches.reserve(entries.size());
    for (const auto& entry : entries) {
        const auto* table = entry.as_table();
        if (!table) reader.error("boundary_condition entries must be tables");
        const auto id = reader.integer(*table, "patch_id");
        if (id < 0) reader.error("patch_id must be a nonnegative integer");
        if (!seen.insert(id).second) reader.error("duplicate patch_id " + std::to_string(id));
        auto name = reader.string(*table, "name");
        const TomlReader condition_reader{"Physical boundary configuration: patch " + std::to_string(id) + " ('" + name + "')"};
        auto condition = ConditionReader{condition_reader}.parse(*table);
        result.patches.push_back({id, std::move(name), std::move(condition)});
    }

    const auto& unlabelled = reader.table(root, "unlabelled");
    if (unlabelled.contains("type")) {
        const TomlReader condition_reader{"Physical boundary configuration: unlabelled faces"};
        result.unlabelled = ConditionReader{condition_reader}.parse(unlabelled);
    }
    return result;
}

} // namespace

Configuration parse_config(std::string_view text) {
    return mpi::guarded([&] {
        const TomlReader reader{"Physical boundary configuration"};
        return decode(reader.parse_text(text), reader);
    });
}

Configuration read_config(const std::filesystem::path& file, mpi::CommView comm) {
    return parse_config(toml_config::read_broadcast(file, comm));
}

} // namespace krona::boundary::physical