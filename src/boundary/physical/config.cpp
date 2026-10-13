#include <krona/boundary/physical/config.hpp>
#include <krona/mpi/error.hpp>

#if __has_include(<toml++/toml.hpp>)
#include <toml++/toml.hpp>
#elif __has_include(<toml++/toml.h>)
#include <toml++/toml.h>
#else
#include <toml.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

namespace krona::boundary::physical {
namespace {

std::string string(const toml::table& table, std::string_view key)
{
    const auto value = table[key].value<std::string>();
    mpi::require(value.has_value(), "Physical boundary TOML: missing or invalid string");
    return *value;
}

double number(const toml::node& node)
{
    const auto value = node.value<double>();
    mpi::require(value.has_value() && std::isfinite(*value), "Physical boundary TOML: expected a finite number");
    return *value;
}

std::optional<double> optional_number(const toml::table& table, std::string_view key)
{
    const auto* node = table.get(key);
    return node ? std::optional<double>{number(*node)} : std::nullopt;
}

std::array<double, 3> vector(const toml::table& table, std::string_view key, bool required)
{
    const auto* node = table.get(key);
    if (!node) {
        mpi::require(!required, "Physical boundary TOML: missing required vector");
        return {};
    }
    const auto* array = node->as_array();
    mpi::require(array && array->size() == 3, "Physical boundary TOML: vector must contain three numbers");
    return {number((*array)[0]), number((*array)[1]), number((*array)[2])};
}

Kind kind(std::string_view type)
{
    if (type == "INLET_VELOCITY") return Kind::VelocityInlet;
    if (type == "INLET_TOTAL_TP") return Kind::TotalInlet;
    if (type == "INLET_SUPERSONIC") return Kind::SupersonicInlet;
    if (type == "OUTLET_PRESSURE") return Kind::PressureOutlet;
    if (type == "INLET_OUTLET") return Kind::InletOutlet;
    if (type == "OUTLET_SUPERSONIC") return Kind::SupersonicOutlet;
    if (type == "WALL_ADIABATIC") return Kind::AdiabaticWall;
    if (type == "WALL_ISOTHERMAL") return Kind::IsothermalWall;
    if (type == "WALL_SLIP") return Kind::SlipWall;
    if (type == "SYMMETRY") return Kind::Symmetry;
    if (type == "EXTRAPOLATE") return Kind::Extrapolate;
    mpi::fatal("Physical boundary TOML: unsupported physical type");
}

ConditionConfig condition(const toml::table& entry)
{
    ConditionConfig c;
    c.kind = kind(string(entry, "type"));
    const toml::table empty;
    const auto* node = entry.get("physical");
    mpi::require(!node || node->is_table(), "Physical boundary TOML: physical must be a table");
    const auto& table = node ? *node->as_table() : empty;
    const bool total = c.kind == Kind::TotalInlet;
    const bool pressure = c.kind == Kind::PressureOutlet || c.kind == Kind::InletOutlet || c.kind == Kind::SupersonicInlet;
    const bool velocity = c.kind == Kind::VelocityInlet || c.kind == Kind::InletOutlet || c.kind == Kind::SupersonicInlet ||
                          c.kind == Kind::AdiabaticWall || c.kind == Kind::IsothermalWall || c.kind == Kind::SlipWall;
    const bool temperature = velocity && c.kind != Kind::AdiabaticWall;
    for (const auto& [key, value] : table) {
        (void)value;
        const auto k = key.str();
        mpi::require((pressure && k == "pressure") || (velocity && k == "velocity") ||
                     (temperature && k == "temperature") ||
                     (total && (k == "total_pressure" || k == "total_temperature" || k == "direction")),
                     "Physical boundary TOML: unexpected key or inapplicable physical parameter");
    }
    c.pressure = optional_number(table, "pressure");
    c.temperature = optional_number(table, "temperature");
    c.total_pressure = optional_number(table, "total_pressure");
    c.total_temperature = optional_number(table, "total_temperature");
    c.velocity = vector(table, "velocity", c.kind == Kind::VelocityInlet || c.kind == Kind::InletOutlet ||
                                           c.kind == Kind::SupersonicInlet);
    c.direction = vector(table, "direction", total);
    mpi::require(!pressure || c.pressure.has_value(), "Physical boundary TOML: pressure is required");
    mpi::require(!total || (c.total_pressure && c.total_temperature),
                 "Physical boundary TOML: total pressure and total temperature are required");
    mpi::require((c.kind != Kind::SupersonicInlet && c.kind != Kind::IsothermalWall) || c.temperature.has_value(),
                 "Physical boundary TOML: static temperature is required");
    return c;
}

} // namespace

Configuration parse_config(std::string_view text)
{
    return mpi::guarded([&] {
#if TOML_EXCEPTIONS
        const auto root = toml::parse(text);
#else
        auto parsed = toml::parse(text);
        if (!parsed) mpi::fatal(parsed.error().description());
        const auto root = std::move(parsed).table();
#endif
        const auto* patches = root["boundary_condition"].as_array();
        mpi::require(patches != nullptr, "Physical boundary TOML: boundary_condition must be an array of tables");
        Configuration config;
        config.patches.reserve(patches->size());
        std::vector<std::int64_t> ids;
        ids.reserve(patches->size());
        for (const auto& patch : *patches) {
            const auto* p = patch.as_table();
            mpi::require(p != nullptr, "Physical boundary TOML: boundary_condition entry must be a table");
            const auto id = (*p)["patch_id"].value<std::int64_t>();
            mpi::require(id.has_value() && *id >= 0, "Physical boundary TOML: invalid patch_id");
            config.patches.push_back({*id, string(*p, "name"), condition(*p)});
            ids.push_back(*id);
        }
        std::sort(ids.begin(), ids.end());
        mpi::require(std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "Physical boundary TOML: duplicate patch_id");
        if (const auto* node = root.get("unlabelled")) {
            const auto* table = node->as_table();
            mpi::require(table != nullptr, "Physical boundary TOML: unlabelled must be a table");
            // A scalar-only unlabelled section is independent of this physical catalogue.
            if (table->contains("type") || table->contains("physical")) config.unlabelled = condition(*table);
        }
        return config;
    });
}

Configuration read_config(const std::filesystem::path& path)
{
    return mpi::guarded([&] {
        std::ifstream file(path);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::ostringstream text;
        text << file.rdbuf();
        return parse_config(text.str());
    });
}

} // namespace krona::boundary::physical
