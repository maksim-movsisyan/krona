#include <krona/boundary/scalar/config.hpp>
#include <krona/boundary/detail/support.hpp>

#if __has_include(<toml++/toml.hpp>)
#include <toml++/toml.hpp>
#elif __has_include(<toml++/toml.h>)
#include <toml++/toml.h>
#elif __has_include(<toml.hpp>)
#include <toml.hpp>
#else
#error "Scalar boundary configuration requires toml++ headers"
#endif

#include <cmath>
#include <unordered_set>
#include <utility>

namespace krona::boundary::scalar {
namespace {

[[noreturn]] void config_error(const std::string& message)
{ mpi::fatal("Scalar boundary configuration: " + message); }

[[nodiscard]] std::string required_string(const toml::table& table, std::string_view key,
                                          const std::string& where)
{
    auto value = table[key].value<std::string>();
    if (!value) config_error(where + ": missing string '" + std::string{key} + "'");
    return std::move(*value);
}

[[nodiscard]] ConditionConfig parse_condition(const toml::table& table, const std::string& where)
{
    const auto type = required_string(table, "type", where);
    ConditionConfig condition;
    std::string_view parameter;
    if (type == "FIXED_VALUE") { condition.kind = Kind::FixedValue; parameter = "value"; }
    else if (type == "NORMAL_GRADIENT") { condition.kind = Kind::NormalGradient; parameter = "gradient"; }
    else if (type == "FIXED_FLUX") { condition.kind = Kind::FixedFlux; parameter = "flux"; }
    else if (type == "ZERO_GRADIENT") condition.kind = Kind::NormalGradient;
    else config_error(where + ": unsupported scalar type '" + type + "'");
    // Strict scalar keys catch misspellings; unrelated physics and import metadata stay untouched.
    for (const auto& [key, node] : table) {
        (void)node;
        if (key.str() != "type" && (parameter.empty() || key.str() != parameter))
            config_error(where + ": unexpected key '" + std::string{key.str()} + "'");
    }
    if (!parameter.empty()) {
        const auto node = table[parameter];
        const auto value = node.value<double>();
        if (!node || !node.is_number() || !value || !std::isfinite(*value))
            config_error(where + ": '" + std::string{parameter} + "' must be a finite number");
        condition.parameter = *value;
    }
    return condition;
}

[[nodiscard]] Configuration decode(const toml::table& root, std::string_view field)
{
    if (field.empty()) config_error("field name must not be empty");
    Configuration result;
    result.field = field;
    const auto* entries = root.get_as<toml::array>("boundary_condition");
    if (!entries) config_error("expected [[boundary_condition]] entries or boundary_condition = []");
    std::unordered_set<std::int64_t> seen;
    result.patches.reserve(entries->size());
    for (const auto& node : *entries) {
        const auto* table = node.as_table();
        if (!table) config_error("boundary_condition entries must be tables");
        const auto id = (*table)["patch_id"].value<std::int64_t>();
        if (!id || *id < 0) config_error("patch_id must be a nonnegative integer");
        if (!seen.insert(*id).second) config_error("duplicate patch_id " + std::to_string(*id));
        const auto name = required_string(*table, "name", "patch " + std::to_string(*id));
        const auto where = "patch " + std::to_string(*id) + " ('" + name + "'), scalar '" + std::string{field} + "'";
        const auto* scalar = table->get_as<toml::table>("scalar");
        const auto* condition = scalar ? scalar->get_as<toml::table>(field) : nullptr;
        if (!condition) config_error(where + ": missing scalar condition table");
        result.patches.push_back({*id, name, parse_condition(*condition, where)});
    }
    if (const auto* unlabelled = root.get_as<toml::table>("unlabelled")) {
        if (const auto* scalar = unlabelled->get_as<toml::table>("scalar")) {
            if (const auto* condition = scalar->get_as<toml::table>(field))
                result.unlabelled = parse_condition(*condition, "unlabelled faces, scalar '" + std::string{field} + "'");
        }
    }
    return result;
}

#if !TOML_EXCEPTIONS
[[noreturn]] void parse_error(const toml::parse_error& error)
{
    std::ostringstream message;
    message << error;
    config_error(message.str());
}
#endif

} // namespace

Configuration parse_config(std::string_view text, std::string_view field)
{
    return detail::guarded([&] {
#if TOML_EXCEPTIONS
        return decode(toml::parse(text), field);
#else
        auto parsed = toml::parse(text);
        if (!parsed) parse_error(parsed.error());
        return decode(parsed.table(), field);
#endif
    });
}

Configuration read_config(const std::filesystem::path& file, std::string_view field)
{
    return detail::guarded([&] {
#if TOML_EXCEPTIONS
        return decode(toml::parse_file(file.string()), field);
#else
        auto parsed = toml::parse_file(file.string());
        if (!parsed) parse_error(parsed.error());
        return decode(parsed.table(), field);
#endif
    });
}

} // namespace krona::boundary::scalar
