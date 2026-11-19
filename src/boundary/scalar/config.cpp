#include <krona/boundary/scalar/config.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/toml_config/toml.hpp>

#include <unordered_set>

namespace krona::boundary::scalar {
namespace {

/// @brief Reads a condition's type and the constant or CSV path expected by that type.
/// @param table One scalar field's condition table.
/// @param reader TOML reader carrying the patch and scalar name for diagnostics.
[[nodiscard]] ConditionConfig parse_condition(const toml::table& table, const toml_config::TomlReader& reader) {
    const auto type = reader.string(table, "type");
    Kind kind;
    std::string_view parameter;
    if (type == "FIXED_VALUE") {
        kind = Kind::FixedValue;
        parameter = "value";
    } else if (type == "NORMAL_GRADIENT") {
        kind = Kind::NormalGradient;
        parameter = "gradient";
    } else if (type == "FIXED_FLUX") {
        kind = Kind::FixedFlux;
        parameter = "flux";
    } else if (type == "ZERO_GRADIENT") {
        // Store zero gradient as an ordinary prescribed derivative.
        reader.check_keys(table, {"type"});
        return {Kind::NormalGradient, 0.0};
    } else {
        reader.error("unsupported scalar type '" + type + "'");
    }

    // A string names a CSV profile. A number applies across the entire patch.
    reader.check_keys(table, {"type", parameter});
    if (table[parameter].is_string())
        return {kind, std::filesystem::path{reader.nonempty_string(table, parameter)}};
    return {kind, reader.number(table, parameter)};
}

/// @brief Collects the requested scalar's definition for every labelled patch and the optional unlabelled group.
/// @param root Parsed TOML document.
/// @param field Name used to select a scalar table in each patch.
/// @param reader TOML reader carrying the configuration source for diagnostics.
[[nodiscard]] Configuration decode(const toml::table& root, std::string_view field,
                                   const toml_config::TomlReader& reader) {
    if (field.empty()) reader.error("field name must not be empty");
    Configuration result;
    result.field = field;
    const auto& entries = reader.array(root, "boundary_condition");
    std::unordered_set<std::int64_t> seen;
    result.patches.reserve(entries.size());
    // Reject duplicate IDs here. The manager checks IDs and names against the mesh catalog.
    for (const auto& node : entries) {
        const auto* table = node.as_table();
        if (!table) reader.error("boundary_condition entries must be tables");
        const auto id = reader.integer(*table, "patch_id");
        if (id < 0) reader.error("patch_id must be a nonnegative integer");
        if (!seen.insert(id).second) reader.error("duplicate patch_id " + std::to_string(id));
        auto name = reader.string(*table, "name");
        const toml_config::TomlReader condition_reader{
            "Scalar boundary configuration: patch " + std::to_string(id) + " ('" + name +
            "'), scalar '" + std::string{field} + "'"};
        const auto& scalar = condition_reader.table(*table, "scalar");
        if (!scalar.contains(field)) condition_reader.error("missing scalar condition table");
        auto condition = parse_condition(condition_reader.table(scalar, field, true), condition_reader);
        result.patches.push_back({id, std::move(name), std::move(condition)});
    }

    // A mesh with unlabelled boundary faces will require this optional entry when finalizing.
    const auto& unlabelled = reader.table(root, "unlabelled");
    const auto& scalar = reader.table(unlabelled, "scalar");
    if (scalar.contains(field)) {
        const toml_config::TomlReader condition_reader{
            "Scalar boundary configuration: unlabelled faces, scalar '" + std::string{field} + "'"};
        result.unlabelled = parse_condition(condition_reader.table(scalar, field, true), condition_reader);
    }
    return result;
}

} // namespace

Configuration parse_config(std::string_view text, std::string_view field) {
    return mpi::guarded([&] {
        const toml_config::TomlReader reader{"Scalar boundary configuration"};
        return decode(reader.parse_text(text), field, reader);
    });
}

Configuration read_config(const std::filesystem::path& file, std::string_view field, mpi::CommView comm) {
    return parse_config(toml_config::read_broadcast(file, comm), field);
}

} // namespace krona::boundary::scalar