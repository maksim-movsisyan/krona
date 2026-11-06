#pragma once

#include <krona/mpi/error.hpp>

#if __has_include(<toml++/toml.hpp>)
#include <toml++/toml.hpp>
#elif __has_include(<toml++/toml.h>)
#include <toml++/toml.h>
#elif __has_include(<toml.hpp>)
#include <toml.hpp>
#else
#error "Configuration parsing requires toml++"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace krona::toml_config {

/// @brief Reads TOML values and reports errors with a module-specific prefix.
/// Helpers are local; wrap the complete module decoder in mpi::guarded.
class TomlReader final
{
public:
    /// @brief Owns the diagnostic prefix; no communicator is retained.
    explicit TomlReader(std::string context = "Configuration") : context_(std::move(context)) {}

    /// @brief Aborts through mpi::fatal with the reader's diagnostic prefix.
    [[noreturn]] void error(std::string_view message) const
    {
        mpi::fatal(context_ + ": " + std::string{message});
    }

    /// @brief Parses text locally; source is an optional diagnostic filename.
    [[nodiscard]] toml::table parse_text(std::string_view text, std::string_view source = {}) const noexcept
    {
        return mpi::guarded([&]() -> toml::table {
#if TOML_EXCEPTIONS
            return toml::parse(text, source);
#else
            auto result = toml::parse(text, source);
            if (!result) parse_error(result.error());
            return std::move(result).table();
#endif
        });
    }

    /// @brief Reads and parses a file on the calling rank only.
    [[nodiscard]] toml::table parse_file(const std::filesystem::path& file) const noexcept
    {
        return mpi::guarded([&]() -> toml::table {
#if TOML_EXCEPTIONS
            return toml::parse_file(file.string());
#else
            auto result = toml::parse_file(file.string());
            if (!result) parse_error(result.error());
            return std::move(result).table();
#endif
        });
    }

    /// @brief Rejects keys outside allowed and explicitly ignored metadata.
    void check_keys(const toml::table& t, 
                    std::initializer_list<std::string_view> allowed,
                    std::string_view where = {},
                    std::initializer_list<std::string_view> ignored = {}) const
    {
        for (const auto& [key, node] : t) {
            (void)node;
            const auto name = key.str();
            if (std::find(allowed.begin(), allowed.end(), name) != allowed.end() ||
                std::find(ignored.begin(), ignored.end(), name) != ignored.end()) continue;
            const auto prefix = where.empty() ? std::string{} : std::string{where} + ": ";
            error(prefix + "unexpected key '" + std::string{name} + "'");
        }
    }

    /// @brief Selects one spelling of a key and rejects simultaneous aliases.
    /// Returns canonical when neither spelling is present.
    [[nodiscard]] std::string_view alias(const toml::table& t, 
                                         std::string_view canonical, 
                                         std::string_view alternate) const
    {
        if (t.contains(canonical) && t.contains(alternate))
            error("duplicate parameter aliases for '" + std::string{canonical} + "'");
        return t.contains(canonical) || !t.contains(alternate) ? canonical : alternate;
    }

    /// @brief Borrows a table; an absent optional table yields an empty table.
    [[nodiscard]] const toml::table& table(const toml::table& t, 
                                           std::string_view key, 
                                           bool required = false) const
    {
        static const toml::table empty;
        const auto* node = t.get(key);
        if (!node && !required) return empty;
        if (!node || !node->is_table()) expected(key, "a table");
        return *node->as_table();
    }

    /// @brief Borrows an array; an absent optional array yields an empty array.
    [[nodiscard]] const toml::array& array(const toml::table& t, 
                                           std::string_view key, 
                                           bool required = true) const
    {
        static const toml::array empty;
        const auto* node = t.get(key);
        if (!node && !required) return empty;
        if (!node || !node->is_array()) expected(key, "an array");
        return *node->as_array();
    }

    /// @brief Reads a required string, preserving empty strings and embedded NULs.
    [[nodiscard]] std::string string(const toml::table& t, std::string_view key) const
    {
        const auto value = t[key].value<std::string>();
        if (!value) expected(key, "a string");
        return *value;
    }

    /// @brief Uses fallback only when the string key is absent.
    [[nodiscard]] std::string string(const toml::table& t, std::string_view key, std::string fallback) const
    {
        return t.contains(key) ? string(t, key) : std::move(fallback);
    }

    /// @brief Reads a required nonempty string without embedded NULs.
    [[nodiscard]] std::string nonempty_string(const toml::table& t, std::string_view key) const
    {
        return checked_string(string(t, key), key);
    }

    /// @brief Reads an optional nonempty string without embedded NULs.
    [[nodiscard]] std::string nonempty_string(const toml::table& t, std::string_view key, std::string fallback) const
    {
        return checked_string(string(t, key, std::move(fallback)), key);
    }

    /// @brief Reads a required finite number; integer TOML values are accepted.
    [[nodiscard]] double number(const toml::table& t, std::string_view key) const
    {
        const auto node = t[key];
        const auto value = node.value<double>();
        if (!node.is_number() || !value || !std::isfinite(*value))
            expected(key, "a finite number");
        return *value;
    }

    /// @brief Uses fallback only when the numeric key is absent.
    [[nodiscard]] double number(const toml::table& t, std::string_view key, double fallback) const
    {
        return t.contains(key) ? number(t, key) : fallback;
    }

    /// @brief Reads one of two numeric aliases and rejects duplicates.
    [[nodiscard]] double number(const toml::table& t, std::string_view canonical, std::string_view alternate) const
    {
        return number(t, alias(t, canonical, alternate));
    }

    /// @brief Reads a required boolean without numeric or string conversion.
    [[nodiscard]] bool boolean(const toml::table& t, std::string_view key) const
    {
        const auto value = t[key].value<bool>();
        if (!value) expected(key, "a boolean");
        return *value;
    }

    /// @brief Uses fallback only when the boolean key is absent.
    [[nodiscard]] bool boolean(const toml::table& t, std::string_view key, bool fallback) const
    {
        return t.contains(key) ? boolean(t, key) : fallback;
    }

    /// @brief Reads a required integer representable by T; floats are rejected.
    template<std::integral T = std::int64_t> requires (!std::same_as<T, bool>)
    [[nodiscard]] T integer(const toml::table& t, std::string_view key) const
    {
        const auto node = t[key];
        const auto value = node.value<std::int64_t>();
        if (!node.is_integer() || !value)
            expected(key, "an integer in the destination type's range");
        if constexpr (std::is_unsigned_v<T>) {
            if (*value < 0 || static_cast<std::uint64_t>(*value) > std::numeric_limits<T>::max())
                expected(key, "an integer in the destination type's range");
        } else {
            if (*value < std::numeric_limits<T>::lowest() || *value > std::numeric_limits<T>::max())
                expected(key, "an integer in the destination type's range");
        }
        return static_cast<T>(*value);
    }

    /// @brief Uses fallback only when the integer key is absent.
    template<std::integral T> requires (!std::same_as<T, bool>)
    [[nodiscard]] T integer(const toml::table& t, std::string_view key, T fallback) const
    {
        return t.contains(key) ? integer<T>(t, key) : fallback;
    }

    /// @brief Reads a positive integer or checks a supplied default.
    template<std::integral T> requires (!std::same_as<T, bool>)
    [[nodiscard]] T positive_integer(const toml::table& t, std::string_view key, T fallback) const
    {
        const auto value = integer(t, key, fallback);
        if (value <= 0) expected(key, "a positive integer");
        return value;
    }

    /// @brief Reads exactly N finite numbers; N defaults to three.
    template<std::size_t N = 3>
    [[nodiscard]] std::array<double, N> vector(const toml::table& t, std::string_view key) const
    {
        const auto& values = array(t, key);
        if (values.size() != N)
            expected(key, "an array of " + std::to_string(N) + " finite numbers");
        std::array<double, N> result{};
        for (std::size_t i = 0; i < N; ++i) {
            const auto value = values[i].value<double>();
            if (!values[i].is_number() || !value || !std::isfinite(*value))
                error("'" + std::string{key} + "[" + std::to_string(i) + "]' must be a finite number");
            result[i] = *value;
        }
        return result;
    }

    /// @brief Uses fallback only when the vector key is absent.
    template<std::size_t N = 3>
    [[nodiscard]] std::array<double, N> vector(const toml::table& t, std::string_view key, std::array<double, N> fallback) const
    {
        return t.contains(key) ? vector<N>(t, key) : fallback;
    }

    /// @brief Maps a required string to one explicitly listed value.
    template<class E>
    [[nodiscard]] E choice(const toml::table& t, std::string_view key,
                           std::initializer_list<std::pair<std::string_view, E>> choices) const
    {
        return selected(string(t, key), key, choices);
    }

    /// @brief Maps an optional string to one explicitly listed value.
    template<class E>
    [[nodiscard]] E choice(const toml::table& t, std::string_view key, std::string fallback,
                           std::initializer_list<std::pair<std::string_view, E>> choices) const
    {
        return selected(string(t, key, std::move(fallback)), key, choices);
    }

private:
    [[noreturn]] void expected(std::string_view key, std::string_view type) const
    {
        error("'" + std::string{key} + "' must be " + std::string{type});
    }

    [[noreturn]] void parse_error(const toml::parse_error& cause) const
    {
        std::ostringstream message;
        message << cause;
        error(message.str());
    }

    [[nodiscard]] std::string checked_string(std::string value, std::string_view key) const
    {
        if (value.empty() || value.find('\0') != std::string::npos)
            expected(key, "a nonempty string without embedded NULs");
        return value;
    }

    template<class E>
    [[nodiscard]] E selected(const std::string& value, std::string_view key,
                             std::initializer_list<std::pair<std::string_view, E>> choices) const
    {
        for (const auto& [name, code] : choices)
            if (value == name) return code;
        error("unsupported " + std::string{key} + " '" + value + "'");
    }

    std::string context_;
};

} // namespace krona::toml_config
