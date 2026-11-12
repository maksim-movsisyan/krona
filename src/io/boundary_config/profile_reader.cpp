#include <krona/io/boundary_config/profile_reader.hpp>

#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace krona::io::boundary_config {
namespace {

using boundary::profile::TabulatedProfile;

struct Table {
    TabulatedProfile::Axes axes;
    std::vector<double> values;
};

struct Sample {
    std::array<double, 3> point{};
    double value = 0;
    std::size_t line = 0;
};

[[noreturn]] void csv_error(const std::filesystem::path& path, std::size_t line, std::string_view message) {
    auto location = path.string();
    if (line != 0) location += ":" + std::to_string(line);
    mpi::fatal(location + ": " + std::string{message});
}

std::string_view trim(std::string_view text) {
    constexpr std::string_view whitespace = " \t\r";
    const auto first = text.find_first_not_of(whitespace);
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}

Table read_table(const std::filesystem::path& path) {
    std::ifstream in{path};
    if (!in) csv_error(path, 0, "Cannot open profile CSV");

    constexpr std::array<std::string_view, 4> names{"x", "y", "z", "value"};
    std::array<std::size_t, names.size()> columns{};
    std::array<bool, names.size()> present{};
    std::size_t column_count = 0;
    std::vector<Sample> samples;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        auto remaining = trim(line);
        if (remaining.empty()) continue;
        const bool header = column_count == 0;
        Sample sample;
        sample.line = line_number;
        std::size_t column = 0;
        for (;;) {
            if (column == (header ? names.size() : column_count))
                csv_error(path, line_number, "Too many CSV columns");
            const auto comma = remaining.find(',');
            auto field = trim(remaining.substr(0, comma));
            if (header) {
                const auto name = std::ranges::find(names, field);
                if (name == names.end()) csv_error(path, line_number, "Unknown profile column");
                const std::size_t d = static_cast<std::size_t>(name - names.begin());
                if (present[d]) csv_error(path, line_number, "Duplicate profile column");
                present[d] = true;
                columns[column] = d;
            } else {
                if (field.starts_with('+')) {
                    field.remove_prefix(1);
                    if (field.starts_with('-')) csv_error(path, line_number, "Invalid profile number");
                }
                if (field.empty()) csv_error(path, line_number, "Missing profile number");
                const auto d = columns[column];
                auto& number = d < sample.point.size() ? sample.point[d] : sample.value;
                const auto last = field.data() + field.size();
                const auto parsed = std::from_chars(field.data(), last, number);
                if (parsed.ec != std::errc{} || parsed.ptr != last || !std::isfinite(number))
                    csv_error(path, line_number, "Expected a finite profile number");
            }
            ++column;
            if (comma == std::string_view::npos) break;
            remaining.remove_prefix(comma + 1);
        }
        if (header) {
            if (!present.back() || column < 2) csv_error(path, line_number, "Expected value and at least one axis");
            column_count = column;
        } else {
            if (column != column_count) csv_error(path, line_number, "Too few CSV columns");
            samples.push_back(sample);
        }
    }
    if (in.bad() || !in.eof()) csv_error(path, line_number, "Error reading profile CSV");
    if (samples.empty()) csv_error(path, 0, "Profile CSV contains no samples");

    Table table;
    std::array<std::size_t, 3> strides{};
    std::size_t extent = 1;
    for (std::size_t d = 0; d < table.axes.size(); ++d) {
        if (!present[d]) continue;
        std::vector<double> coordinates;
        coordinates.reserve(samples.size());
        for (const auto& sample : samples) coordinates.push_back(sample.point[d]);
        std::ranges::sort(coordinates);
        const auto unique = std::ranges::unique(coordinates);
        auto& axis = table.axes[d];
        axis.assign(coordinates.begin(), unique.begin());
        if (axis.size() < 2) csv_error(path, 0, "Each active axis needs at least two coordinates");
        if (axis.size() > samples.size() / extent)
            csv_error(path, 0, "Profile rows do not fill a complete Cartesian grid");
        strides[d] = extent;
        extent *= axis.size();
    }
    if (extent != samples.size()) csv_error(path, 0, "Expected exactly one row per grid point");

    // A finite sample replaces each NaN sentinel exactly once.
    table.values.assign(extent, std::numeric_limits<double>::quiet_NaN());
    for (const auto& sample : samples) {
        std::size_t index = 0;
        for (std::size_t d = 0; d < table.axes.size(); ++d) {
            const auto& axis = table.axes[d];
            if (axis.empty()) continue;
            const auto at = std::ranges::lower_bound(axis, sample.point[d]);
            index += static_cast<std::size_t>(at - axis.begin()) * strides[d];
        }
        if (!std::isnan(table.values[index])) csv_error(path, sample.line, "Duplicate profile grid point");
        table.values[index] = sample.value;
    }
    return table;
}

void broadcast_column(std::vector<double>& column, mpi::CommView comm) {
    auto count = static_cast<std::uint64_t>(column.size());
    mpi::broadcast(comm, count);
    mpi::require(count <= column.max_size(), "Profile column is too large");
    if (comm.rank() != 0) column.resize(static_cast<std::size_t>(count));
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    auto pending = std::span{column};
    while (!pending.empty()) {
        const auto chunk = std::min(limit, pending.size());
        mpi::broadcast(comm, pending.first(chunk));
        pending = pending.subspan(chunk);
    }
}

} // namespace

TabulatedProfile read_tabulated_profile_csv(const std::filesystem::path& path, mpi::CommView comm) {
    return mpi::guarded([&] {
        mpi::require(!comm.is_null(), "Profile CSV loading requires a live communicator");
        Table table;
        if (comm.rank() == 0) table = read_table(path);
        for (auto& axis : table.axes) broadcast_column(axis, comm);
        broadcast_column(table.values, comm);
        return TabulatedProfile{std::move(table.axes), std::move(table.values)};
    });
}

} // namespace krona::io::boundary_config
