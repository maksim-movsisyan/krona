#pragma once

#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace krona::boundary::profile {

/// @brief Stores a Cartesian table of scalar values and interpolates it on the host.
/// @details A boundary configuration can sample the table at local face centroids during setup.
class TabulatedProfile {
public:
    /// @brief Coordinates in x, y, z order. An empty vector omits that axis.
    using Axes = std::array<std::vector<double>, 3>;

    /// @brief Takes ownership of the axes and grid values and checks that the grid is complete.
    /// @param axes Coordinates in x, y, z order. Supply at least one axis. Each active axis
    /// needs at least two finite, strictly increasing coordinates.
    /// @param values One finite value per grid point. X varies fastest, then y and z.
    /// Skip omitted axes when flattening the grid.
    TabulatedProfile(Axes axes, std::vector<double> values)
        : axes_(std::move(axes)), values_(std::move(values)) {
        std::size_t extent = 1;
        for (std::size_t d = 0; d < axes_.size(); ++d) {
            const auto& axis = axes_[d];
            if (axis.empty()) continue;
            mpi::require(axis.size() >= 2 && std::ranges::all_of(axis, [](double x) { return std::isfinite(x); }),
                         "Profile axes require at least two finite coordinates");
            for (std::size_t i = 1; i < axis.size(); ++i) {
                mpi::require(axis[i] > axis[i - 1], "Profile axis coordinates must be strictly increasing");
            }
            mpi::require(axis.size() < std::numeric_limits<std::size_t>::max() / extent,
                         "Profile table extent overflows size_t");

            // A step along this axis skips the product of all earlier active axis lengths.
            strides_[d] = extent;
            extent *= axis.size();
        }
        mpi::require(extent != 1 && values_.size() == extent,
                     "Profile values must fill a grid with at least one active axis");

        mpi::require(std::ranges::all_of(values_, [](double value) { return std::isfinite(value); }),
                     "Profile values must be finite");
    }

    /// @brief Interpolates linearly along each active axis. Both endpoints of an active axis are valid.
    /// @param x,y,z Point coordinates. Coordinates along omitted axes are ignored.
    [[nodiscard]] double evaluate(double x, double y, double z) const {
        const std::array point{x, y, z};
        std::size_t base = 0;
        std::array<double, 3> weight{};
        for (std::size_t d = 0; d < axes_.size(); ++d) {
            const auto& axis = axes_[d];
            if (axis.empty()) continue;
            const double coordinate = point[d];
            mpi::require(axis.front() <= coordinate && coordinate <= axis.back(),
                         "Profile coordinate is outside the tabulated axis");

            // Keep the last node available as the upper neighbour when the point is on that node.
            const auto upper = std::upper_bound(axis.begin(), axis.end() - 1, coordinate);
            const auto lower = upper - 1;
            base += static_cast<std::size_t>(lower - axis.begin()) * strides_[d];
            const double width = *upper - *lower;
            // Halve first when a finite interval spans more than double can represent.
            weight[d] = std::isfinite(width) ? (coordinate - *lower) / width
                                            : (coordinate * 0.5 - *lower * 0.5) /
                                                  (*upper * 0.5 - *lower * 0.5);
        }

        // Interpolate the corners of the enclosing interval, rectangle or box along x, then y and z.
        const auto along_x = [&](std::size_t i) {
            return strides_[0] == 0 ? values_[i] : std::lerp(values_[i], values_[i + strides_[0]], weight[0]);
        };
        const auto along_y = [&](std::size_t i) {
            return strides_[1] == 0 ? along_x(i) : std::lerp(along_x(i), along_x(i + strides_[1]), weight[1]);
        };
        return strides_[2] == 0 ? along_y(base) : std::lerp(along_y(base), along_y(base + strides_[2]), weight[2]);
    }

private:
    Axes axes_;
    std::vector<double> values_;
    std::array<std::size_t, 3> strides_{}; ///< Flat-grid stride per axis, zero when omitted.
};

} // namespace krona::boundary::profile
