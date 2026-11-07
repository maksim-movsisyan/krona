#pragma once

#include <cmath>

namespace krona::mesh::generation {

/// @brief Stores a Cartesian point or vector without a device dependency.
struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    /// @brief Creates the zero vector.
    constexpr Vec3() noexcept = default;

    /// @brief Stores Cartesian components.
    constexpr Vec3(double x_, double y_, double z_) noexcept : x(x_), y(y_), z(z_) {}

    /// @brief Adds components.
    [[nodiscard]] constexpr Vec3 operator+(const Vec3 &other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    /// @brief Subtracts components.
    [[nodiscard]] constexpr Vec3 operator-(const Vec3 &other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    /// @brief Multiplies every component by a scalar.
    [[nodiscard]] constexpr Vec3 operator*(double scale) const noexcept {
        return {x * scale, y * scale, z * scale};
    }

    /// @brief Divides every component by a nonzero scalar.
    [[nodiscard]] constexpr Vec3 operator/(double scale) const noexcept {
        return {x / scale, y / scale, z / scale};
    }

    /// @brief Returns the scalar product.
    [[nodiscard]] constexpr double dot(const Vec3 &other) const noexcept {
        return x * other.x + y * other.y + z * other.z;
    }

    /// @brief Returns the vector product.
    [[nodiscard]] constexpr Vec3 cross(const Vec3 &other) const noexcept {
        return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }

    /// @brief Returns the Euclidean length without squaring large components.
    [[nodiscard]] double norm() const noexcept { return std::hypot(x, y, z); }
};

} // namespace krona::mesh::generation
