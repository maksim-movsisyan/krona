#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace krona::boundary::physical {

template<class Real>
struct Vector3 {
    Real x = 0, y = 0, z = 0;
    KOKKOS_INLINE_FUNCTION Real dot(Vector3 other) const
    { return x * other.x + y * other.y + z * other.z; }
    KOKKOS_INLINE_FUNCTION Vector3 operator+(Vector3 other) const
    { return {x + other.x, y + other.y, z + other.z}; }
    KOKKOS_INLINE_FUNCTION Vector3 operator-(Vector3 other) const
    { return {x - other.x, y - other.y, z - other.z}; }
    KOKKOS_INLINE_FUNCTION Vector3 operator*(Real scale) const
    { return {x * scale, y * scale, z * scale}; }
    KOKKOS_INLINE_FUNCTION bool finite() const
    { return Kokkos::isfinite(x) && Kokkos::isfinite(y) && Kokkos::isfinite(z); }
};

/// @brief Primitive face state; the athermal specialization has no temperature member.
template<class Real, bool Thermal>
struct State {
    using real_type = Real;
    static constexpr bool has_temperature = Thermal;
    Real density = 0, pressure = 0;
    Vector3<Real> velocity;
    Real temperature = 0;
};

template<class Real>
struct State<Real, false> {
    using real_type = Real;
    static constexpr bool has_temperature = false;
    Real density = 0, pressure = 0;
    Vector3<Real> velocity;
};

/// @brief Physical face geometry; unit normal points out of the owned cell.
template<class Real>
struct Context {
    LocalIndex face = 0, owner = 0;
    std::size_t boundary_index = 0, extended_index = 0;
    Vector3<GeometryReal> position, displacement, tangential_displacement;
    Vector3<Real> normal;
    GeometryReal distance = 0, area = 0;
    Real time = 0;
    Real temperature_tangential_increment = 0;
};

struct UnchangedContext {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Context<Real> operator()(Context<Real> c) const { return c; }
};

enum class Status : std::uint8_t {
    Success, InvalidOwner, InvalidParameters, InvalidThermodynamics,
    UnsupportedCondition, RegimeMismatch, Backflow, NoSubsonicSolution
};

/// @brief Failed evaluations carry a diagnostic and must not be used as a face state.
template<class S>
struct Evaluation {
    S state{};
    Status status = Status::Success;
    KOKKOS_INLINE_FUNCTION bool valid() const { return status == Status::Success; }
};

[[nodiscard]] inline const char* to_string(Status status) noexcept
{
    switch (status) {
        case Status::Success: return "success";
        case Status::InvalidOwner: return "invalid owner state";
        case Status::InvalidParameters: return "invalid boundary parameters";
        case Status::InvalidThermodynamics: return "invalid boundary thermodynamics";
        case Status::UnsupportedCondition: return "condition unsupported by the model";
        case Status::RegimeMismatch: return "normal Mach regime mismatch";
        case Status::Backflow: return "backflow requires an explicit inflow condition";
        case Status::NoSubsonicSolution: return "no compatible subsonic stagnation inlet state";
    }
    return "unknown boundary status";
}

} // namespace krona::boundary::physical
