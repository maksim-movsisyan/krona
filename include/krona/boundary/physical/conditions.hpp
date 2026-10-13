#pragma once

#include <krona/boundary/physical/state.hpp>

namespace krona::boundary::physical {

enum class ThermalKind { Extrapolated, FixedTemperature };

/// @brief Static temperature data; Extrapolated is first-order zero-normal-gradient closure.
template<class Real = ComputeReal>
struct Temperature {
    ThermalKind kind = ThermalKind::Extrapolated;
    Real value = 0;
    template<class S>
    KOKKOS_INLINE_FUNCTION Real face_value(const S& owner, const Context<Real>& c) const
    {
        if constexpr (S::has_temperature)
            return kind == ThermalKind::FixedTemperature ? value : owner.temperature + c.temperature_tangential_increment;
        else return Real{0};
    }
};

namespace detail {
template<class Closure>
KOKKOS_INLINE_FUNCTION bool compatible(Temperature<typename Closure::real_type> t)
{ return Closure::has_temperature || t.kind == ThermalKind::Extrapolated; }
}

/// @brief Copies the owner primitive state to the face; does not imply a nonreflecting outlet.
struct Extrapolate {
    template<class Real, class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>&, const typename Closure::state_type& owner, const Closure& closure) const
    { return {owner, closure.valid(owner) ? Status::Success : Status::InvalidOwner}; }
};

/// @brief Prescribes velocity and optional static temperature; model closes the pressure.
template<class Real = ComputeReal>
struct VelocityInlet {
    Vector3<Real> velocity{};
    Temperature<Real> temperature{};
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if (!detail::compatible<Closure>(temperature)) return {{}, Status::UnsupportedCondition};
        return closure.velocity_inlet(owner, velocity, temperature.face_value(owner, c), c.normal);
    }
};

/// @brief Prescribes stagnation pressure/temperature and unit inflow direction.
template<class Real = ComputeReal>
struct TotalInlet {
    Real pressure = 0, temperature = 0;
    Vector3<Real> direction{};
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if constexpr (Closure::compressible)
            return closure.total_inlet(owner, pressure, temperature, direction, c.normal);
        else return {{}, Status::UnsupportedCondition};
    }
};

/// @brief Prescribes the complete static inflow state; requires supersonic normal inflow.
template<class Real = ComputeReal>
struct SupersonicInlet {
    Real pressure = 0, temperature = 0;
    Vector3<Real> velocity{};
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type&, const Closure& closure) const
    {
        if constexpr (Closure::compressible) {
            auto result = closure.make_state(pressure, temperature, velocity);
            if (result.valid() && -velocity.dot(c.normal) <= closure.sound_speed(result.state))
                result.status = Status::RegimeMismatch;
            return result;
        } else return {{}, Status::UnsupportedCondition};
    }
};

/// @brief Static-pressure outlet; the model determines acoustic closure and supersonic behavior.
template<class Real = ComputeReal>
struct PressureOutlet {
    Real pressure = 0;
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    { return closure.pressure_outlet(owner, pressure, c.normal); }
};

/// @brief Pressure outlet with explicit velocity/temperature data if the outlet closure detects backflow.
template<class Real = ComputeReal>
struct InletOutlet {
    Real pressure = 0;
    Vector3<Real> backflow_velocity{};
    Temperature<Real> backflow_temperature{};
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if (!detail::compatible<Closure>(backflow_temperature)) return {{}, Status::UnsupportedCondition};
        auto result = closure.pressure_outlet(owner, pressure, c.normal);
        if (result.status != Status::Backflow) return result;
        if (backflow_velocity.dot(c.normal) > Real{0}) return {{}, Status::InvalidParameters};
        // This branch imposes the specified static inflow state, not a characteristic outlet.
        result = closure.make_state(pressure, backflow_temperature.face_value(owner, c), backflow_velocity);
        if constexpr (Closure::compressible)
            if (result.valid() && -backflow_velocity.dot(c.normal) >= closure.sound_speed(result.state))
                result.status = Status::RegimeMismatch;
        return result;
    }
};

/// @brief Extrapolates all primitives only when the owner has supersonic normal outflow.
struct SupersonicOutlet {
    template<class Real, class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if constexpr (Closure::compressible) {
            if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
            return {owner, owner.velocity.dot(c.normal) > closure.sound_speed(owner)
                ? Status::Success : Status::RegimeMismatch};
        } else return {{}, Status::UnsupportedCondition};
    }
};

/// @brief Prescribes face velocity; pressure is extrapolated and density follows the closure.
template<class Real = ComputeReal>
struct AdiabaticWall {
    Vector3<Real> velocity{};
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        return closure.wall_state(owner, velocity, Temperature<Real>{}.face_value(owner, c));
    }
};

/// @brief Prescribes no-slip velocity and a fixed static wall temperature.
template<class Real = ComputeReal>
struct IsothermalWall {
    Vector3<Real> velocity{};
    Real temperature = 0;
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>&, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if constexpr (!Closure::has_temperature) return {{}, Status::UnsupportedCondition};
        else {
            if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
            return closure.wall_state(owner, velocity, temperature);
        }
    }
};

/// @brief Removes relative normal velocity at the face; the wall geometry is stationary.
template<class Real = ComputeReal>
struct SlipWall {
    Vector3<Real> velocity{};
    Temperature<Real> temperature{};
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if (!detail::compatible<Closure>(temperature)) return {{}, Status::UnsupportedCondition};
        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        const auto relative = owner.velocity - velocity;
        const auto face_velocity = owner.velocity - c.normal * relative.dot(c.normal);
        return closure.wall_state(owner, face_velocity, temperature.face_value(owner, c));
    }
};

/// @brief Stationary symmetry plane; thermodynamic completion supports the thermal correction.
struct Symmetry {
    template<class Real, class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type>
    operator()(const Context<Real>& c, const typename Closure::state_type& owner, const Closure& closure) const
    {
        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        const auto velocity = owner.velocity - c.normal * owner.velocity.dot(c.normal);
        Real temperature = 0;
        if constexpr (Closure::has_temperature)
            temperature = owner.temperature + c.temperature_tangential_increment;
        return closure.wall_state(owner, velocity, temperature);
    }
};

} // namespace krona::boundary::physical
