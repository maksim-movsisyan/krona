#pragma once

#include <krona/boundary/physical/conditions.hpp>
#include <krona/mpi/error.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

namespace krona::boundary::physical {

/// @brief Open boundaries with prescribed data and extrapolation of remaining primitives.
struct PrescribedOpenBoundary {
    static constexpr bool supports_total_inlet = false;
    template<class EOS> void validate(const EOS&) const {}

    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type> velocity_inlet(
        const Closure& closure, const typename Closure::state_type& owner,
        Vector3<typename Closure::real_type> velocity, typename Closure::real_type temperature,
        Vector3<typename Closure::real_type> normal) const
    {
        using Real = typename Closure::real_type;
        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        if (Kokkos::abs(owner.velocity.dot(normal)) >= closure.sound_speed(owner)) return {{}, Status::RegimeMismatch};
        if (velocity.dot(normal) > Real{0}) return {{}, Status::Backflow};
        auto result = closure.make_state(owner.pressure, temperature, velocity);
        if (result.valid() && -velocity.dot(normal) >= closure.sound_speed(result.state)) result.status = Status::RegimeMismatch;
        return result;
    }
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type> pressure_outlet(
        const Closure& closure, const typename Closure::state_type& owner,
        typename Closure::real_type pressure, Vector3<typename Closure::real_type> normal) const
    {
        using Real = typename Closure::real_type;
        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        const Real un = owner.velocity.dot(normal);
        if (un < Real{0}) return {{}, Status::Backflow};
        if (un >= closure.sound_speed(owner)) return {owner};
        auto result = closure.make_state(pressure, owner.temperature, owner.velocity);
        if (result.valid() && un >= closure.sound_speed(result.state)) result.status = Status::RegimeMismatch;
        return result;
    }
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type> total_inlet(
        const Closure&, const typename Closure::state_type&, typename Closure::real_type,
        typename Closure::real_type, Vector3<typename Closure::real_type>, Vector3<typename Closure::real_type>) const
    { return {{}, Status::UnsupportedCondition}; }
};

/// @brief Local perfect-gas characteristic closures; thermodynamics remains in the supplied EOS.
struct PerfectGasCharacteristics {
    static constexpr bool supports_total_inlet = true;
    template<class EOS> void validate(const EOS&) const
    {
        static_assert(EOS::calorically_perfect, "PerfectGasCharacteristics requires a calorically perfect gas");
    }
    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type> velocity_inlet(const Closure& closure, const typename Closure::state_type& owner,
        Vector3<typename Closure::real_type> velocity, typename Closure::real_type temperature, Vector3<typename Closure::real_type> normal) const
    {
        using Real = typename Closure::real_type;

        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        const Real a = closure.sound_speed(owner), un = owner.velocity.dot(normal), ub = velocity.dot(normal);
        if (Kokkos::abs(un) >= a) return {{}, Status::RegimeMismatch};
        if (ub > Real{0}) return {{}, Status::Backflow};
        auto result = closure.make_state(owner.pressure + owner.density * a * (un - ub), temperature, velocity);
        if (result.valid() && -ub >= closure.sound_speed(result.state)) result.status = Status::RegimeMismatch;
        return result;
    }

    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type> pressure_outlet(const Closure& closure, const typename Closure::state_type& owner,
        typename Closure::real_type pressure, Vector3<typename Closure::real_type> normal) const
    {
        using Real = typename Closure::real_type;
        const Real gamma = closure.eos.gamma;

        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        if (!Kokkos::isfinite(pressure) || !(pressure > Real{0})) return {{}, Status::InvalidParameters};
        const Real un = owner.velocity.dot(normal), a = closure.sound_speed(owner);
        if (un < Real{0}) return {{}, Status::Backflow};
        if (un >= a) return {owner};
        const Real temperature = owner.temperature * Kokkos::pow(pressure / owner.pressure,
                                                                  (gamma - Real{1}) / gamma);
        auto result = closure.make_state(pressure, temperature, owner.velocity);
        if (!result.valid()) return result;
        const Real ab = closure.sound_speed(result.state);
        const Real ub = un + Real{2} * (a - ab) / (gamma - Real{1});
        result.state.velocity = owner.velocity + normal * (ub - un);
        if (!result.state.velocity.finite()) result.status = Status::InvalidThermodynamics;
        else if (ub < Real{0}) result.status = Status::Backflow;
        else if (ub >= ab) result.status = Status::RegimeMismatch;
        return result;
    }

    template<class Closure>
    KOKKOS_INLINE_FUNCTION Evaluation<typename Closure::state_type> total_inlet(const Closure& closure, const typename Closure::state_type& owner,
        typename Closure::real_type total_pressure, typename Closure::real_type total_temperature, Vector3<typename Closure::real_type> direction, Vector3<typename Closure::real_type> normal) const
    {
        using Real = typename Closure::real_type;
        const Real gamma = closure.eos.gamma;

        if (!closure.valid(owner)) return {{}, Status::InvalidOwner};
        const Real mu = direction.dot(normal), un = owner.velocity.dot(normal), a = closure.sound_speed(owner);
        if (!Kokkos::isfinite(total_pressure) || !(total_pressure > Real{0}) ||
            !Kokkos::isfinite(total_temperature) || !(total_temperature > Real{0}) ||
            !direction.finite() || !(mu < Real{0})) return {{}, Status::InvalidParameters};
        const Real direction_norm = direction.dot(direction);
        if (Kokkos::abs(direction_norm - Real{1}) > Real{256} * static_cast<Real>(std::numeric_limits<Real>::epsilon()))
            return {{}, Status::InvalidParameters};
        if (Kokkos::abs(un) >= a) return {{}, Status::RegimeMismatch};
        const Real g = gamma - Real{1}, invariant = un + Real{2} * a / g;
        const Real h0 = closure.eos.cp() * total_temperature;
        const Real aa = Real{0.5} + Real{0.25} * g * mu * mu;
        const Real bb = -Real{0.5} * g * invariant * mu;
        const Real acoustic_energy = Real{0.25} * g * invariant * invariant;
        const Real raw_cc = acoustic_energy - h0;
        const Real roundoff = Real{32} * static_cast<Real>(std::numeric_limits<Real>::epsilon()) * (acoustic_energy + h0);
        if (raw_cc > roundoff) return {{}, Status::NoSubsonicSolution};
        const Real cc = raw_cc < Real{0} ? raw_cc : Real{0};
        const Real discriminant = bb * bb - Real{4} * aa * cc;
        if (!Kokkos::isfinite(discriminant) || discriminant < Real{0} || cc > Real{0})
            return {{}, Status::NoSubsonicSolution};
        const Real root = Kokkos::sqrt(discriminant);
        // Rationalized positive root avoids cancellation for weak inflow.
        const Real speed = bb >= Real{0} && bb + root > Real{0}
            ? -Real{2} * cc / (bb + root) : (-bb + root) / (Real{2} * aa);
        const Real temperature = total_temperature - speed * speed / (Real{2} * closure.eos.cp());
        if (!Kokkos::isfinite(speed) || speed < Real{0} || !(temperature > Real{0}))
            return {{}, Status::NoSubsonicSolution};
        const Real pressure = total_pressure * Kokkos::pow(temperature / total_temperature, gamma / g);
        auto result = closure.make_state(pressure, temperature, direction * speed);
        if (result.valid()) {
            const Real ab = closure.sound_speed(result.state), ub = result.state.velocity.dot(normal);
            if (-ub >= ab) result.status = Status::NoSubsonicSolution;
            const Real recovered = ub + Real{2} * ab / g;
            const Real scale = Kokkos::abs(invariant) + a;
            if (Kokkos::abs(recovered - invariant) > Real{512} * static_cast<Real>(std::numeric_limits<Real>::epsilon()) * scale)
                result.status = Status::NoSubsonicSolution;
        }
        return result;
    }
};

/// @brief Compressible boundary evaluator composed from an EOS and an open-boundary policy.
template<class EOS, class OpenBoundary = PrescribedOpenBoundary>
struct CompressibleClosure {
    using eos_type = EOS;
    using real_type = typename EOS::real_type;
    using Real = real_type;
    using state_type = State<Real, true>;
    static constexpr bool compressible = true, has_temperature = true;
    EOS eos{};
    OpenBoundary open{};

    void validate() const
    {
        if constexpr (requires { eos.valid_parameters(); })
            mpi::require(eos.valid_parameters(), "Physical boundary: invalid EOS parameters");
        open.validate(eos);
    }
    template<class C> void validate_condition(const C&) const {}
    void validate_condition(const TotalInlet<Real>&) const
    { mpi::require(OpenBoundary::supports_total_inlet, "Physical boundary: selected closure does not support total inlet"); }

    KOKKOS_INLINE_FUNCTION Real sound_speed(const state_type& q) const
    { return eos.sound_speed(q.density, q.temperature); }
    KOKKOS_INLINE_FUNCTION bool valid(const state_type& q) const
    {
        return q.velocity.finite() && Kokkos::isfinite(q.density) && q.density > Real{0} &&
               eos.valid(q.pressure, q.temperature) && Kokkos::isfinite(sound_speed(q)) && sound_speed(q) > Real{0};
    }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> make_state(Real pressure, Real temperature, Vector3<Real> velocity) const
    {
        if (!eos.valid(pressure, temperature) || !velocity.finite()) return {{}, Status::InvalidThermodynamics};
        state_type q{eos.density(pressure, temperature), pressure, velocity, temperature};
        return {q, valid(q) ? Status::Success : Status::InvalidThermodynamics};
    }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> wall_state(const state_type& owner, Vector3<Real> velocity, Real temperature) const
    { return make_state(owner.pressure, temperature, velocity); }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> velocity_inlet(const state_type& owner, Vector3<Real> velocity,
        Real temperature, Vector3<Real> normal) const
    { return open.velocity_inlet(*this, owner, velocity, temperature, normal); }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> pressure_outlet(const state_type& owner, Real pressure, Vector3<Real> normal) const
    { return open.pressure_outlet(*this, owner, pressure, normal); }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> total_inlet(const state_type& owner, Real pressure, Real temperature,
        Vector3<Real> direction, Vector3<Real> normal) const
    { return open.total_inlet(*this, owner, pressure, temperature, direction, normal); }
};

/// @brief Constant-density boundary evaluator, optionally with transported temperature.
template<class Real = ComputeReal, bool Thermal = false>
struct IncompressibleClosure {
    using real_type = Real;
    using state_type = State<Real, Thermal>;
    static constexpr bool compressible = false, has_temperature = Thermal;
    Real density = 1;

    void validate() const
    { mpi::require(std::isfinite(density) && density > Real{0}, "Physical boundary: density must be positive"); }
    template<class C>
    void validate_condition(const C& c) const
    {
        if constexpr (std::is_same_v<C, TotalInlet<Real>> || std::is_same_v<C, SupersonicInlet<Real>> ||
                      std::is_same_v<C, SupersonicOutlet>)
            mpi::require(false, "Physical boundary: condition requires compressible flow");
        if constexpr (std::is_same_v<C, IsothermalWall<Real>>)
            mpi::require(Thermal, "Physical boundary: this flow has no temperature");
        if constexpr (requires { c.temperature.kind; })
            mpi::require(Thermal || c.temperature.kind == ThermalKind::Extrapolated, "Physical boundary: this flow has no temperature");
        if constexpr (requires { c.backflow_temperature.kind; })
            mpi::require(Thermal || c.backflow_temperature.kind == ThermalKind::Extrapolated, "Physical boundary: this flow has no temperature");
    }
    KOKKOS_INLINE_FUNCTION bool valid(const state_type& q) const
    {
        bool result = q.velocity.finite() && Kokkos::isfinite(q.pressure) && Kokkos::isfinite(q.density) && q.density > Real{0};
        if constexpr (Thermal) result = result && Kokkos::isfinite(q.temperature) && q.temperature > Real{0};
        return result;
    }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> make_state(Real pressure, Real temperature, Vector3<Real> velocity) const
    {
        state_type q{density, pressure, velocity};
        if constexpr (Thermal) q.temperature = temperature;
        return {q, valid(q) ? Status::Success : Status::InvalidThermodynamics};
    }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> wall_state(const state_type& owner, Vector3<Real> velocity, Real temperature) const
    { return make_state(owner.pressure, temperature, velocity); }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> velocity_inlet(const state_type& owner, Vector3<Real> velocity,
        Real temperature, Vector3<Real> normal) const
    {
        if (!valid(owner)) return {{}, Status::InvalidOwner};
        if (velocity.dot(normal) > Real{0}) return {{}, Status::Backflow};
        return make_state(owner.pressure, temperature, velocity);
    }
    KOKKOS_INLINE_FUNCTION Evaluation<state_type> pressure_outlet(const state_type& owner, Real pressure, Vector3<Real> normal) const
    {
        if (!valid(owner)) return {{}, Status::InvalidOwner};
        if (owner.velocity.dot(normal) < Real{0}) return {{}, Status::Backflow};
        auto q = owner; q.pressure = pressure;
        return {q, valid(q) ? Status::Success : Status::InvalidThermodynamics};
    }
};

} // namespace krona::boundary::physical
