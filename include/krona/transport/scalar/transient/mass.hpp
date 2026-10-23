#pragma once

#include <krona/transport/scalar/detail/views.hpp>

#include <type_traits>

namespace krona::transport::scalar::transient {

/// @brief Integrated cell mass rho*V, not density. For a unit-density scalar use V.
template<class Real = ComputeReal>
struct UniformMass {
    Real value = 1;
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};

template<class View> 
struct CellMass {
    View values;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t i) const { return values(i); }
};

/// @brief Avoids an intermediate mass array. Density is a scalar or rank-one view.
template<class Volumes, class Density>
struct DensityVolumeMass {
    Volumes volumes;
    Density density;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t i) const
    {
        const auto rho = density_at(i);
        const auto volume = volumes(i);
        using Result = decltype(rho*volume);
        if (!Kokkos::isfinite(rho) || !(rho > 0) || !Kokkos::isfinite(volume) || !(volume > 0))
            return Kokkos::quiet_NaN_v<Result>;
        return rho*volume;
    }
private:
    KOKKOS_INLINE_FUNCTION auto density_at(std::size_t i) const
    {
        if constexpr (std::is_arithmetic_v<Density>) return density;
        else return density(i);
    }
};

/// @brief One accepted layer, with its own mass and scalar component. Borrows all arrays.
template<class Mass, class Field>
struct TimeLevel { Mass mass; Field field; };

namespace detail {
struct NoHistory {
    static constexpr std::size_t size = 0;
    template<class Real, std::size_t J, class Weights, class Flags>
    KOKKOS_INLINE_FUNCTION Real rhs(std::size_t, const Weights&, Flags) const { return Real{0}; }
};
template<class Level, class Tail>
struct History {
    static constexpr std::size_t size = Tail::size+1;
    Level level;
    Tail tail;
    template<class Real, std::size_t J, class Weights, class Flags>
    KOKKOS_INLINE_FUNCTION Real rhs(std::size_t i, const Weights& weights, Flags flags) const;
};
template<class Exec, class Mass>
void validate_mass(const Mass&, std::size_t) {} // Custom providers own their extent/access contract.
template<class Exec, class View>
void validate_mass(CellMass<View> mass, std::size_t count)
{ scalar::detail::validate_view<Exec>(mass.values, count); }
template<class Exec, class V, class D>
void validate_mass(DensityVolumeMass<V, D> mass, std::size_t count)
{
    scalar::detail::validate_view<Exec>(mass.volumes, count);
    if constexpr (!std::is_arithmetic_v<D>) scalar::detail::validate_view<Exec>(mass.density, count);
}
template<class Exec>
void validate_history(NoHistory, std::size_t) {}
template<class Exec, class L, class T>
void validate_history(const History<L, T>& history, std::size_t count)
{
    validate_mass<Exec>(history.level.mass, count);
    scalar::detail::validate_view<Exec>(history.level.field, count);
    validate_history<Exec>(history.tail, count);
}
} // namespace detail

/// @brief Newest accepted layer first. No snapshots, copies, allocation or history rotation.
inline auto time_history() { return detail::NoHistory{}; }
template<class Level, class... Rest>
auto time_history(Level level, Rest... rest)
{
    auto tail = time_history(rest...);
    return detail::History<Level, decltype(tail)>{level, tail};
}

} // namespace krona::transport::scalar::transient
