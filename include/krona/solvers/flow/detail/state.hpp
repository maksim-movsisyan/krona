#pragma once

#include <krona/boundary/physical/fields.hpp>
#include <krona/flow/conservative.hpp>
#include <krona/thermodynamics/state.hpp>

namespace krona::solvers::flow::detail {

namespace ph = boundary::physical;

template<class Field>
auto primitive_views(Field field) {
    return ph::primitive_components(field.component(0), field.component(1), field.component(2),
                                    field.component(3), field.component(4));
}

/// @brief Checks that positive p and T survive conversion to the field's storage precision.
template<class Storage, class Views, class Real>
KOKKOS_INLINE_FUNCTION bool storable_primitive(const Views& views, const ph::Primitive<Real, true>& state) {
    return views.can_store(state) && static_cast<Storage>(state.pressure) > Storage{0}
           && static_cast<Storage>(state.temperature) > Storage{0};
}

/// @brief Initializes both representations of the state on owned cells from a primitive field sampler.
template<class Real, class Storage, class Exec, class Mesh, class Conserved, class Views, class EOS, class Initial>
void initialize_state(const Exec& exec, Mesh mesh, Conserved conserved, Views primitive, EOS eos, Initial initial) {
    using CellPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    const auto geometry = mesh.geometry;
    std::size_t invalid = 0;
    Kokkos::parallel_reduce("flow_solver/initialize", CellPolicy(exec, 0, conserved.extent(0)),
        KOKKOS_LAMBDA(std::size_t cell, std::size_t& bad) {
            const auto state = initial(static_cast<Real>(geometry.cell_centroid_x(cell)),
                                       static_cast<Real>(geometry.cell_centroid_y(cell)),
                                       static_cast<Real>(geometry.cell_centroid_z(cell)));
            thermodynamics::AcousticState<Real> thermo;
            if (!storable_primitive<Storage>(primitive, state) ||
                !eos.try_acoustic_from_pT(state.pressure, state.temperature, thermo)) {
                ++bad;
                return;
            }

            const auto value = krona::flow::to_conservative(thermo.density, state.velocity, thermo.internal_energy);
            if (!math::isfinite(value.momentum) || !(value.total_energy_density > Real{0}) ||
                !Kokkos::isfinite(value.total_energy_density)) {
                ++bad;
                return;
            }
            conserved(cell) = value;
            primitive.write(cell, state);
        }, invalid);
    mpi::require(invalid == 0, "Steady flow: nonphysical or unrepresentable initial state");
}

/// @brief Applies a cell-major conservative increment and publishes the resulting primitive variables.
template<class Real, class Storage, class Exec, class Conserved, class Increment, class Views, class EOS>
void update_state(const Exec& exec, Conserved conserved, Increment increment, Views primitive, EOS eos,
                  Real relaxation) {
    using CellPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    std::size_t invalid = 0;
    Kokkos::parallel_reduce("flow_solver/update", CellPolicy(exec, 0, conserved.extent(0)),
        KOKKOS_LAMBDA(std::size_t cell, std::size_t& bad) {
            const auto first = 5 * cell;
            const krona::flow::Conservative<Real> delta{static_cast<Real>(increment(first, 0)),
                {static_cast<Real>(increment(first + 1, 0)), static_cast<Real>(increment(first + 2, 0)),
                 static_cast<Real>(increment(first + 3, 0))}, static_cast<Real>(increment(first + 4, 0))};
            const auto next = conserved(cell) + delta * relaxation;
            krona::flow::InternalEnergyState<Real> state;
            thermodynamics::AcousticState<Real> thermo;
            if (!krona::flow::try_split_conservative(next, state) ||
                !eos.try_acoustic_from_rhoe(state.density, state.internal_energy, thermo)) {
                ++bad;
                return;
            }

            const ph::Primitive<Real, true> q{thermo.pressure, state.velocity, thermo.temperature};
            if (!storable_primitive<Storage>(primitive, q)) {
                ++bad;
                return;
            }
            conserved(cell) = next;
            primitive.write(cell, q);
        }, invalid);
    mpi::require(invalid == 0, "Steady flow: update produced a nonphysical or unrepresentable state");
}

} // namespace krona::solvers::flow::detail
