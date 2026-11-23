#pragma once

#include "initial_field.hpp"

#include <krona/boundary/physical/fields.hpp>
#include <krona/flow/conservative.hpp>
#include <krona/numerics/riemann/eos.hpp>
#include <krona/mesh/device/aux_view.hpp>

#include <type_traits>

namespace krona::apps::euler_steady {

namespace ri = numerics::riemann;
namespace ph = boundary::physical;

using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = Exec::memory_space;
using CellPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
using MeshView = mesh::device::MeshView<Space>;
using Gas = thermodynamics::IdealGas<Real>;
using Conserved = Kokkos::View<flow::Conservative<Real>*, Space>;

/// @brief Checks that positive p and T will survive storing the primitive field.
template<class Views>
KOKKOS_INLINE_FUNCTION bool storable_primitive(const Views& views, const ph::Primitive<Real, true>& state) {
    return views.can_store(state) && static_cast<StorageReal>(state.pressure) > StorageReal{0}
           && static_cast<StorageReal>(state.temperature) > StorageReal{0};
}

/// @brief Builds the owned conservative state and its primitive mirror from cell-centred data.
/// Views is the physical boundary PrimitiveViews adapter over the five components of q.
template<class Views>
void initialize_state(const Exec& exec, MeshView mesh, Conserved conserved, Views primitive,
                      Gas gas, Case problem, ph::Primitive<Real, true> reference) {
    const auto geometry = mesh.geometry;
    std::size_t invalid = 0;
    Kokkos::parallel_reduce("euler_steady/initialize", CellPolicy(exec, 0, conserved.extent(0)),
        KOKKOS_LAMBDA(std::size_t cell, std::size_t& bad) {
            const auto state = initial_field(problem, reference, static_cast<Real>(geometry.cell_centroid_x(cell)),
                static_cast<Real>(geometry.cell_centroid_y(cell)), static_cast<Real>(geometry.cell_centroid_z(cell)));
            if (!storable_primitive(primitive, state)) {
                ++bad;
                return;
            }

            const auto flux_state = ri::from_pT(gas, state.pressure, state.temperature, state.velocity);
            const auto value = ri::conservative(flux_state);
            if (!(value.density > Real{0}) || !Kokkos::isfinite(value.density) || !math::isfinite(value.momentum)
                || !(value.total_energy_density > Real{0}) || !Kokkos::isfinite(value.total_energy_density)) {
                ++bad;
                return;
            }
            conserved(cell) = value;
            primitive.write(cell, state);
        }, invalid);
    mpi::require(invalid == 0, "Steady Euler: invalid initial state");
}

/// @brief Adds V/dtau to each component of the owned diagonal block.
/// The target comes from transport::make_block_assembly_view. The face acoustic sum gives V/dtau directly.
template<class Target, class Views>
void add_pseudo_time_diagonal(const Exec& exec, MeshView mesh, mesh::device::AuxConnectivityView<Space> connectivity,
                              Target target, Views primitive, Gas gas, Real cfl) {
    const auto geometry = mesh.geometry;
    const auto blocks = target.blocks;
    const auto diagonal = target.entries.diagonal;
    std::size_t invalid = 0;
    Kokkos::parallel_reduce("euler_steady/pseudo_time", CellPolicy(exec, 0, static_cast<std::size_t>(mesh.sizes.n_own)),
        KOKKOS_LAMBDA(std::size_t cell, std::size_t& bad) {
            const auto state = primitive.template read<Real>(cell);
            const Real sound = Kokkos::sqrt(gas.gamma * gas.gas_constant * state.temperature);
            Real acoustic_sum = 0;
            for (auto k = connectivity.cell_faces_offsets(cell); k < connectivity.cell_faces_offsets(cell + 1); ++k) {
                const std::size_t face = static_cast<std::size_t>(connectivity.cell_faces(k));
                const ri::Vector<Real> normal{static_cast<Real>(geometry.face_normal_x(face)),
                                              static_cast<Real>(geometry.face_normal_y(face)),
                                              static_cast<Real>(geometry.face_normal_z(face))};
                acoustic_sum += static_cast<Real>(geometry.face_area(face)) *
                    (Kokkos::abs(state.velocity.dot(normal)) + sound);
            }

            using Sum = std::common_type_t<Real, SolverReal>;
            constexpr Sum limit = static_cast<Sum>(Kokkos::finite_max_v<SolverReal>);
            const Sum shift = static_cast<Sum>(acoustic_sum) / static_cast<Sum>(cfl);
            if (!(shift > Sum{0}) || !(shift <= limit) || !(static_cast<SolverReal>(shift) > SolverReal{0})) {
                ++bad;
                return;
            }
            const auto entry = diagonal(cell);
            for (std::size_t component = 0; component < 5; ++component) {
                const Sum next = static_cast<Sum>(blocks(entry, component, component)) + shift;
                if (!Kokkos::isfinite(next) || Kokkos::abs(next) > limit) {
                    ++bad;
                    return;
                }
                blocks(entry, component, component) = static_cast<SolverReal>(next);
            }
        }, invalid);
    mpi::require(invalid == 0, "Steady Euler: invalid pseudo-time diagonal");
}

/// @brief Applies the cell-major Tpetra increment in conservative variables and refreshes q.
/// With mixed precision, conservative state uses ComputeReal and q uses StorageReal.
template<class Increment, class Views>
void update_state(const Exec& exec, Conserved conserved, Increment increment, Views primitive, Gas gas, Real relaxation) {
    std::size_t invalid = 0;
    Kokkos::parallel_reduce("euler_steady/update", CellPolicy(exec, 0, conserved.extent(0)),
        KOKKOS_LAMBDA(std::size_t cell, std::size_t& bad) {
            const auto first = 5 * cell;
            const flow::Conservative<Real> delta{static_cast<Real>(increment(first, 0)),
                {static_cast<Real>(increment(first + 1, 0)), static_cast<Real>(increment(first + 2, 0)),
                 static_cast<Real>(increment(first + 3, 0))}, static_cast<Real>(increment(first + 4, 0))};
            const auto next = conserved(cell) + delta * relaxation;
            const auto state = ri::from_conservative(gas, next);
            const ph::Primitive<Real, true> q{state.pressure, state.velocity, state.internal_energy / gas.cv()};
            if (!storable_primitive(primitive, q)) {
                ++bad;
                return;
            }
            conserved(cell) = next;
            primitive.write(cell, q);
        }, invalid);
    mpi::require(invalid == 0, "Steady Euler: update produced a nonphysical or unrepresentable state");
}

} // namespace krona::apps::euler_steady
