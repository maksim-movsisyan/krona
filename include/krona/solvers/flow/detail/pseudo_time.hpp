#pragma once

#include <krona/boundary/physical/state.hpp>
#include <krona/mesh/device/aux_view.hpp>
#include <krona/numerics/riemann/types.hpp>
#include <krona/thermodynamics/state.hpp>

#include <type_traits>

namespace krona::solvers::flow::detail {

namespace ph = boundary::physical;
namespace ri = numerics::riemann;

struct InviscidScale {
    template<class R>
    KOKKOS_INLINE_FUNCTION R operator()(ph::Primitive<R, true>, const thermodynamics::AcousticState<R>&) const {
        return R{0};
    }
};

/// @brief Estimates the largest molecular diffusivity in the conservative equations.
template<class Properties, class Real>
struct DiffusiveScale {
    Properties properties;
    Real cv;
    Real bulk;

    KOKKOS_INLINE_FUNCTION Real operator()(ph::Primitive<Real, true> state,
                                           const thermodynamics::AcousticState<Real>& thermo) const {
        const auto coefficients = properties.template evaluate<Real>(state);
        if (!coefficients.valid())
            return Kokkos::quiet_NaN_v<Real>;
        const Real momentum = (Real{4} / Real{3} * coefficients.dynamic_viscosity + bulk) / thermo.density;
        const Real thermal = coefficients.thermal_conductivity / (thermo.density * cv);
        return Kokkos::max(momentum, thermal);
    }
};

/// @brief Returns the positive projected distance to the opposite centre or the physical face.
template<class Real, class Mesh>
KOKKOS_INLINE_FUNCTION Real face_distance(Mesh mesh, std::size_t face, std::size_t cell) {
    const auto& g = mesh.geometry;
    const auto& c = mesh.connectivity;
    const auto owner = static_cast<std::size_t>(c.face_owner(face));
    const ri::Vector<GeometryReal> n{g.face_normal_x(face), g.face_normal_y(face), g.face_normal_z(face)};
    const ri::Vector<GeometryReal> xo{g.cell_centroid_x(cell), g.cell_centroid_y(cell), g.cell_centroid_z(cell)};
    if (face >= static_cast<std::size_t>(mesh.sizes.n_inner_faces)) {
        const ri::Vector<GeometryReal> xf{g.face_centroid_x(face), g.face_centroid_y(face), g.face_centroid_z(face)};
        return static_cast<Real>((xf - xo).dot(n));
    }
    const bool on_owner = cell == owner;
    const auto other = on_owner ? static_cast<std::size_t>(c.face_neigh(face)) : owner;
    const ri::Vector<GeometryReal> xn{g.cell_centroid_x(other), g.cell_centroid_y(other), g.cell_centroid_z(other)};
    return static_cast<Real>((xn - xo).dot(on_owner ? n : -n));
}

/// @brief Adds V/dtau to each component of the owned diagonal block.
/// Target comes from transport::make_block_assembly_view. The acoustic and molecular face sums give V/dtau directly.
template<class Exec, class Target, class Views, class EOS, class Real, class Diffusion = InviscidScale>
void add_pseudo_time_diagonal(const Exec& exec, mesh::device::MeshView<typename Exec::memory_space> mesh,
                             mesh::device::AuxConnectivityView<typename Exec::memory_space> connectivity,
                             Target target, Views primitive, EOS eos, Real cfl, Diffusion diffusion = {}) {
    using CellPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    const auto geometry = mesh.geometry;
    const auto blocks = target.blocks;
    const auto diagonal = target.entries.diagonal;
    std::size_t invalid = 0;
    Kokkos::parallel_reduce("flow_solver/pseudo_time", CellPolicy(exec, 0, static_cast<std::size_t>(mesh.sizes.n_own)),
        KOKKOS_LAMBDA(std::size_t cell, std::size_t& bad) {
            const auto state = primitive.template read<Real>(cell);
            thermodynamics::AcousticState<Real> thermo;
            if (!eos.try_acoustic_from_pT(state.pressure, state.temperature, thermo)) {
                ++bad;
                return;
            }
            Real spectral_sum = 0;
            const Real diffusivity = diffusion(state, thermo);
            if (!Kokkos::isfinite(diffusivity) || diffusivity < Real{0}) {
                ++bad;
                return;
            }
            for (auto k = connectivity.cell_faces_offsets(cell); k < connectivity.cell_faces_offsets(cell + 1); ++k) {
                const std::size_t face = static_cast<std::size_t>(connectivity.cell_faces(k));
                const ri::Vector<Real> normal{static_cast<Real>(geometry.face_normal_x(face)),
                                              static_cast<Real>(geometry.face_normal_y(face)),
                                              static_cast<Real>(geometry.face_normal_z(face))};
                Real speed = Kokkos::abs(state.velocity.dot(normal)) + thermo.sound_speed;
                if constexpr (!std::is_same_v<Diffusion, InviscidScale>) {
                    const Real distance = face_distance<Real>(mesh, face, cell);
                    if (!(distance > Real{0}) || !Kokkos::isfinite(distance)) {
                        ++bad;
                        return;
                    }
                    speed += Real{2} * diffusivity / distance;
                }
                spectral_sum += static_cast<Real>(geometry.face_area(face)) * speed;
            }

            using Sum = std::common_type_t<Real, SolverReal>;
            constexpr Sum limit = static_cast<Sum>(Kokkos::finite_max_v<SolverReal>);
            const Sum shift = static_cast<Sum>(spectral_sum) / static_cast<Sum>(cfl);
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
    mpi::require(invalid == 0, "Steady flow: invalid pseudo-time diagonal");
}

} // namespace krona::solvers::flow::detail
