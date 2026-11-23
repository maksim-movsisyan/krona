#pragma once

#include <krona/execution/types.hpp>
#include <krona/transport/detail/face_boundary.hpp>
#include <krona/transport/face_range.hpp>
#include <cmath>
#include <type_traits>

namespace krona::transport::viscous::detail {

template<class Exec, class Blocks, class Entries, class Sizes>
void validate_target(const Blocks& blocks, const Entries& entries, const Sizes& sizes) {
    using Value = typename Blocks::value_type;
    static_assert(Blocks::rank == 3 && std::is_floating_point_v<Value> && !std::is_const_v<Value>);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename Blocks::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<Exec, typename Entries::memory_space>::accessible);
    mpi::require(blocks.extent(1) == 5 && blocks.extent(2) == 5 &&
                 entries.diagonal.extent(0) == static_cast<std::size_t>(sizes.n_own) &&
                 entries.owner_entries.extent(0) == static_cast<std::size_t>(sizes.n_inner_faces) &&
                 entries.neighbour_entries.extent(0) == static_cast<std::size_t>(sizes.n_internal_faces),
                 "Viscous: Jacobian target does not match the mesh");
}

inline FaceRange inner_range(FaceRange faces, LocalIndex count) {
    if (faces.end == -1) faces.end = count;
    mpi::require(faces.begin >= 0 && faces.end >= faces.begin && faces.end <= count,
                 "Viscous: invalid inner face range");
    return faces;
}

template<class Plan, class Boundary, class R>
void validate_boundary(const Plan& plan, const Boundary& boundary, R time) {
    static_assert(Boundary::model_type::compressible && Boundary::model_type::thermal);
    mpi::require(boundary.execution_space() == plan.execution_space_instance(),
                 "Viscous: boundary execution instance differs from the operator");
    mpi::require(std::isfinite(time), "Viscous: time must be finite");
    transport::detail::validate_face_boundary(plan, boundary);
}

} // namespace krona::transport::viscous::detail
