#pragma once

#include <krona/mesh/device/detail/validation/common.hpp>

namespace krona::mesh::device::validation {

/// @brief Counts local cell IDs outside [first, end).
/// @tparam Map Kokkos view containing the send or receive IDs.
template <class Map>
struct ValidateIndices {
    Map map;
    LocalIndex first, end;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, std::size_t& bad) const {
        const auto id = map(i);
        if (id < first || id >= end)
            ++bad;
    }
};

/// @brief Checks that the host halo plan describes the local send and receive maps.
/// @tparam Space Kokkos memory space of the mesh view.
/// @tparam Plan Host object with nb_ranks, send_offsets and recv_offsets, such as host::HaloPlanView.
/// @param mesh Mesh view with MeshData::Halo attached.
/// @param plan Peer ranks and offsets into the two cell maps.
/// @param rank This rank in the communicator used for the halo exchange.
/// @param comm_size Number of ranks in that communicator.
/// @return Whether peer ranks and offsets are valid for this rank. Only map lengths are read.
template <class Space, class Plan>
[[nodiscard]] bool valid_halo_plan(MeshView<Space> mesh, const Plan& plan, int rank, int comm_size) {
    if (!mesh.has(MeshData::Halo) || rank < 0 || rank >= comm_size)
        return false;
    const auto peers = plan.nb_ranks.size();
    if (peers == std::numeric_limits<std::size_t>::max() || plan.send_offsets.size() != peers + 1 ||
        plan.recv_offsets.size() != peers + 1)
        return false;

    if (plan.send_offsets.front() != 0 || plan.recv_offsets.front() != 0 ||
        plan.send_offsets.back() != mesh.halo.send_owned_local.extent(0) ||
        plan.recv_offsets.back() != mesh.halo.recv_ghost_local.extent(0))
        return false;

    int previous = -1;
    for (std::size_t p = 0; p < peers; ++p) {
        const int peer = plan.nb_ranks[p];
        if (peer <= previous || peer >= comm_size || peer == rank ||
            plan.send_offsets[p] > plan.send_offsets[p + 1] ||
            plan.recv_offsets[p] > plan.recv_offsets[p + 1])
            return false;
        previous = peer;
    }
    return true;
}

/// @brief Checks that each ghost cell appears at most once in the receive map.
/// @tparam Exec Kokkos execution space used to run the check.
/// @tparam Space Kokkos memory space of the mesh view. Exec must be able to access it.
/// @param exec Instance used to initialize the counters and run the check. Finish writes
/// to the receive map on any other instance first.
/// @param mesh Mesh view with MeshData::Halo attached.
/// @return Number of receive entries outside the ghost range or beyond the first entry
/// for their destination. Returns one if counts, availability or the receive descriptor are invalid.
/// Uses one temporary counter per ghost and waits for the result before returning.
template <class Exec, class Space>
[[nodiscard]] std::size_t duplicate_halo_receives(const Exec& exec, MeshView<Space> mesh) {
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    if (!detail::valid_sizes(mesh.sizes) || !mesh.has(MeshData::Halo) ||
        (mesh.halo.recv_ghost_local.extent(0) != 0 && mesh.halo.recv_ghost_local.data() == nullptr))
        return 1;
    if (mesh.halo.recv_ghost_local.extent(0) == 0)
        return 0;

    using Counts = Kokkos::View<std::size_t*, Kokkos::Device<Exec, typename Exec::memory_space>>;
    Counts counts(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "mesh/halo_destination_counts"),
                  static_cast<std::size_t>(mesh.sizes.n_cells - mesh.sizes.n_own));
    // Keep temporary memory alive even if submission fails.
    try {
        Kokkos::deep_copy(exec, counts, std::size_t{0});
        const auto map = mesh.halo.recv_ghost_local;
        const auto own = mesh.sizes.n_own, end = mesh.sizes.n_cells;
        return detail::reduce(
            "mesh/validate_unique_halo", exec, map.extent(0), KOKKOS_LAMBDA(std::size_t i, std::size_t& bad) {
                const auto id = map(i);
                if (id < own || id >= end) {
                    ++bad;
                    return;
                }

                // Only the first entry for this ghost sees a zero counter.
                if (Kokkos::atomic_fetch_add(&counts(static_cast<std::size_t>(id - own)), std::size_t{1}) !=
                    0)
                    ++bad;
            });
    } catch (...) {
        exec.fence("mesh/validate_unique_halo_failed");
        throw;
    }
}

} // namespace krona::mesh::device::validation
