#pragma once

#include <cassert>
#include <cstddef>
#include <span>

namespace krona::mesh::host {

/// @brief Read-only host metadata for MPI; offsets count cells, not scalar entries or bytes.
/// Peer p uses indices [offsets[p], offsets[p+1]) in the matching MeshView halo map.
/// Storage and CPU source arrays must outlive all users; do not capture this view in GPU kernels.
struct HaloPlanView { 
    std::span<const int> nb_ranks;
    std::span<const std::size_t> send_offsets;
    std::span<const std::size_t> recv_offsets;

    [[nodiscard]] std::size_t n_neighbors() const noexcept { return nb_ranks.size(); }
    [[nodiscard]] std::size_t send_count(std::size_t peer) const noexcept
    {
        assert(peer < nb_ranks.size());
        return send_offsets[peer + 1] - send_offsets[peer];
    }
    [[nodiscard]] std::size_t recv_count(std::size_t peer) const noexcept
    {
        assert(peer < nb_ranks.size());
        return recv_offsets[peer + 1] - recv_offsets[peer];
    }
};

} // namespace krona::mesh::host
