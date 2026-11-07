#pragma once

#include <krona/mesh/host/mesh_part.hpp>

#include <span>

namespace krona::mesh::host {

/// @brief Selects the owned-cell ordering; NONE still groups and sorts faces.
enum class ReorderMethod : int { NONE = 0, HILBERT_SFC = 1, RCM = 2 };

/// @brief Returns the stable CLI/diagnostic name of an ordering method.
[[nodiscard]] inline const char* reorder_method_to_string(ReorderMethod method) noexcept
{
    switch (method) {
        case ReorderMethod::NONE: return "NONE";
        case ReorderMethod::HILBERT_SFC: return "HILBERT_SFC";
        case ReorderMethod::RCM: return "RCM";
    }
    return "UNKNOWN";
}

/// @brief Reorders owned cells and groups faces locally, preserving ghost and halo order.
/// SFC needs cell centroids; existing metrics are moved with their corresponding entities.
void reorder_local_mesh(MeshPart& mesh, ReorderMethod method = ReorderMethod::NONE);

/// @brief Applies a caller-supplied owned new-to-old permutation, then groups faces.
/// The permutation has n_own entries and must not alias mesh storage; no MPI exchanges occur.
void reorder_local_mesh(MeshPart& mesh, std::span<const LocalIndex> owned_new_to_old);

} // namespace krona::mesh::host
