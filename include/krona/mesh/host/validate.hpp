#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <cstdint>
#include <vector>

namespace krona::mesh::host {

/// @brief Replicated final counts, load extrema and labelled-boundary flux areas.
struct MeshStatistics {
    std::uint64_t owned_cells = 0, ghost_copies = 0;
    std::uint64_t interior_faces = 0, cut_faces = 0, boundary_faces = 0, labelled_faces = 0;
    std::uint64_t min_owned = 0, max_owned = 0, min_ghosts = 0, max_ghosts = 0;
    double total_volume = 0;
    std::vector<std::uint64_t> patch_face_counts;
    std::vector<double> patch_flux_areas;
};

/// @brief Checks a grouped, geometric MeshPart locally and aborts on its first invalid invariant.
/// Checks local halo layouts; remote message matching and geometric winding come from earlier stages.
void validate_meshpart(const MeshPart& mesh);

/// @brief Collects final statistics and checks global conservation with three packed reductions.
/// Call collectively after local validation; patch metadata must match on every rank.
[[nodiscard]] MeshStatistics collect_mesh_statistics(const MeshPart& mesh, mpi::CommView comm);

} // namespace krona::mesh::host
