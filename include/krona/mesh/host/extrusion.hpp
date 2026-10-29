#pragma once

#include <krona/mesh/host/raw_mesh.hpp>
#include <krona/mpi/communicator.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <string>

namespace krona::mesh::host {

/// @brief Configures a single extrusion layer and bounded communication buffers.
struct ExtrusionOptions {
    double thickness_fraction = 0.1;
    double planarity_tolerance =
        64.0 * static_cast<double>(std::numeric_limits<GeometryReal>::epsilon());
    std::size_t batch_elements = 65536;
    std::string bottom_patch = "Z_MIN";
    std::string top_patch = "Z_MAX";
    std::array<double, 3> direction{}; // Zero selects a deterministic plane normal.
};

/// @brief Describes the displacement from the original layer to the new layer.
struct ExtrusionInfo {
    GeometryReal thickness = 0;
    std::array<double, 3> normal{}; // Unit vector; displacement = thickness * normal.
    int normal_axis = 2;          // Dominant component, retained for axis-aligned callers.
    int normal_sign = 1;
};

/// @brief Collectively extrudes a planar TRI/QUAD import into PRISM/HEXA cells.
/// Original node IDs remain unchanged; top IDs are shifted by the original node count.
[[nodiscard]] ExtrusionInfo extrude_one_layer(
    RawMesh& mesh, mpi::CommView comm, const ExtrusionOptions& options = {});
 
// Preserves cell ownership; balances output nodes and boundary ordinals again.
// h = thickness_fraction * max(bbox spans); original coordinates are preserved.
// Requires valid TRI/QUAD cells, BAR boundaries and the original block ownership.
// All ranks use matching metadata/options and obey MPI's thread rules.
// This converts an import snapshot; later migration/adaptation owns its own ID mapping.

} // namespace krona::mesh::host