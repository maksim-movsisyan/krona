#pragma once

#include <krona/mesh/extrusion.hpp>
#include <krona/mesh/raw_mesh.hpp>
#include <krona/mpi/communicator.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace krona::io::cgns {

/// @brief Policy for boundary point sets that cannot identify imported faces.
enum class UnsupportedBoundary { reject, ignore };

/// @brief Selects a zone, bounds temporary buffers and configures 2D extrusion.
struct ReaderOptions {
    int base_index = 1;
    int zone_index = 1;
    std::size_t batch_elements = 65536;
    bool collective_io = true;
    MPI_Info io_info = MPI_INFO_NULL;
    UnsupportedBoundary unsupported_boundary = UnsupportedBoundary::reject;
    mesh::ExtrusionOptions extrusion;
};

/// @brief Small replicated source metadata, separate from the resulting 3D mesh.
struct SourceMetadata {
    float version = 0;
    int cell_dimension = 0;
    int physical_dimension = 0;
    std::string base_name;
    std::string zone_name;
    std::vector<std::string> ignored_boundary_conditions;
};

/// @brief Returns an owned import snapshot and optional information about 2D extrusion.
struct ReadResult {
    mesh::RawMesh mesh;
    SourceMetadata source;
    std::optional<mesh::ExtrusionInfo> extrusion;
};

/// @brief Collectively reads one HDF5 unstructured zone and returns a 3D mesh.
/// Linear uniform/MIXED sections are supported; planar TRI/QUAD imports are extruded.
[[nodiscard]] ReadResult read_parallel(
    const std::string& path, mpi::CommView comm, const ReaderOptions& options = {});

// All ranks use identical path/options and call on the MPI main thread.
// Legacy uniform sections are allowed; implicit bulk conversion of variable-arity sections is rejected.
// Bulk arrays are read as owned slices; only small metadata is replicated.
// CGNS configuration is process-global: do not overlap other CGNS calls with this import.
// Missing element-based BCs are allowed; unassigned faces retain invalid_patch_id.

} // namespace krona::io::cgns