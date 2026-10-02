#pragma once

#include "krona/mesh/raw_mesh.hpp"
#include "krona/mpi/communicator.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace krona::io::cgns
{

enum class UnsupportedBoundary
{
    reject,
    ignore
};

struct ReaderOptions
{
    int base_index = 1;
    int zone_index = 1;
    std::size_t batch_elements = 65536;
    UnsupportedBoundary unsupported_boundary = UnsupportedBoundary::reject;
};

struct SourceMetadata
{
    float version = 0;
    std::string base_name;
    std::string zone_name;
    std::vector<std::string> ignored_boundary_conditions;
};

struct ReadResult
{
    mesh::RawMesh mesh;
    SourceMetadata source;
};

// Collective, blocking, host-memory import of one unstructured zone.
// All participants must pass identical path/options on the MPI main thread.
// Linear 2D/3D cells, uniform/MIXED sections; element-based BCs are optional.
// Errors abort comm.abort_scope(); no extrusion, renumbering or coordinate gather.
[[nodiscard]] ReadResult read_parallel(const std::string &path, mpi::CommView comm,
                                       const ReaderOptions &options = {});

} // namespace krona::io::cgns
