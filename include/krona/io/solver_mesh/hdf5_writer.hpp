#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <filesystem>
#include <span>

namespace krona::io::solver_mesh {

/// @brief Writes a validated solver mesh collectively to a shared, versioned HDF5 file.
/// An empty mapping means rank order; otherwise part2rank must be a replicated permutation.
void export_mesh_hdf5(const mesh::host::MeshPart& mesh, const std::filesystem::path& path,
                      mpi::CommView comm, std::span<const int> part2rank = {});

} // namespace krona::io::solver_mesh