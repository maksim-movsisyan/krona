#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <filesystem>

namespace krona::io::solver_mesh {

/// @brief Reads this rank's solver mesh collectively and validates its local structure.
/// Requires the saved MPI size/rank numbering and geometry precision; index widening is supported.
[[nodiscard]] mesh::host::MeshPart import_mesh_hdf5(const std::filesystem::path& path,
                                            mpi::CommView comm);

} // namespace krona::io::solver_mesh