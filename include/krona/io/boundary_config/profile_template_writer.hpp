#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <filesystem>

namespace krona::io::boundary_config {

/// @brief Collectively writes one editable CSV per named patch from a validated mesh with face geometry.
/// All ranks in the mesh communicator must call this function. Rank 0 creates the directory and
/// writes or replaces <patch-name>.csv. Empty patches get a header-only file.
void generate_patch_profile_templates(const mesh::host::MeshPart& mesh,
                                      const std::filesystem::path& directory, mpi::CommView comm);

} // namespace krona::io::boundary_config
