#pragma once

#include <krona/mesh/host/mesh_part.hpp>

#include <cstdint>
#include <filesystem>
#include <span>

namespace krona::io::boundary_config {

/// @brief Writes an editable TOML BC template on rank 0 using already reduced patch counts.
/// Empty paths and other ranks are no-ops; a requested output that cannot be written aborts.
void generate_bc_template_config(const mesh::host::MeshPart& mesh, const std::filesystem::path& output,
                                 std::span<const std::uint64_t> global_patch_faces);

} // namespace krona::io::boundary_config
