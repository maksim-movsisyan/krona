#pragma once

#include <krona/mesh/localmesh.hpp>

#include <cstdint>
#include <filesystem>
#include <span>

namespace krona::mesh {

/// @brief Writes an editable TOML BC template on rank 0 using already reduced patch counts.
/// Empty paths and other ranks are no-ops; a requested output that cannot be written aborts.
void generate_bc_template_config(const MeshPart& mesh, const std::filesystem::path& output,
                                 std::span<const std::uint64_t> global_patch_faces);

} // namespace krona::mesh
