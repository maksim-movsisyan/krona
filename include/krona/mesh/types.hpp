#pragma once

#include <krona/config.hpp>
#include <cstdint>

namespace krona::mesh { 

/// @brief Identifies a source patch independently of cell and node index widths.
using PatchId = std::int32_t;
/// @brief Marks an exterior face without a source patch label.
inline constexpr PatchId invalid_patch_id = -1;
/// @brief Marks an absent local neighbour or lookup result.
inline constexpr LocalIndex invalid_local_index = -1;

} // namespace krona::mesh
