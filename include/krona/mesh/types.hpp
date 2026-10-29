#pragma once

#include <krona/config.hpp>
#include <cstdint>

namespace krona::mesh {

using PatchId = std::int32_t;
inline constexpr PatchId invalid_patch_id = -1;
inline constexpr LocalIndex invalid_local_index = -1;

} // namespace krona::mesh
