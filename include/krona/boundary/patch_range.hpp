#pragma once

#include <krona/config.hpp>
#include <krona/mesh/types.hpp>

namespace krona::boundary {

/// @brief Absolute local face interval [begin,end); patch_id == -1 denotes unlabelled faces.
struct PatchRange {
    mesh::PatchId patch_id = -1;
    LocalIndex begin = 0, end = 0;
    [[nodiscard]] bool empty() const noexcept { return begin == end; }
};

} // namespace krona::boundary
