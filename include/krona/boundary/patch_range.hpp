#pragma once

#include <krona/config.hpp>
#include <krona/mesh/types.hpp>

namespace krona::boundary {

/// @brief Contiguous interval of boundary faces in the local mesh numbering.
struct PatchRange {
    mesh::PatchId patch_id = -1; ///< ID in the mesh patch catalog, or -1 for unlabelled faces.
    LocalIndex begin = 0;       ///< First local face ID in the interval.
    LocalIndex end = 0;         ///< One past the last local face ID.

    /// @brief Reports whether the patch has zero faces on this rank.
    [[nodiscard]] bool empty() const noexcept { return begin == end; }
};

} // namespace krona::boundary
