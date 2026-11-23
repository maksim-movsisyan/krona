#pragma once

#include <krona/config.hpp>

namespace krona::transport {

/// @brief Half-open interval of local face indices.
/// Inner-face calls accept faces between two owned cells and faces at MPI interfaces.
struct FaceRange {
    LocalIndex begin = 0; ///< First face to include.
    LocalIndex end = -1;  ///< First face to exclude. Use -1 to go through the last inner face.
};

} // namespace krona::transport
