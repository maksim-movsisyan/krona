#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>

namespace krona::boundary {

/// @brief Exterior-face geometry; unit normal points out of the owned cell.
template<class Real>
struct FaceContext {
    LocalIndex face = 0, owner = 0;
    std::size_t boundary_index = 0, extended_index = 0;
    Real x = 0, y = 0, z = 0;
    Real nx = 0, ny = 0, nz = 0;
    Real distance = 0, area = 0, time = 0;
    Real tx = 0, ty = 0, tz = 0;
};

} // namespace krona::boundary
