#pragma once

#include <krona/transport/face_range.hpp>

namespace krona::transport::scalar::diffusion {

/// @brief Geometry checks for a diffusion plan.
struct DiffusionOptions {
    bool require_orthogonal = true; ///< Require the cell connection to be nearly parallel to the face normal.
    double orthogonality_tolerance = 1.0e-6; ///< Maximum tangential separation divided by the normal separation.
};

/// @brief Integrated physical balance: outward diffusion flux minus volume source.
struct BalanceResult {
    double source_integral = 0;       ///< Source density integrated over all owned cells across ranks.
    double outward_boundary_flux = 0; ///< Net outward diffusion flux through physical boundaries.
    double defect = 0;                ///< Outward boundary flux minus the integrated source.
}; 

} // namespace krona::transport::scalar::diffusion
