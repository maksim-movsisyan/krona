#pragma once

namespace krona::transport::scalar::diffusion {

/// @brief TPFA only; disabling this check does not add a nonorthogonal correction.
struct DiffusionOptions {
    bool require_orthogonal = true;
    double orthogonality_tolerance = 1.0e-6;
};

/// @brief Integrated physical balance: outward diffusion flux minus volume source.
struct BalanceResult {
    double source_integral = 0;
    double outward_boundary_flux = 0;
    double defect = 0;
};

} // namespace krona::transport::scalar::diffusion
