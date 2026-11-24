#pragma once

#include <krona/linalg/solver.hpp>
#include <krona/transport/scalar/config.hpp>
#include <krona/transport/scalar/diffusion/types.hpp>
#include <vector>

namespace krona::solvers::scalar {

/// @brief Numerical settings for a single scalar equation.
struct Options {
    bool convection = true;                                             ///< Assemble the convection operator.
    bool diffusion = true;                                              ///< Assemble the diffusion operator.
    bool nonorthogonal_correction = false;                              ///< Apply the explicit diffusion correction.
    transport::scalar::diffusion::DiffusionOptions diffusion_geometry;
    numerics::gradient::Options gradient;
    transport::scalar::ReconstructionOptions reconstruction;
    transport::scalar::TimeScheme time_scheme = transport::scalar::TimeScheme::BackwardEuler;
    double time_step = 0.01;                                            ///< Fixed transient step size.
    transport::scalar::IterationOptions outer;
    linalg::SolverOptions linear;
    int field_tag = 71;                                                 ///< MPI tag for exchanging the unknown.
    int gradient_tag = 72;                                              ///< MPI tag for exchanging gradients and limiter factors.

    /// @brief Reports whether the convection operator uses a high-order face profile.
    [[nodiscard]] bool reconstructs() const noexcept {
        return convection && reconstruction.method != transport::scalar::ReconstructionMethod::Upwind;
    }

    /// @brief Reports whether the solver needs gradients and physical boundary samples.
    [[nodiscard]] bool needs_gradients() const noexcept {
        return reconstructs() || nonorthogonal_correction;
    }
};

/// @brief Result of one steady solve or accepted transient step.
template<class Real>
struct StepResult {
    Real time = 0;                      ///< Physical time at which the coefficients were evaluated.
    std::vector<int> linear_iterations; ///< Linear iteration count for each outer update.
};

} // namespace krona::solvers::scalar
