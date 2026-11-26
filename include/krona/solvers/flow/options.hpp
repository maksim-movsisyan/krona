#pragma once

#include <krona/linalg/solver.hpp>
#include <krona/transport/compressible/config.hpp>
#include <krona/transport/euler/status.hpp>

#include <optional>

namespace krona::solvers::flow {

/// @brief Numerical controls for the steady compressible equations.
struct Options {
    bool viscous_enabled = false; ///< Adds viscous momentum and energy fluxes when enabled.
    double bulk_viscosity = 0;    ///< Bulk viscosity used by the viscous operator.
    transport::compressible::Flux flux = transport::compressible::Flux::Hllc;
    transport::compressible::ReconstructionOptions reconstruction;
    numerics::gradient::Options gradient;
    transport::compressible::PseudoTimeOptions pseudo_time;
    linalg::SolverOptions linear;
    int field_tag = 81;    ///< MPI tag for exchanging primitive values. Peers must agree.
    int gradient_tag = 82; ///< MPI tag for exchanging gradients and limiter factors. Peers must agree.

    /// @brief Reports whether convection reconstruction or viscous transport requires gradients.
    [[nodiscard]] bool needs_gradients() const noexcept {
        return viscous_enabled || reconstruction.needs_gradients();
    }
};

/// @brief Reports the residual of the input state and the work done to update it.
struct IterationResult {
    double residual_norm = 0;               ///< Global Euclidean norm before the update.
    std::optional<int> linear_iterations;   ///< Krylov iterations when a state update was performed.
    transport::euler::AssemblyReport euler; ///< Local reconstruction and Riemann fallback counts.
};

/// @brief Reports why the steady iteration stopped and the residual at the returned state.
struct SolveResult {
    int updates = 0;
    bool converged = false;
    double initial_residual = 0;
    double final_residual = 0;
};

} // namespace krona::solvers::flow
