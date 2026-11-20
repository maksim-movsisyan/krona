#pragma once

#include <krona/config.hpp>
#include <krona/numerics/gradient/types.hpp>
#include <krona/mpi/communicator.hpp>

#include <filesystem>
#include <string>
#include <string_view>

namespace krona::transport::scalar {

/// @brief Selects how convection reconstructs the scalar at a face.
enum class ReconstructionMethod { Upwind, Linear, MusclKappa, Tvd };

/// @brief Chooses the slope-ratio limiter for TVD reconstruction.
enum class RatioLimiter { Minmod, Superbee, MonotonizedCentral, VanLeer, VanAlbada, Koren };

/// @brief Chooses whether and how to bound high-order face values using nearby cell values.
enum class CellLimiter { None, BarthJespersen, Venkatakrishnan };

/// @brief Selects steady assembly or the implicit derivative for a transient run.
enum class TimeScheme { Steady, BackwardEuler, Bdf2 };

/// @brief Options for reconstructing face values from the cell field.
struct ReconstructionOptions {
    ReconstructionMethod method = ReconstructionMethod::Upwind; ///< Reconstruction used for convection.
    RatioLimiter tvd_limiter = RatioLimiter::VanLeer;           ///< Slope-ratio limiter used by TVD.
    double kappa = 1.0/3.0;                                     ///< Blend parameter for the MUSCL kappa scheme.
    bool skew_correction = true;                                ///< Enable skew correction for MUSCL kappa or TVD.
    CellLimiter cell_limiter = CellLimiter::None;               ///< Limiter applied to the high-order cell reconstruction.
    double regularization = 0;                                  ///< Venkatakrishnan epsilon, in the units of the scalar.
};

/// @brief Controls successive outer updates of the field.
struct IterationOptions {
    int iterations = 1;    ///< Number of outer updates.
    double relaxation = 1; ///< Fraction of the proposed update applied each time, in (0, 1].
};

/// @brief Controls where and when the application writes field snapshots.
struct OutputOptions {
    std::filesystem::path directory; ///< Destination for VTU snapshots.
    int every = 0;                   ///< Accepted transient steps between snapshots. Zero writes only the final field.
};

/// @brief Settings passed to the application's linear solver.
struct LinearOptions {
    std::string method = "GMRES";              ///< Linear solver name.
    std::string preconditioner = "RELAXATION"; ///< Preconditioner name.
    double relative_tolerance = 1.e-8;         ///< Relative convergence tolerance.
    double absolute_tolerance = 0;             ///< Absolute convergence tolerance.
    int maximum_iterations = 500;              ///< Iteration limit for one solve.
    int restart_length = 50;                   ///< GMRES restart length.
    bool verify_residual = false;              ///< Request an explicit check of the final residual.
};

/// @brief Host settings for the scalar equation selected by [scalars.<field>].
struct Configuration {
    std::string field;                                  ///< Name of the scalar field being solved.
    double density = 1;                                 ///< Uniform density used for cell masses and face mass fluxes.
    double diffusion = 0.01;                            ///< Coefficient Gamma in the diffusive flux.
    double source = 0;                                  ///< Configured source per unit cell volume.
    bool convection_enabled = true;                     ///< Include convection in the equation.
    bool diffusion_enabled = true;                      ///< Include diffusion in the equation.
    bool gradient_enabled = true;                       ///< A gradient method is available when an operator needs it.
    numerics::gradient::Options gradient;               ///< Scheme and geometry options for cell gradients.
    ReconstructionOptions reconstruction;               ///< Convection face reconstruction and limiters.
    bool nonorthogonal_correction = false;              ///< Add the explicit correction for nonorthogonal diffusion geometry.
    bool require_orthogonal = true;                     ///< Require sufficiently orthogonal diffusion geometry during setup.
    double orthogonality_tolerance = 1.e-6;             ///< Allowed nonorthogonality when require_orthogonal is set.
    TimeScheme time_scheme = TimeScheme::BackwardEuler; ///< Time derivative formula, or steady mode.
    double time_step = 0.01;                            ///< Step length for a transient run.
    int steps = 1;                                      ///< Number of requested transient steps.
    IterationOptions outer;                             ///< Outer iteration and relaxation settings.
    LinearOptions linear;                               ///< Linear solver and preconditioner settings.
    OutputOptions output;                               ///< Snapshot destination and frequency.

    /// @brief Reports whether convection reconstruction or diffusion correction needs cell gradients.
    [[nodiscard]] bool needs_gradients() const noexcept
    { return nonorthogonal_correction || (convection_enabled && reconstruction.method != ReconstructionMethod::Upwind); }
};

/// @brief Parses the selected field only; other fields and application tables are untouched.
/// @param text Complete TOML input containing [scalars.<field>].
/// @param field Name of the scalar table to decode.
[[nodiscard]] Configuration parse_config(std::string_view text, std::string_view field);

/// @brief Reads on rank zero and parses the broadcast text on every rank.
/// @param file TOML file read on rank zero of comm.
/// @param field Name of the scalar table to decode.
/// @param comm Communicator shared by all callers.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, std::string_view field, mpi::CommView comm);

} // namespace krona::transport::scalar
