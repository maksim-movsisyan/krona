#pragma once
#include <krona/config.hpp>
#include <krona/numerics/gradient/types.hpp>
#include <krona/mpi/communicator.hpp>
#include <filesystem>
#include <string>
#include <string_view>

namespace krona::transport::scalar {
enum class ReconstructionMethod { Upwind, Linear, MusclKappa, Tvd };
enum class RatioLimiter { Minmod, Superbee, MonotonizedCentral, VanLeer, VanAlbada, Koren };
enum class CellLimiter { None, BarthJespersen, Venkatakrishnan };
enum class TimeScheme { Steady, BackwardEuler, Bdf2 };
struct ReconstructionOptions {
    ReconstructionMethod method = ReconstructionMethod::Upwind;
    RatioLimiter tvd_limiter = RatioLimiter::VanLeer;
    double kappa = 1.0/3.0;
    bool skew_correction = true;
    CellLimiter cell_limiter = CellLimiter::None;
    double regularization = 0; // Venkatakrishnan epsilon, in scalar units.
};
struct IterationOptions {
    int maximum_iterations = 100;
    double relative_tolerance = 1.e-6, absolute_tolerance = 0, relaxation = 1;
};
struct LinearOptions {
    std::string method = "GMRES", preconditioner = "RELAXATION";
    double relative_tolerance = 1.e-8, absolute_tolerance = 0;
    int maximum_iterations = 500, restart_length = 50;
    bool verify_residual = false;
};
/// @brief Host settings for [scalars.<field>]; no mesh, field, solver or device ownership.
struct Configuration {
    std::string field;
    double density = 1, diffusion = 0.01, source = 0;
    bool convection_enabled = true, diffusion_enabled = true;
    bool gradient_enabled = true;
    numerics::gradient::Options gradient;
    ReconstructionOptions reconstruction;
    bool nonorthogonal_correction = false, require_orthogonal = true;
    double orthogonality_tolerance = 1.e-6;
    TimeScheme time_scheme = TimeScheme::BackwardEuler;
    double time_step = 0.01;
    int steps = 1;
    IterationOptions outer;
    LinearOptions linear;
    [[nodiscard]] bool needs_gradients() const noexcept
    { return nonorthogonal_correction || (convection_enabled && reconstruction.method != ReconstructionMethod::Upwind); }
};
/// @brief Parses the selected field only; other fields and application tables are untouched.
[[nodiscard]] Configuration parse_config(std::string_view text, std::string_view field);
/// @brief Reads on rank zero and parses the broadcast text on every rank.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, std::string_view field, mpi::CommView comm);
} // namespace krona::transport::scalar