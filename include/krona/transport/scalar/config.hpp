#pragma once
#include <krona/config.hpp>
#include <krona/numerics/gradient/types.hpp>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace krona::transport::scalar {
enum class ReconstructionMethod { Upwind, Linear, MusclKappa, Tvd };
enum class RatioLimiter { Minmod, Superbee, MonotonizedCentral, VanLeer, VanAlbada, Koren };
enum class CellLimiter { None, BarthJespersen, Venkatakrishnan };
enum class TimeScheme { BackwardEuler, Bdf2 };
struct ConstantInitial { double value = 0; };
struct LinearInitial {
    double value = 0;
    std::array<double, 3> gradient{}, origin{};
};
struct GaussianInitial {
    double background = 0, amplitude = 1, width = 0.1;
    std::array<double, 3> centre{0.5, 0.5, 0.5};
};
using InitialCondition = std::variant<ConstantInitial, LinearInitial, GaussianInitial>;
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
    std::array<double, 3> velocity{};
    bool convection = true, diffusion_enabled = true;
    bool gradient_enabled = true;
    numerics::gradient::Options gradient;
    ReconstructionOptions reconstruction;
    bool nonorthogonal_correction = false, require_orthogonal = true;
    double orthogonality_tolerance = 1.e-6;
    TimeScheme time_scheme = TimeScheme::BackwardEuler;
    IterationOptions outer;
    LinearOptions linear;
    InitialCondition initial = ConstantInitial{};
    std::optional<InitialCondition> reference; // Built-in analytic profile; defaults to initial data.
    bool verification = false;
    std::optional<double> maximum_l2_error;
    [[nodiscard]] bool needs_gradients() const noexcept
    { return nonorthogonal_correction || (convection && reconstruction.method != ReconstructionMethod::Upwind); }
};
/// @brief Parses the selected field only; other fields and application tables are untouched.
[[nodiscard]] Configuration parse_config(std::string_view text, std::string_view field);
/// @brief Local file read; no MPI communication. Invalid input follows mpi::fatal policy.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, std::string_view field);
} // namespace krona::transport::scalar
