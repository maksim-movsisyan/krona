#pragma once

#include <krona/mpi/communicator.hpp>
#include <krona/numerics/gradient/types.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <string_view>

namespace krona::transport::compressible {

/// @brief Selects the Riemann flux used in the convective residual.
enum class Flux { Hllc, Hll, Rusanov };

/// @brief Selects the formula used to reconstruct primitives between cells.
enum class ReconstructionMethod { FirstOrder, Linear, MusclKappa, Tvd };

/// @brief Chooses the slope-ratio law for TVD reconstruction.
enum class RatioLimiter { Minmod, Superbee, MonotonizedCentral, VanLeer, VanAlbada, Koren };

/// @brief Chooses whether to bound reconstructed values by the neighboring samples.
enum class CellLimiter { None, BarthJespersen };

/// @brief Controls reconstruction of the five primitive variables at faces.
struct ReconstructionOptions {
    ReconstructionMethod method = ReconstructionMethod::FirstOrder;
    RatioLimiter tvd_limiter = RatioLimiter::VanLeer;
    CellLimiter cell_limiter = CellLimiter::None;
    double kappa = 1.0 / 3.0;
    bool skew_correction = true; ///< Moves line-based MUSCL and TVD profiles to the face centroid.

    [[nodiscard]] bool needs_gradients() const noexcept { return method != ReconstructionMethod::FirstOrder; }
};

enum class ViscosityModel { Constant, Sutherland };
enum class ConductivityModel { Constant, ConstantPrandtl };

/// @brief Parameters of the gas's molecular transport correlations.
struct MolecularOptions {
    ViscosityModel viscosity = ViscosityModel::Constant;
    ConductivityModel conductivity = ConductivityModel::ConstantPrandtl;
    double dynamic_viscosity = 1.8e-5;
    double bulk_viscosity = 0;
    double thermal_conductivity = 0.025;
    double prandtl = 0.72;
    double reference_viscosity = 1.716e-5;
    double reference_temperature = 273.15;
    double sutherland_temperature = 110.4;
};

/// @brief Uniform reference primitives used by the application initialization.
struct InitialOptions {
    double pressure = 0;
    double temperature = 0;
    std::array<double, 3> velocity{};
};

/// @brief Local pseudo-time step, update relaxation and global residual stopping criteria.
struct PseudoTimeOptions {
    double cfl = 1.0;
    double relaxation = 1.0;
    double absolute_tolerance = 1e-8;
    double relative_tolerance = 1e-8;
    int maximum_iterations = 200;
};

/// @brief Host configuration of the Krylov solve and preconditioner.
struct LinearOptions {
    std::string method = "GMRES";
    std::string preconditioner = "RELAXATION";
    double relative_tolerance = 1e-6;
    double absolute_tolerance = 0;
    int maximum_iterations = 100;
    int restart_length = 30;
    bool verify_residual = false;
};

/// @brief Host settings selected by [flow] in the solver TOML.
struct Configuration {
    double gamma = 1.4;
    double gas_constant = 287.05;
    InitialOptions initial;
    bool viscous_enabled = false; ///< Include momentum diffusion, viscous work and heat conduction.
    MolecularOptions molecular;
    PseudoTimeOptions pseudo_time;
    LinearOptions linear;
    Flux flux = Flux::Hllc;
    ReconstructionOptions reconstruction;
    numerics::gradient::Options gradient;
    std::filesystem::path output_directory;
};

/// @brief Parses the [flow] table and validates the settings locally.
/// @param text Whole TOML input containing [flow.initial].
[[nodiscard]] Configuration parse_config(std::string_view text);

/// @brief Reads on rank zero, broadcasts the text and parses on every rank.
/// @param file TOML file read by rank zero of comm.
/// @param comm Communicator shared by all callers.
[[nodiscard]] Configuration read_config(const std::filesystem::path& file, mpi::CommView comm);

} // namespace krona::transport::compressible
