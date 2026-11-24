#include "driver.hpp"
#include "cli.hpp"
#include "initial_field.hpp"
#include "source.hpp"
#include "velocity.hpp"

#include <krona/boundary/scalar/tabulated_config.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/solvers/scalar/solver.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/transport/diffusion/isotropic.hpp>

#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace krona::apps::scalar_transport {
namespace {

namespace bc = boundary::scalar;
namespace scalar_solver = solvers::scalar;
namespace ts = transport::scalar;
namespace tt = ts::transient;
namespace grad = numerics::gradient;

using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
using Clock = std::chrono::steady_clock;

void broadcast_options(Options& options, mpi::CommView comm) {
    std::array settings{static_cast<int>(options.problem), static_cast<int>(options.verbosity)};
    mpi::broadcast(comm, std::span{settings});
    toml_config::broadcast(options.mesh, comm);
    toml_config::broadcast(options.config, comm);
    toml_config::broadcast(options.boundary, comm);
    toml_config::broadcast(options.vtu_directory, comm);
    toml_config::broadcast(options.field, comm);
    options.problem = static_cast<Case>(settings[0]);
    options.verbosity = static_cast<Verbosity>(settings[1]);
}

void write_pvd(const std::filesystem::path& directory,
               const std::vector<std::pair<Real, std::string>>& snapshots) {
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.imbue(std::locale::classic());
    out.open(directory / "scalar.pvd");
    out << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n"
           "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
           "  <Collection>\n";

    for (const auto& [time, file] : snapshots) {
        out << "    <DataSet timestep=\"" << time
            << "\" group=\"\" part=\"0\" file=\"" << file << "\"/>\n";
    }

    out << "  </Collection>\n"
           "</VTKFile>\n";
    out.close();
}

linalg::SolverOptions solver_options(const ts::LinearOptions& input) {
    linalg::SolverOptions options;
    options.method = input.method;
    options.preconditioner.type = input.preconditioner;
    options.relative_tolerance = input.relative_tolerance;
    options.absolute_tolerance = input.absolute_tolerance;
    options.maximum_iterations = input.maximum_iterations;
    options.restart_length = input.restart_length;
    options.verify_residual = input.verify_residual;
    return options;
}

/// @brief Copies the owned field for output or final analytic error diagnostics.
template<class Field>
void report_solution(const Options& options, const ts::Configuration& config, const mesh::host::MeshPart& host,
                     Field owned_field, Real time, const std::filesystem::path& directory,
                     bool final_step, mpi::CommView comm) {
    const bool exact_available = has_analytic_solution(options.problem);
    if (directory.empty() && !(exact_available && final_step)) return;

    const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, owned_field);
    const auto count = values.extent(0);
    std::vector<double> exact, error;
    const std::string exact_name = config.field + "_exact";
    const std::string error_name = config.field + "_error";
    std::vector<io::vtk::Field> output;
    output.push_back(io::vtk::field(config.field, std::span{values.data(), count}));

    if (exact_available) {
        const double diffusivity = config.diffusion_enabled ? config.diffusion/config.density : 0;
        exact.resize(count);
        error.resize(count);
        double squared = 0, volume = 0, maximum = 0;
        for (std::size_t i = 0; i < count; ++i) {
            exact[i] = analytic_solution(options.problem, static_cast<double>(host.cell_centroid_x[i]),
                static_cast<double>(host.cell_centroid_y[i]), static_cast<double>(host.cell_centroid_z[i]),
                static_cast<double>(time), diffusivity);
            error[i] = static_cast<double>(values(i))-exact[i];
            const double cell_volume = static_cast<double>(host.cell_volume[i]);
            if (final_step) {
                squared += cell_volume*error[i]*error[i];
                volume += cell_volume;
                maximum = std::max(maximum, std::abs(error[i]));
            }
        }
        if (final_step) {
            const double global_squared = mpi::allreduce<mpi::Reduction::sum>(comm, squared);
            const double global_volume = mpi::allreduce<mpi::Reduction::sum>(comm, volume);
            const double global_maximum = mpi::allreduce<mpi::Reduction::max>(comm, maximum);
            if (comm.rank() == 0)
                std::fprintf(stderr, "KRONA scalar_transport: analytic error at t=%.8g: RMS=%.8g, Linf=%.8g\n",
                    static_cast<double>(time), std::sqrt(global_squared/global_volume), global_maximum);
        }
        output.push_back(io::vtk::field(exact_name, std::span{exact}));
        output.push_back(io::vtk::field(error_name, std::span{error}));
    }
    if (!directory.empty())
        io::vtk::write_vtu(host, directory, "scalar", comm, {{}, output, {}});
}

int solve(const Options& options, mpi::CommView comm) {
    const auto start = Clock::now();
    const bool verbose = comm.rank() == 0 && options.verbosity != Verbosity::quiet;
    const bool detailed = comm.rank() == 0 && options.verbosity == Verbosity::detailed;
    
    const auto config = ts::read_config(options.config, options.field, comm);
    const auto bc_config = bc::read_config(options.boundary, options.field, comm);
    
    const bool steady = config.time_scheme == ts::TimeScheme::Steady;
    const bool gradients_needed = config.needs_gradients();

    const auto output_directory = options.vtu_directory.empty()
        ? config.output.directory : std::filesystem::path{options.vtu_directory};

    const Real density = static_cast<Real>(config.density);
    const Real diffusion_coefficient = static_cast<Real>(config.diffusion);
    mpi::require(std::isfinite(density) && density > 0 &&
                 (!config.diffusion_enabled || (std::isfinite(diffusion_coefficient) && diffusion_coefficient > 0)),
                 "Scalar transport: physics or step controls are unrepresentable in ComputeReal");

    if (options.problem != Case::custom) {
        mpi::require(!steady && config.source == 0, "Scalar transport: analytic cases require transient, source-free transport");
        mpi::require(options.problem != Case::diffusion || (!config.convection_enabled && config.diffusion_enabled),
                     "Scalar transport: diffusion case requires diffusion only");
        mpi::require(options.problem != Case::gaussian || config.convection_enabled,
                     "Scalar transport: gaussian case requires convection");
    }

    const Exec exec;
    const auto host = io::solver_mesh::import_mesh_hdf5(options.mesh, comm);
    mpi::require(host.n_cells_g > 0, "Scalar transport: solver mesh must contain cells");
    if (verbose)
        std::fprintf(stderr, "KRONA scalar_transport: field=%s cells=%llu ranks=%d backend=%s\n",
            config.field.c_str(), static_cast<unsigned long long>(host.n_cells_g), comm.size(), Exec::name());

    // Attach only the additional face metrics used by the selected operators.
    using mesh::AuxConnType;
    using mesh::AuxGeomType;
    auto geometry_mask = AuxGeomType::None;
    if (config.diffusion_enabled)
        geometry_mask = geometry_mask | AuxGeomType::FaceNormalDistance | AuxGeomType::FaceAreaOverNormalDistance;
    if (config.nonorthogonal_correction)
        geometry_mask = geometry_mask | AuxGeomType::FaceInterpolationWeights | AuxGeomType::FaceNonOrthogonalityVector;
    if (gradients_needed) 
        geometry_mask = geometry_mask | grad::requirements(config.gradient).geometry;
    
    const auto host_conn = mesh::host::build_aux_connectivity(host, AuxConnType::CellFaces);
    const auto host_geom = mesh::host::build_aux_geometry(host, geometry_mask);
    const mesh::device::MeshStorage device(host, {}, exec);
    const mesh::device::AuxConnectivityStorage connectivity(host_conn, exec);
    const mesh::device::AuxGeometryStorage geometry(host_geom, exec);
    const auto m = device.view();
    const std::size_t owned = static_cast<std::size_t>(m.sizes.n_own);
    const std::size_t cells = static_cast<std::size_t>(m.sizes.n_cells);

    const boundary::BoundaryLayout<Exec, Real> layout(host, m, exec);
    bc::TabulatedManager<Exec, Real> boundary(layout);
    bc::configure_tabulated(boundary, bc_config, host, comm);
    boundary.finalize();

    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{diffusion_coefficient}};
    const auto flux = make_mass_flux(options.problem, density, boundary.catalog());

    const auto field_size = gradients_needed ? boundary.required_field_size() : cells;
    const auto location = gradients_needed ? fields::FieldLocation::CellExtnd : fields::FieldLocation::Cell;
    fields::FieldStorage<StorageReal, 1, Exec> phi(config.field, field_size, location, exec);
    
    const auto q = phi.view().component(0);
    const auto q_owned = Kokkos::subview(q, std::pair<std::size_t, std::size_t>{0, owned});
    const auto problem = options.problem;
    Kokkos::parallel_for("scalar_transport/initial_field", Policy(exec, 0, owned), KOKKOS_LAMBDA(std::size_t i) {
        q(i) = static_cast<StorageReal>(initial_field(problem, 
                                                    static_cast<Real>(m.geometry.cell_centroid_x(i)),
                                                    static_cast<Real>(m.geometry.cell_centroid_y(i)),
                                                    static_cast<Real>(m.geometry.cell_centroid_z(i))));
    });
    
    scalar_solver::Options solver_config;
    solver_config.convection = config.convection_enabled;
    solver_config.diffusion = config.diffusion_enabled;
    solver_config.nonorthogonal_correction = config.nonorthogonal_correction;
    solver_config.diffusion_geometry = {config.require_orthogonal, config.orthogonality_tolerance};
    solver_config.gradient = config.gradient;
    solver_config.reconstruction = config.reconstruction;
    solver_config.time_scheme = config.time_scheme;
    solver_config.time_step = config.time_step;
    solver_config.outer = config.outer;
    solver_config.linear = solver_options(config.linear);

    scalar_solver::Solver<Exec, Real> solver(host, m, connectivity.view(), geometry.view(),
                                           device.halo_plan(), phi.view(), solver_config, comm, exec);
    const tt::DensityVolumeMass mass{m.geometry.cell_volume, density};
    if (!steady) solver.initialize_history(mass);

    std::vector<std::pair<Real, std::string>> snapshots;
    const int steps = steady ? 1 : config.steps;
    Real time = 0;
    for (int step = 0; step < steps; ++step) {
        const auto source = make_source(problem, m, solver.next_time(), static_cast<Real>(config.source));
        const auto result = steady ? solver.solve_steady(boundary, flux, law, source)
                                   : solver.advance(boundary, flux, law, mass, source);
        time = result.time;
        if (detailed) {
            for (int outer = 0; outer < config.outer.iterations; ++outer)
                std::fprintf(stderr, "  step=%d outer=%d linear=%d\n", step + 1, outer + 1,
                    result.linear_iterations[static_cast<std::size_t>(outer)]);
            std::fprintf(stderr, "  accepted step=%d t=%.8g outer=%d\n",
                step + 1, static_cast<double>(time), config.outer.iterations);
        }

        const int accepted = step + 1;
        const bool final_step = accepted == steps;
        const bool periodic = !steady && config.output.every > 0;
        const bool write = !output_directory.empty() &&
            (final_step || (periodic && accepted % config.output.every == 0));
        if (write || (final_step && has_analytic_solution(problem))) {
            std::filesystem::path directory;
            if (write) directory = periodic ? output_directory / ("step_" + std::to_string(accepted)) : output_directory;
            report_solution(options, config, host, q_owned, time, directory, final_step, comm);
            
            if (write && periodic && comm.rank() == 0) {
                const auto file = directory.filename() / "scalar.pvtu";
                snapshots.emplace_back(time, file.generic_string());
            }
        }
    }

    if (comm.rank() == 0 && !snapshots.empty())
        write_pvd(output_directory, snapshots);

    if (verbose)
        std::fprintf(stderr, "KRONA scalar_transport: completed %d step(s) at t=%.8g in %.3f s.\n",
            steps, static_cast<double>(time), std::chrono::duration<double>(Clock::now()-start).count());
    return EXIT_SUCCESS;
}

} // namespace

int run(int argc, char* const* argv, mpi::CommView comm) noexcept {
    return mpi::guarded([&] {
        mpi::require(!comm.is_null(), "Scalar transport app requires a live communicator");
        ParseResult parsed;
        if (comm.rank() == 0) parsed = parse_arguments(argc, argv);
        int action = static_cast<int>(parsed.action);
        mpi::broadcast(comm, action);
        if (action == static_cast<int>(CliAction::help)) {
            if (comm.rank() == 0) usage(stdout);
            return EXIT_SUCCESS;
        }
        if (action == static_cast<int>(CliAction::error)) {
            if (comm.rank() == 0) print_error(parsed, stderr);
            return EXIT_FAILURE;
        }
        broadcast_options(parsed.options, comm);

        const Tpetra::ScopeGuard runtime(nullptr, nullptr, comm.native());
        return mpi::guarded([&] { return solve(parsed.options, comm); });
    });
}

} // namespace krona::apps::scalar_transport
