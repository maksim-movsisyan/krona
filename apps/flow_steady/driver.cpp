#include "driver.hpp"
#include "cli.hpp"
#include "initial_field.hpp"
#include "molecular_properties.hpp"
#include "output.hpp"

#include <krona/boundary/physical/tabulated_config.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/solvers/flow/solver.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/transport/compressible/config.hpp>

#include <Tpetra_Core.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>

namespace krona::apps::flow_steady {
namespace {

namespace ct = transport::compressible;
namespace fs = solvers::flow;
namespace ph = boundary::physical;
namespace gr = numerics::gradient;

using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Gas = thermodynamics::IdealGas<Real>;

void broadcast_options(Options& options, mpi::CommView comm) {
    std::array settings{static_cast<int>(options.problem), static_cast<int>(options.verbosity)};
    mpi::broadcast(comm, std::span{settings});
    toml_config::broadcast(options.mesh, comm);
    toml_config::broadcast(options.config, comm);
    toml_config::broadcast(options.boundary, comm);
    toml_config::broadcast(options.vtu_directory, comm);
    options.problem = static_cast<Case>(settings[0]);
    options.verbosity = static_cast<Verbosity>(settings[1]);
}

linalg::SolverOptions linear_options(const ct::LinearOptions& input) {
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

Real compute_value(double value, const char* message) {
    mpi::require(std::abs(value) <= static_cast<double>(Kokkos::finite_max_v<Real>), message);
    return static_cast<Real>(value);
}

int solve(const Options& options, mpi::CommView comm) {
    const auto config = ct::read_config(options.config, comm);
    const auto bc_config = ph::read_config(options.boundary, comm);
    const auto output_directory = options.vtu_directory.empty() ? config.output_directory
        : std::filesystem::path{options.vtu_directory};

    const Gas gas{compute_value(config.gamma, "Steady flow: gamma is unrepresentable"),
                  compute_value(config.gas_constant, "Steady flow: gas_constant is unrepresentable")};
    gas.validate();
    auto properties = molecular_properties(config.molecular, gas);

    fs::Options settings;
    settings.viscous_enabled = config.viscous_enabled;
    settings.bulk_viscosity = config.molecular.bulk_viscosity;
    settings.flux = config.flux;
    settings.reconstruction = config.reconstruction;
    settings.gradient = config.gradient;
    settings.pseudo_time = config.pseudo_time;
    settings.linear = linear_options(config.linear);

    const Exec exec;
    const auto host = io::solver_mesh::import_mesh_hdf5(options.mesh, comm);
    mpi::require(host.n_cells_g > 0, "Steady flow: solver mesh must contain cells");

    const bool extended = settings.needs_gradients();
    const auto geometry_mask = extended ? gr::requirements(settings.gradient).geometry : mesh::AuxGeomType::None;
    const auto host_geometry = mesh::host::build_aux_geometry(host, geometry_mask);
    const auto host_connectivity = mesh::host::build_aux_connectivity(host, mesh::AuxConnType::CellFaces);

    const mesh::device::MeshStorage device(host, {}, exec);
    const mesh::device::AuxConnectivityStorage connectivity(host_connectivity, exec);
    const mesh::device::AuxGeometryStorage geometry(host_geometry, exec);
    const auto mesh_view = device.view();
    const auto owned = static_cast<std::size_t>(mesh_view.sizes.n_own);

    const boundary::BoundaryLayout<Exec, Real> layout(host, mesh_view, exec);
    using Model = ph::Compressible<Gas, ph::PerfectGasCharacteristics>;
    ph::TabulatedManager<Model, Exec, Real> boundary(layout, Model{gas, {}});
    ph::configure_tabulated(boundary, bc_config, host, comm);
    boundary.finalize();

    const std::size_t cells = static_cast<std::size_t>(mesh_view.sizes.n_cells);
    const std::size_t field_size = extended ? boundary.required_field_size() : cells;
    const auto location = extended ? fields::FieldLocation::CellExtnd : fields::FieldLocation::Cell;
    fields::FieldStorage<StorageReal, 1, Exec> pressure("pressure", field_size, location, exec);
    fields::FieldStorage<StorageReal, 3, Exec> velocity("velocity", field_size, location, exec);
    fields::FieldStorage<StorageReal, 1, Exec> temperature("temperature", field_size, location, exec);
    const auto primitives = fields::pack(pressure.view(), velocity.view(), temperature.view());

    constexpr auto own = fields::FieldLocation::OwnCell;
    fields::FieldStorage<Real, 1, Exec> density("density", owned, own, exec);
    fields::FieldStorage<Real, 3, Exec> momentum("momentum", owned, own, exec);
    fields::FieldStorage<Real, 1, Exec> energy("total_energy_density", owned, own, exec);
    const auto conserved = fields::pack(density.view(), momentum.view(), energy.view());

    using Solver = fs::Solver<Exec, Real, decltype(properties)>;
    const auto gradient_size = extended ? cells : 0;
    constexpr auto cell = fields::FieldLocation::Cell;
    fields::FieldStorage<StorageReal, 3, Exec> grad_pressure("grad_pressure", gradient_size, cell, exec);
    fields::FieldStorage<StorageReal, 9, Exec> grad_velocity("grad_velocity", gradient_size, cell, exec);
    fields::FieldStorage<StorageReal, 3, Exec> grad_temperature("grad_temperature", gradient_size, cell, exec);
    std::optional<Solver::GradientPack> gradients;
    if (extended)
        gradients = fields::pack(grad_pressure.view(), grad_velocity.view(), grad_temperature.view());

    Solver solver(host, mesh_view, connectivity.view(), geometry.view(), device.halo_plan(),
                  conserved, primitives, gradients, gas, properties, settings, comm, exec);

    const auto& initial = config.initial;
    constexpr auto invalid = "Steady flow: initial state is unrepresentable";
    const ph::Primitive<Real, true> reference{compute_value(initial.pressure, invalid),
          {compute_value(initial.velocity[0], invalid),     
                     compute_value(initial.velocity[1], invalid),
                     compute_value(initial.velocity[2], invalid)},
         compute_value(initial.temperature, invalid)};
    solver.initialize(InitialField<Real>{options.problem, reference});

    const bool root = comm.rank() == 0;
    const bool detailed = root && options.verbosity == Verbosity::detailed;
    const char* flow_name = config.viscous_enabled ? "navier_stokes" : "euler";
    if (root && options.verbosity != Verbosity::quiet)
        std::fprintf(stderr, "KRONA %s: cells=%llu ranks=%d backend=%s CFL=%.6g physics=%s\n",
                     program, static_cast<unsigned long long>(host.n_cells_g), comm.size(), Exec::name(),
                     config.pseudo_time.cfl, flow_name);

    const auto result = solver.solve_steady(boundary, [&](int index, const fs::IterationResult& iteration) {
        if (!detailed) return;
        std::fprintf(stderr, "KRONA %s: iteration %d, spatial residual %.9e\n",
                     program, index, iteration.residual_norm);
        if (iteration.linear_iterations)
            std::fprintf(stderr, "KRONA %s: linear iterations %d\n", program, *iteration.linear_iterations);
    });
    write_solution(host, output_directory, flow_name, primitives, conserved, comm);
    if (root)
        std::fprintf(stderr, "KRONA %s: %s after %d updates, spatial residual %.9e -> %.9e\n",
                     program, result.converged ? "converged" : "iteration limit reached", result.updates,
                     result.initial_residual, result.final_residual);
    return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int run(int argc, char* const* argv, mpi::CommView comm) noexcept {
    return mpi::guarded([&] {
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

} // namespace krona::apps::flow_steady
