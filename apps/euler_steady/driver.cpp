#include "driver.hpp"
#include "cli.hpp"
#include "config.hpp"
#include "reconstruction.hpp"
#include "step.hpp"

#include <krona/boundary/physical/tabulated_config.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/linalg/field_transfer.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/numerics/reconstruction/stencil_data.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/transport/euler/assembly.hpp>
#include <krona/transport/euler/operator.hpp>

#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace krona::apps::euler_steady {
namespace {

namespace eu = transport::euler;
namespace rec = numerics::reconstruction;

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

linalg::SolverOptions solver_options(const LinearOptions& input) {
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

/// @brief Selects the residual flux from the configuration with one kernel type.
struct NumericalFlux {
    Flux kind;

    KOKKOS_INLINE_FUNCTION ri::Result<Real> operator()(ri::State<Real> left, ri::State<Real> right,
                                                       ri::Vector<Real> normal) const {
        switch (kind) {
            case Flux::Hllc: return ri::Hllc{}(left, right, normal);
            case Flux::Hll: return ri::Hll{}(left, right, normal);
            case Flux::Rusanov: return ri::Rusanov{}(left, right, normal);
        }
        return {{}, Real{0}, ri::Status::InvalidState};
    }
};

Real compute_value(double value, const char* message) {
    mpi::require(std::abs(value) <= static_cast<double>(Kokkos::finite_max_v<Real>), message);
    return static_cast<Real>(value);
}

/// @brief Exports the owned primitive fields and the density from the conservative state.
template<class View>
void write_solution(const mesh::host::MeshPart& host, const std::filesystem::path& directory,
                    View q, Conserved conserved, mpi::CommView comm) {
    if (directory.empty()) return;

    const auto cells = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, conserved);
    const auto count = cells.extent(0);
    std::vector<Real> density(count);
    for (std::size_t i = 0; i < count; ++i) density[i] = cells(i).density;

    const auto primitive = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, q);
    const auto component = [&](std::size_t c) {
        // The application's default FieldStorage layout keeps each component contiguous.
        const auto values = Kokkos::subview(primitive, Kokkos::ALL(), c);
        return std::span<const StorageReal>{values.data(), count};
    };
    const std::array fields{io::vtk::field("density", std::span{density}),
        io::vtk::field("pressure", component(0)), io::vtk::field("temperature", component(4)),
        io::vtk::vector_field("velocity", component(1), component(2), component(3))};
    io::vtk::write_vtu(host, directory, "euler", comm, {{}, fields, {}});
}

int solve(const Options& options, mpi::CommView comm) {
    const auto config = read_config(options.config, comm);
    const auto bc_config = ph::read_config(options.boundary, comm);
    const auto output_directory = options.vtu_directory.empty()
        ? config.output_directory : std::filesystem::path{options.vtu_directory};
    const auto& controls = config.pseudo_time;
    const Gas gas{compute_value(config.gamma, "Steady Euler: gamma is unrepresentable"),
                  compute_value(config.gas_constant, "Steady Euler: gas_constant is unrepresentable")};
    gas.validate();
    const Real cfl = compute_value(controls.cfl, "Steady Euler: CFL is unrepresentable");
    const Real relaxation = compute_value(controls.relaxation, "Steady Euler: relaxation is unrepresentable");
    mpi::require(cfl > Real{0} && relaxation > Real{0}, "Steady Euler: CFL or relaxation rounded to zero");

    const auto& initial = config.initial;
    constexpr auto state_error = "Steady Euler: reference initial state is unrepresentable";
    const ph::Primitive<Real, true> reference{compute_value(initial.pressure, state_error),
        {compute_value(initial.velocity[0], state_error), compute_value(initial.velocity[1], state_error),
         compute_value(initial.velocity[2], state_error)}, compute_value(initial.temperature, state_error)};

    // Load the partition and bind the common mesh, physical boundaries and algebraic graph.
    const Exec exec;
    const auto host = io::solver_mesh::import_mesh_hdf5(options.mesh, comm);
    mpi::require(host.n_cells_g > 0, "Steady Euler: solver mesh must contain cells");
    const auto host_connectivity = mesh::host::build_aux_connectivity(host, mesh::AuxConnType::CellFaces);
    const mesh::device::MeshStorage device(host, {}, exec);
    const mesh::device::AuxConnectivityStorage connectivity(host_connectivity, exec);
    const auto m = device.view();
    const auto owned = static_cast<std::size_t>(m.sizes.n_own);

    const boundary::BoundaryLayout<Exec, Real> layout(host, m, exec);
    using Model = ph::Compressible<Gas, ph::PerfectGasCharacteristics>;
    ph::TabulatedManager<Model, Exec, Real> boundary(layout, Model{gas, {}});
    ph::configure_tabulated(boundary, bc_config, host, comm);
    boundary.finalize();

    fields::FieldStorage<StorageReal, 5, Exec> q("primitives", boundary.required_field_size(),
        fields::FieldLocation::CellExtnd, exec);
    fields::FieldStorage<AccumulationReal, 5, Exec> residual("Euler residual", owned,
        fields::FieldLocation::OwnCell, exec);
    const Conserved conserved("Euler conservative state", owned);
    const auto q_view = q.view();
    const auto primitive = ph::primitive_components(q_view.component(0), q_view.component(1), q_view.component(2),
                                                    q_view.component(3), q_view.component(4));
    // Initialize the extended field. The second-order path fills physical-face samples before reconstruction.
    q.fill_async(exec, StorageReal{0});
    initialize_state(exec, m, conserved, primitive, gas, options.problem, reference);

    halo::Exchange<StorageReal, Exec> halo(m, device.halo_plan(), comm, 81, exec);
    halo.add_field(q_view);
    halo.prepare();

    const eu::EulerPlan<Exec, Real> plan(host, m, connectivity.view(), comm, exec);
    const eu::EulerOperator euler{plan, eu::Thermodynamics{gas}, NumericalFlux{config.flux}};
    const rec::StencilData<Space, Real> stencil(m, connectivity.view());
    const auto values = fields::pack(q_view);
    const auto closure = eu::EvaluatedExterior<>{};
    const auto inner_jacobian = eu::FrozenRusanov{gas};
    const auto output = fields::pack(residual.view());
    eu::AssemblyStatus<Exec> status(exec);

    const linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
    linalg::BsrMatrix<SolverReal, Exec> matrix(assembly.pattern(), 5);
    linalg::Vector<SolverReal, Exec> rhs(matrix.distribution()), increment(matrix.distribution());
    linalg::Solver linear(matrix, solver_options(config.linear));

    const bool root = comm.rank() == 0;
    const bool detailed = root && options.verbosity == Verbosity::detailed;
    if (root && options.verbosity != Verbosity::quiet)
        std::fprintf(stderr, "KRONA euler_steady: cells=%llu ranks=%d backend=%s CFL=%.6g reconstruction=%s\n",
            static_cast<unsigned long long>(host.n_cells_g), comm.size(), Exec::name(), controls.cfl,
            config.reconstruction == ReconstructionOrder::First ? "first_order" : "second_order");

    // The iteration is shared by the piecewise-constant and linear reconstruction types.
    const auto run_iterations = [&](const auto& reconstruction, const auto& prepare_reconstruction) {
        double initial_norm = 0, norm = 0;
        int updates = 0;
        bool converged = false;
        while (true) {
            // Form the stationary residual of the current iterate before deciding whether to update it.
            halo.exchange();
            prepare_reconstruction();
            residual.fill_async(exec, AccumulationReal{0});
            status.begin_async();
            euler.add_residual_async(reconstruction, output, boundary, status, closure);
            linalg::copy_from_fields(rhs, output, exec);
            norm = static_cast<double>(rhs.native()->norm2());
            mpi::require(std::isfinite(norm), "Steady Euler: nonfinite spatial residual norm");

            if (updates == 0) initial_norm = norm;
            const double tolerance = std::max(controls.absolute_tolerance, controls.relative_tolerance * initial_norm);
            converged = norm <= tolerance;
            if (detailed)
                std::fprintf(stderr, "KRONA euler_steady: iteration %d, spatial residual %.9e\n", updates, norm);
            if (converged || updates == controls.maximum_iterations) {
                (void)status.check(comm);
                break;
            }

            // Assemble (J + V/dtau I) and solve for a conservative increment with a zero initial guess.
            const auto target = eu::make_jacobian_view(matrix, assembly, exec);
            Kokkos::deep_copy(exec, target.blocks, SolverReal{0});
            euler.add_jacobian_async(target, reconstruction, boundary, status, inner_jacobian, closure);
            add_pseudo_time_diagonal(exec, m, connectivity.view(), target, primitive, gas, cfl);
            (void)status.check(comm);
            matrix.commit();

            increment.fill(SolverReal{0}, exec);
            const auto result = linear.solve(rhs, increment);
            mpi::require(result.converged, "Steady Euler: linear solve did not converge");
            if (detailed)
                std::fprintf(stderr, "KRONA euler_steady: linear iterations %d\n", result.iterations);

            // Keep the conservative iterate as the source of truth and publish primitives for the next residual.
            update_state(exec, conserved, increment.read(exec), primitive, gas, relaxation);
            increment.commit();
            ++updates;
        }

        write_solution(host, output_directory, q_view.kokkos_view(), conserved, comm);
        if (root)
            std::fprintf(stderr, "KRONA euler_steady: %s after %d updates, spatial residual %.9e -> %.9e\n",
                converged ? "converged" : "iteration limit reached", updates, initial_norm, norm);
        return converged ? EXIT_SUCCESS : EXIT_FAILURE;
    };

    if (config.reconstruction == ReconstructionOrder::First) {
        const auto reconstruction = eu::Reconstruction{rec::face_profiles(stencil, values, rec::FirstOrder{})};
        return run_iterations(reconstruction, [] {});
    }

    // Linear reconstruction needs 3 derivatives per primitive and one limiter factor per primitive.
    const auto cells = static_cast<std::size_t>(m.sizes.n_cells);
    fields::FieldStorage<StorageReal, 15, Exec> gradients("primitive gradients", cells, fields::FieldLocation::Cell, exec);
    fields::FieldStorage<StorageReal, 5, Exec> limiter("primitive limiter", cells, fields::FieldLocation::Cell, exec);
    const numerics::gradient::GradientOperator<Exec> gradient(m, connectivity.view(), {}, {}, exec);
    const auto grad = rec::gradient_pack(gradients.view());
    const auto unlimited = rec::face_profiles(stencil, values, rec::Linear{grad});
    const auto factors = fields::pack(limiter.view());
    const auto reconstruction = eu::Reconstruction{rec::limited(unlimited, factors)};

    const bool limit = config.limiter == Limiter::BarthJespersen;
    halo::Exchange<StorageReal, Exec> reconstruction_halo(m, device.halo_plan(), comm, 82, exec);
    reconstruction_halo.add_field(gradients.view());
    if (limit) reconstruction_halo.add_field(limiter.view());
    else limiter.fill_async(exec, StorageReal{1});
    reconstruction_halo.prepare();

    ph::Status<Exec> boundary_status(exec);
    return run_iterations(reconstruction, [&] {
        // Ghost primitives are current. Publish physical samples before computing owned slopes.
        boundary_status.begin_async();
        boundary.for_each_face_async(FillInviscidFaceSamples{primitive, boundary.model(), boundary_status.device_flags(exec)});
        boundary_status.check();
        gradient.apply_async(q_view, gradients.view());
        if (limit)
            rec::compute_limiters_async<Real>(exec, unlimited, factors);
        reconstruction_halo.exchange();
    });
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

} // namespace krona::apps::euler_steady
