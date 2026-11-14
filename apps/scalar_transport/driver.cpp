#include "driver.hpp"
#include "cli.hpp"
#include "initial_field.hpp"
#include "reconstruction.hpp"
#include "source.hpp"

#include <krona/boundary/scalar/tabulated_config.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/numerics/time/derivative.hpp>
#include <krona/toml_config/file.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/transport/scalar/source.hpp>
#include <krona/transport/scalar/transient/operator.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace krona::apps::scalar_transport {
namespace {

namespace bc = boundary::scalar;
namespace cv = ts::convection;
namespace df = ts::diffusion;
namespace tt = ts::transient;
namespace grad = numerics::gradient;
namespace nt = numerics::time;

using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = Exec::memory_space;
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

template<class View>
double global_norm(View values, const Exec& exec, mpi::CommView comm) {
    double squared = 0;
    Kokkos::parallel_reduce("scalar_transport/norm", Policy(exec, 0, values.extent(0)),
        KOKKOS_LAMBDA(std::size_t i, double& sum) {
            const double value = static_cast<double>(values(i));
            sum += value*value;
        }, squared);
    const auto total = mpi::allreduce<mpi::Reduction::sum>(comm, squared);
    mpi::require(std::isfinite(total), "Scalar transport: residual norm overflow");
    return std::sqrt(total);
}

/// @brief Copies the owned result for output and optionally measures its analytic error.
template<class Field>
void report_solution(const Options& options, const ts::Configuration& config, const mesh::host::MeshPart& host,
                     Field owned_field, Real time, mpi::CommView comm) {
    const bool exact_available = has_analytic_solution(options.problem);
    if (!exact_available && options.vtu_directory.empty()) return;

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
            squared += cell_volume*error[i]*error[i];
            volume += cell_volume;
            maximum = std::max(maximum, std::abs(error[i]));
        }
        const double global_squared = mpi::allreduce<mpi::Reduction::sum>(comm, squared);
        const double global_volume = mpi::allreduce<mpi::Reduction::sum>(comm, volume);
        const double global_maximum = mpi::allreduce<mpi::Reduction::max>(comm, maximum);
        if (comm.rank() == 0)
            std::fprintf(stderr, "KRONA scalar_transport: analytic error at t=%.8g: RMS=%.8g, Linf=%.8g\n",
                static_cast<double>(time), std::sqrt(global_squared/global_volume), global_maximum);
        output.push_back(io::vtk::field(exact_name, std::span{exact}));
        output.push_back(io::vtk::field(error_name, std::span{error}));
    }
    if (!options.vtu_directory.empty())
        io::vtk::write_vtu(host, options.vtu_directory, "scalar", comm, {{}, output, {}});
}

int solve(const Options& options, mpi::CommView comm) {
    const auto start = Clock::now();
    const bool verbose = comm.rank() == 0 && options.verbosity != Verbosity::quiet;
    const bool detailed = comm.rank() == 0 && options.verbosity == Verbosity::detailed;

    const auto config = ts::read_config(options.config, options.field, comm);
    const auto bc_config = bc::read_config(options.boundary, options.field, comm);

    const bool steady = config.time_scheme == ts::TimeScheme::Steady;
    const bool bdf2 = config.time_scheme == ts::TimeScheme::Bdf2;

    const bool reconstruct = config.reconstruction.method != ts::ReconstructionMethod::Upwind;
    const bool limit = config.reconstruction.cell_limiter != ts::CellLimiter::None;

    const bool gradients_needed = config.needs_gradients();
    const Real density = static_cast<Real>(config.density);
    const Real diffusion_coefficient = static_cast<Real>(config.diffusion);
    const Real dt = steady ? Real{0} : static_cast<Real>(config.time_step);
    const Real relaxation = static_cast<Real>(config.outer.relaxation);

    mpi::require(std::isfinite(density) && density > 0 && relaxation > 0 &&
                 (steady || (std::isfinite(dt) && dt > 0)) &&
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
    bc::Status<Exec> boundary_status(exec);
    ts::AssemblyStatus<Exec> status(exec);

    std::optional<cv::ConvectionOperator<Exec, Real>> convection;
    std::optional<df::DiffusionOperator<Exec, Real>> diffusion;
    std::optional<grad::GradientOperator<Exec>> gradient;
    const ts::SourceOperator<Exec, Real> sources(m, exec);
    const tt::TransientOperator<Exec, Real> temporal(owned, exec);
    const tt::DensityVolumeMass mass{m.geometry.cell_volume, density};

    if (config.convection_enabled)
        convection.emplace(cv::ConvectionPlan<Exec, Real>{m, connectivity.view(), comm, exec});
    if (config.diffusion_enabled)
        diffusion.emplace(df::DiffusionPlan<Exec, Real>{m, connectivity.view(), geometry.view(), comm, exec,
            {config.require_orthogonal, config.orthogonality_tolerance}});
    if (gradients_needed) 
        gradient.emplace(m, connectivity.view(), geometry.view(), config.gradient, exec);

    const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{diffusion_coefficient}};
    const auto flux = make_mass_flux(options.problem, density, boundary.catalog());
    
    fields::FieldStorage<StorageReal, 1, Exec> phi(config.field, boundary.required_field_size(),
                                                   fields::FieldLocation::CellExtnd, exec);
    fields::FieldStorage<StorageReal, 3, Exec> g("gradient", gradients_needed ? cells : 0, fields::FieldLocation::Cell, exec);
    fields::FieldStorage<StorageReal, 1, Exec> theta("limiter", reconstruct ? cells : 0, fields::FieldLocation::Cell, exec);

    const auto q = phi.view().component(0);
    const auto q_owned = Kokkos::subview(q, std::pair<std::size_t, std::size_t>{0, owned});

    const auto problem = options.problem;
    Kokkos::parallel_for("scalar_transport/initial_field", Policy(exec, 0, owned), KOKKOS_LAMBDA(std::size_t i) {
        q(i) = static_cast<StorageReal>(initial_field(problem, static_cast<Real>(m.geometry.cell_centroid_x(i)),
            static_cast<Real>(m.geometry.cell_centroid_y(i)), static_cast<Real>(m.geometry.cell_centroid_z(i))));
    });
    if (gradients_needed) g.fill_async(exec, StorageReal{0});
    if (reconstruct) theta.fill_async(exec, StorageReal{1});

    halo::Exchange<StorageReal, Exec> phi_halo(m, device.halo_plan(), comm, 71, exec);
    phi_halo.add_field(phi.view());
    phi_halo.prepare();
    std::optional<halo::Exchange<StorageReal, Exec>> gradient_halo;
    if (gradients_needed) {
        gradient_halo.emplace(m, device.halo_plan(), comm, 72, exec);
        gradient_halo->add_field(g.view());
        if (limit) gradient_halo->add_field(theta.view());
        gradient_halo->prepare();
    }

    const auto correction = grad::GradientCorrection{grad::cell_gradients(g.view())};
    const rec::MeshStencil<Space, Real> stencil(m, connectivity.view());
    const auto values = fields::pack(phi.view());
    const auto gradients = rec::gradient_pack(g.view());
    const auto factors = fields::pack(theta.view());
    const auto unlimited = rec::data(values, gradients);
    const auto limited = rec::data(values, gradients, factors);

    // Sample boundary slots for gradients and limiter bounds. Assembly reads BCs directly.
    const auto prepare_iterate = [&](Real time) {
        phi_halo.exchange();
        if (!gradient) return;
        boundary_status.begin_async();
        const auto fill_boundary = [&](const auto& boundary_law, const auto& boundary_correction) {
            boundary.apply_async(phi.view(), boundary_status, boundary_law, boundary_correction, time);
        };
        const auto fill = [&](const auto& boundary_correction) {
            if (diffusion) fill_boundary(law, boundary_correction);
            else fill_boundary(bc::NoFluxLaw{}, boundary_correction);
        };
        if (config.nonorthogonal_correction) fill(correction);
        else fill(grad::NoCorrection{});

        gradient->apply_async(phi.view(), g.view());
        if (limit && config.nonorthogonal_correction) fill(correction);
        boundary_status.check();

        if (limit) {
            with_reconstruction<Real>(config.reconstruction, [&](auto scheme) {
                if (config.reconstruction.cell_limiter == ts::CellLimiter::BarthJespersen)
                    rec::compute_limiters_async<Real>(exec, stencil, unlimited, factors, rec::BarthJespersen{}, scheme);
                else {
                    const auto epsilon = static_cast<Real>(config.reconstruction.regularization);
                    mpi::require(std::isfinite(epsilon), "Scalar transport: limiter regularization is unrepresentable");
                    rec::compute_limiters_async<Real>(exec, stencil, unlimited, factors,
                        rec::Venkatakrishnan{rec::ConstantRegularization<Real>{epsilon}}, scheme);
                }
            });
        }
        gradient_halo->exchange();
    };

    // The explicit updates and residuals use the same reconstruction and frozen coefficients.
    const auto with_convection = [&](const auto& submit) {
        if (!convection) return;
        if (!reconstruct) {
            submit(cv::NoReconstruction{});
            return;
        }
        with_reconstruction<Real>(config.reconstruction, [&](auto scheme) {
            // Unit entries in factors give the unlimited scheme when limiting is disabled.
            submit(cv::make_reconstruction(convection->plan(), rec::Limited{scheme}, limited));
        });
    };

    const linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
    linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
    linalg::Vector<SolverReal, Exec> rhs(matrix.distribution()), solution(matrix.distribution());
    linalg::Solver linear(matrix, solver_options(config.linear));
    const auto base_rhs = Kokkos::View<SolverReal*, Space>("scalar_transport/base_rhs", owned);
    const auto residual = Kokkos::View<Real*, Space>("scalar_transport/residual", owned);
    const auto q_n = Kokkos::View<StorageReal*, Space>("scalar_transport/accepted", steady ? 0 : owned);
    const auto q_nm1 = Kokkos::View<StorageReal*, Space>("scalar_transport/previous", bdf2 ? owned : 0);
    if (!steady) Kokkos::deep_copy(exec, q_n, q_owned);
    Real time = 0;

    const int steps = steady ? 1 : config.steps;
    for (int step = 0; step < steps; ++step) {
        const Real next = steady ? Real{0} : static_cast<Real>(step + 1)*dt;
        mpi::require(steady || (std::isfinite(next) && next > time), "Scalar transport: time cannot advance");
        const auto source = make_source(problem, m, next, static_cast<Real>(config.source));

        const auto with_time_term = [&](const auto& submit) {
            if (steady) return;
            const tt::TimeLevel newest{mass, q_n};
            if (bdf2 && step > 0)
                submit(nt::Bdf2{}.coefficients(dt), tt::time_history(newest, tt::TimeLevel{mass, q_nm1}));
            else submit(nt::BackwardEuler{}.coefficients(dt), tt::time_history(newest));
        };

        // Assemble the base system at the new time with the accepted layers frozen.
        {
            const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
            Kokkos::deep_copy(exec, target.values, SolverReal{0});
            Kokkos::deep_copy(exec, target.rhs, SolverReal{0});
            status.begin_async();

            with_time_term([&](auto weights, auto history) { temporal.add_system_async(target, status, weights, mass, history); });
            if (convection) convection->add_system_async(target, boundary, status, flux, cv::NoReconstruction{}, cv::FixedValueInflow{}, next);
            if (diffusion) diffusion->add_system_async(target, boundary, status, law, next);
            sources.add_async(target.rhs, status, source);
            (void)status.check(comm);
            Kokkos::deep_copy(exec, base_rhs, target.rhs);
        }
        matrix.commit();
        rhs.commit();

        const auto full_residual = [&] {
            Kokkos::deep_copy(exec, residual, Real{0});
            status.begin_async();
            with_time_term([&](auto weights, auto history) {
                temporal.add_residual_async(q, residual, status, weights, mass, history);
            });
            with_convection([&](auto reconstruction) {
                convection->add_residual_async(q, residual, boundary, status, flux, reconstruction, cv::FixedValueInflow{}, next);
            });
            if (diffusion) {
                if (config.nonorthogonal_correction)
                    diffusion->add_residual_async(q, residual, boundary, status, law, correction, next);
                else diffusion->add_residual_async(q, residual, boundary, status, law, next);
            }
            sources.add_async(residual, status, source);
            (void)status.check(comm);
            return global_norm(residual, exec, comm);
        };

        prepare_iterate(next);
        double norm = full_residual();
        const double tolerance = std::max(config.outer.absolute_tolerance, config.outer.relative_tolerance *
            std::max(norm, global_norm(base_rhs, exec, comm)));
        int outer = 0;
        for (; norm > tolerance && outer < config.outer.maximum_iterations; ++outer) {
            // Restore the frozen RHS before adding the current explicit corrections.
            {
                const auto b = Kokkos::subview(rhs.overwrite(exec), Kokkos::ALL(), 0);
                Kokkos::deep_copy(exec, b, base_rhs);
                if (reconstruct || config.nonorthogonal_correction) {
                    status.begin_async();
                    if (reconstruct) with_convection([&](auto reconstruction) {
                        convection->add_correction_async(b, boundary, status, flux, reconstruction, next);
                    });
                    if (config.nonorthogonal_correction)
                        diffusion->add_correction_async(b, boundary, status, law, correction, next);
                    (void)status.check(comm);
                }
            }
            rhs.commit();

            solution.copy_from_component(q, exec);
            const auto solved = linear.solve(rhs, solution);
            mpi::require(solved.converged, "Scalar transport: linear solve did not converge");
            if (relaxation == Real{1}) solution.copy_to_component(q, exec);
            else {
                const auto candidate = solution.read(exec);
                Kokkos::parallel_for("scalar_transport/relax", Policy(exec, 0, owned), KOKKOS_LAMBDA(std::size_t i) {
                    const Real old = static_cast<Real>(q(i));
                    q(i) = static_cast<StorageReal>(old + relaxation*(static_cast<Real>(candidate(i, 0))-old));
                });
                solution.commit();
            }
            prepare_iterate(next);
            norm = full_residual();
            if (detailed)
                std::fprintf(stderr, "  step=%d outer=%d linear=%d residual=%.6g target=%.6g\n",
                    step + 1, outer + 1, solved.iterations, norm, tolerance);
        }
        mpi::require(norm <= tolerance, "Scalar transport: outer iteration did not converge");
        if (bdf2) Kokkos::deep_copy(exec, q_nm1, q_n);
        if (!steady) Kokkos::deep_copy(exec, q_n, q_owned);
        time = next;
        if (detailed)
            std::fprintf(stderr, "  accepted step=%d t=%.8g outer=%d residual=%.6g\n",
                step + 1, static_cast<double>(time), outer, norm);
    }

    exec.fence("scalar transport: solution ready");
    report_solution(options, config, host, q_owned, time, comm);
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