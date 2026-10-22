// Steady passive-scalar transport in a bar; both spatial operators add to one CSR system.
#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>

#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Scalar = krona::SolverReal;
namespace bc = krona::boundary::scalar;
namespace ts = krona::transport::scalar;
namespace cv = ts::convection;
namespace df = ts::diffusion;
namespace vtk = krona::io::vtk;
namespace rec = krona::numerics::reconstruction;
namespace grad = krona::numerics::gradient;

constexpr double length = 1.0;
constexpr double cross_section = 0.25*0.125;
constexpr double density = 1.0;
constexpr double diffusion_coefficient = 0.1;
constexpr double default_peclet = 10.0;
constexpr int maximum_corrections = 200;
constexpr double correction_relaxation = 0.8;
constexpr bool reduced_precision = std::is_same_v<krona::StorageReal, float> ||
    std::is_same_v<Real, float> || std::is_same_v<Scalar, float>;
constexpr double nonlinear_tolerance = reduced_precision ? 5.0e-5 : 1.0e-8;

/// @brief Application-owned reconstruction workspace; no changes to library operators are needed.
class HighOrderState {
    using Space = typename Exec::memory_space;
    rec::MeshStencil<Space, Real> stencil_;
    grad::GradientOperator<Exec> gradient_;
    krona::fields::FieldStorage<krona::StorageReal, 3, Exec> gradients_;
    krona::fields::FieldStorage<krona::StorageReal, 1, Exec> factors_;
    krona::halo::Exchange<krona::StorageReal, Exec> halo_;
    Exec exec_;
public:
    HighOrderState(const cv::ConvectionPlan<Exec, Real>& plan,
        krona::mesh::compute::AuxGeometryView<Space> geometry,
        krona::mesh::compute::HaloPlanView halo_plan, krona::mpi::CommView comm, const Exec& exec)
        : stencil_(plan.mesh_view(), plan.connectivity_view()),
          gradient_(plan.mesh_view(), plan.connectivity_view(), geometry,
              {.scheme = grad::Scheme::GreenGaussCell, .interpolation = grad::Interpolation::Arithmetic}, exec),
          gradients_("scalar_gradient", static_cast<std::size_t>(plan.mesh_view().sizes.n_cells),
              krona::fields::FieldLocation::Custom, exec),
          factors_("scalar_limiter", static_cast<std::size_t>(plan.mesh_view().sizes.n_cells),
              krona::fields::FieldLocation::Custom, exec),
          halo_(plan.mesh_view(), halo_plan, comm, 1, exec, krona::halo::Transfer::HostStaged), exec_(exec)
    {
        halo_.add_field(gradients_.view());
        halo_.add_field(factors_.view());
        halo_.prepare();
    }

    /// @brief Scalar MPI ghosts and physical face values must already be current on exec_.
    template<class Field>
    void update_async(Field scalar)
    {
        gradient_.apply(scalar, gradients_.view());
        const auto input = rec::data(rec::field_pack(scalar), rec::gradient_pack(gradients_.view()));
        rec::compute_limiters_async<Real>(exec_, stencil_, input, rec::field_pack(factors_.view()),
            rec::BarthJespersen{}, rec::SecondOrderUpwind{});
        // Owned factors need owned gradients and current scalar samples, not remote gradients.
        // Exchange the three gradient components and the factor together only after both are produced.
        halo_.start();
        halo_.finish_async();
    }

    template<class Field>
    [[nodiscard]] auto reconstruction(Field scalar, const cv::ConvectionPlan<Exec, Real>& plan) const
    {
        return cv::make_reconstruction(plan, rec::LimitedLinear{},
            rec::data(rec::field_pack(scalar), rec::gradient_pack(gradients_.view()),
                rec::field_pack(factors_.view())));
    }
};

/// @brief Integrated mass/time: rho*u dot n * area on inner/open faces, exactly zero on side walls.
/// The closed local patch range expresses impermeability independently of normal roundoff.
struct PrescribedMassFlow {
    Real rho_u = 0;
    krona::LocalIndex impermeable_begin = 0, impermeable_end = 0;
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const
    {
        if (c.face >= impermeable_begin && c.face < impermeable_end) return Real{0};
        return rho_u*c.nx*c.area;
    }
};

/// @brief Exact solution for phi(0)=0, phi(L)=1; negative Pe reverses the prescribed flow.
/// Host-only reference arithmetic avoids overflowing exp(Pe) at large positive Pe.
struct ExactProfile {
    double peclet = default_peclet;

    [[nodiscard]] double value(double x) const
    {
        const double t = std::clamp(x/length, 0.0, 1.0);
        if (peclet == 0) return t;
        if (peclet < 0) return 1.0-positive_value(1.0-t, -peclet);
        return positive_value(t, peclet);
    }

    /// @brief Exact volume average for the supplied axis-aligned hexahedral cells.
    [[nodiscard]] double average(double x0, double x1) const
    {
        const double a = std::clamp(x0/length, 0.0, 1.0);
        const double b = std::clamp(x1/length, 0.0, 1.0);
        krona::mpi::require(b > a, "Scalar transport app: reference needs a positive cell x-width");
        if (peclet == 0) return (a+b)*0.5;
        if (peclet < 0) return 1.0-positive_average(1.0-b, 1.0-a, -peclet);
        return positive_average(a, b, peclet);
    }
private:
    static double positive_value(double t, double p)
    {
        return std::exp(p*(t-1.0))*(-std::expm1(-p*t))/(-std::expm1(-p));
    }

    static double positive_average(double a, double b, double p)
    {
        // The small-Pe expansion avoids cancellation in (average(exp(p*x))-1).
        if (p < 1.0e-4) {
            const double q = (a+b)*0.5, h = (b-a)*0.5;
            const double m2 = q*q+h*h/3.0;
            const double m3 = q*q*q+q*h*h;
            const double m4 = q*q*q*q+2.0*q*q*h*h+h*h*h*h/5.0;
            return q+p*(m2-q)*0.5+p*p*(2.0*m3-3.0*m2+q)/12.0+
                   p*p*p*(m4-2.0*m3+m2)/24.0;
        }
        const long double pe = p, lo = a, hi = b;
        const long double width = pe*(hi-lo);
        const long double exponential_mean = std::exp(pe*(hi-1.0L))*(-std::expm1(-width))/width;
        return static_cast<double>((exponential_mean-std::exp(-pe))/(-std::expm1(-pe)));
    }
};

void usage(std::FILE* stream)
{
    std::fputs(
        "KRONA steady convection-diffusion example\n\n"
        "Usage:\n"
        "  mpirun -np N krona_scalar_convection_diffusion <mesh.h5> <boundary.toml> <output-dir> [Pe] [scheme]\n"
        "  krona_scalar_convection_diffusion --help\n\n"
        "Use examples/transport/scalar/exponential_bar/mesh.toml and boundary.toml.\n"
        "N must match the saved mesh partition. Kokkos accepts --kokkos-* options.\n"
        "The case fixes L=1, rho=1, Gamma=0.1, source=0, phi(0)=0 and phi(1)=1.\n"
        "Signed Pe defaults to 10; Pe=0 gives pure diffusion, Pe<0 reverses the flow.\n"
        "Scheme: limited (default, Barth-Jespersen limited linear) or upwind.\n"
        "Limited mode uses a frozen upwind matrix and iterative RHS corrections.\n"
        "Diffusion is orthogonal; convergence checks the full spatial residual.\n", stream);
}

void broadcast_text(std::string& text, krona::mpi::CommView comm)
{
    if (comm.rank() == 0 && text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        krona::mpi::fatal("Scalar transport app: text exceeds MPI INT_MAX");
    int count = comm.rank() == 0 ? static_cast<int>(text.size()) : 0;
    krona::mpi::check(MPI_Bcast(&count, 1, MPI_INT, 0, comm.native()), "MPI_Bcast(text length)");
    if (comm.rank() != 0) text.resize(static_cast<std::size_t>(count));
    krona::mpi::check(MPI_Bcast(text.data(), count, MPI_CHAR, 0, comm.native()), "MPI_Bcast(text)");
}

[[nodiscard]] bc::Configuration load_boundary(const std::filesystem::path& path, krona::mpi::CommView comm)
{
    std::string text;
    if (comm.rank() == 0) {
        std::ifstream input{path, std::ios::binary};
        if (!input) krona::mpi::fatal("Scalar transport app: cannot open boundary file");
        text.assign(std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{});
        if (input.bad()) krona::mpi::fatal("Scalar transport app: cannot read boundary file");
    }
    broadcast_text(text, comm);
    return bc::parse_config(text, "scalar");
}

template<class View>
double global_l2(View values, krona::mpi::CommView comm, const Exec& exec)
{
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    double local = 0, global = 0;
    Kokkos::parallel_reduce("transport/owned_l2", Policy(exec, 0, values.extent(0)),
        KOKKOS_LAMBDA(std::size_t i, double& sum) {
            const double value = static_cast<double>(values(i)); sum += value*value;
        }, local);
    krona::mpi::check(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm.native()),
        "MPI_Allreduce(owned L2)");
    krona::mpi::require(std::isfinite(global), "Scalar transport app: nonfinite residual norm");
    return std::sqrt(global);
}

template<class Field, class Boundary, class Law, class Output, class Reconstruction>
double spatial_residual(const cv::ConvectionOperator<Exec, Real>& convection,
    const df::DiffusionOperator<Exec, Real>& diffusion, Field field, const Boundary& boundary,
    Law law, PrescribedMassFlow mass_flow, Output output, ts::AssemblyStatus<Exec>& status,
    Reconstruction reconstruction, krona::mpi::CommView comm, const Exec& exec)
{
    Kokkos::deep_copy(exec, output, Real{0});
    status.begin_async();
    convection.add_residual_async(field, output, boundary, status, mass_flow, reconstruction);
    diffusion.add_residual_async(field, output, boundary, status, law);
    (void)status.check(comm);
    return global_l2(output, comm, exec);
}

// NVCC cannot host an extended lambda in solve_deferred: its template arguments
// include local host-lambda types. Keep this device kernel at namespace scope.
template<class Field, class Candidate>
struct RelaxOwnedSolution {
    Field field;
    Candidate candidate;
    Real omega;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        field(i) = static_cast<typename Field::non_const_value_type>(
            (Real{1}-omega)*static_cast<Real>(field(i))+omega*static_cast<Real>(candidate(i, 0)));
    }
};

struct IterationResult {
    krona::linalg::SolveResult linear;
    bool converged = false;
    int corrections = 0, linear_solves = 0, total_krylov_iterations = 0;
    double residual_norm = 0, residual_reference = 0, residual_threshold = 0;
    [[nodiscard]] double relative_residual() const
    { return residual_reference > 0 ? residual_norm/residual_reference : residual_norm; }
};

/// @brief Application orchestration. Callbacks refresh reconstruction, replace the RHS, and
/// measure the full current spatial operator; none of them rebuilds the matrix.
template<class Matrix, class Field, class Prepare, class Correct, class Residual>
IterationResult solve_deferred(const Matrix& matrix, krona::linalg::Vector<Scalar, Exec>& rhs,
    krona::linalg::Vector<Scalar, Exec>& solution, krona::linalg::Solver<Matrix>& solver,
    Field field, bool limited, Prepare prepare, Correct correct_rhs, Residual measure_residual,
    krona::mpi::CommView comm, const Exec& exec)
{
    IterationResult result;
    const auto revision = matrix.revision();
    result.residual_reference = static_cast<double>(rhs.native()->norm2());
    result.linear = solver.solve(rhs, solution); // Upwind solution is the initial nonlinear iterate.
    result.linear_solves = 1;
    result.total_krylov_iterations = result.linear.iterations;
    if (!result.linear.converged) return result;
    solution.copy_to_component(field, exec);
    solution.commit();
    prepare();
    result.residual_norm = measure_residual();
    if (result.residual_reference == 0) result.residual_reference = result.residual_norm;
    result.residual_threshold = nonlinear_tolerance*result.residual_reference;
    if (comm.rank() == 0) {
        std::printf("outer=0 full_relative_residual=%.6e (upwind initialization)\n", result.relative_residual());
        std::fflush(stdout);
    }
    if (!limited) {
        result.converged = result.residual_norm <= result.residual_threshold;
        return result;
    }
    // At least one corrected solve is performed, even if the upwind field already meets
    // the absolute threshold on a very fine grid. No correction is repeatedly accumulated.
    for (int iteration = 1; iteration <= maximum_corrections; ++iteration) {
        correct_rhs();
        krona::mpi::require(matrix.revision() == revision, "Scalar transport app: frozen matrix was modified");
        solution.copy_from_component(field, exec);
        solution.commit();
        result.linear = solver.solve(rhs, solution);
        ++result.linear_solves;
        result.total_krylov_iterations += result.linear.iterations;
        result.corrections = iteration;
        if (!result.linear.converged) return result;
        if (result.linear.iterations_started) {
            const auto candidate = solution.read(exec);
            const Real omega = static_cast<Real>(correction_relaxation);
            using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
            using Kernel = RelaxOwnedSolution<Field, std::remove_cvref_t<decltype(candidate)>>;
            Kokkos::parallel_for("transport/relax_owned_solution", Policy(exec, 0, solution.local_size()),
                Kernel{field, candidate, omega});
            solution.commit();
        }
        // Check fresh gradients/limiters of the relaxed field, not those frozen for the last solve.
        prepare();
        result.residual_norm = measure_residual();
        if (comm.rank() == 0) {
            std::printf("outer=%d linear_iterations=%d full_relative_residual=%.6e\n",
                iteration, result.linear.iterations, result.relative_residual());
            std::fflush(stdout);
        }
        if (result.residual_norm <= result.residual_threshold) {
            result.converged = true;
            break;
        }
    }
    return result;
}

/// @brief Matrix-free diagnostics exercise the same two operators after publishing MPI ghosts.
template<class Convection, class Diffusion, class Field, class Boundary, class Law, class Output,
    class Reconstruction = cv::NoReconstruction>
void report_balance(const Convection& convection, const Diffusion& diffusion, Field field,
    const Boundary& boundary, Law law, PrescribedMassFlow mass_flow, Output residual, ts::AssemblyStatus<Exec>& status,
    krona::mpi::CommView comm, const Exec& exec, Reconstruction reconstruction = {})
{
    const auto count = static_cast<std::size_t>(convection.plan().mesh_view().sizes.n_own);
    Kokkos::deep_copy(exec, residual, Real{0});
    status.begin_async();
    convection.add_residual_async(field, residual, boundary, status, mass_flow, reconstruction);
    (void)status.check(comm);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    double local_convection = 0;
    Kokkos::parallel_reduce("transport/sum_convective_residual", Policy(exec, 0, count),
        KOKKOS_LAMBDA(std::size_t i, double& sum) { sum += static_cast<double>(residual(i)); }, local_convection);
    double convective_residual_sum = 0;
    krona::mpi::check(MPI_Allreduce(&local_convection, &convective_residual_sum, 1, MPI_DOUBLE,
        MPI_SUM, comm.native()), "MPI_Allreduce(convective flux)");

    // Preserve the convection residual and add diffusion: r = -(outward convective + diffusive fluxes).
    status.begin_async();
    diffusion.add_residual_async(field, residual, boundary, status, law);
    (void)status.check(comm);
    double local_square = 0;
    Kokkos::parallel_reduce("transport/residual_norm", Policy(exec, 0, count),
        KOKKOS_LAMBDA(std::size_t i, double& sum) {
            const double r = static_cast<double>(residual(i)); sum += r*r;
        }, local_square);
    double global_square = 0;
    krona::mpi::check(MPI_Allreduce(&local_square, &global_square, 1, MPI_DOUBLE,
        MPI_SUM, comm.native()), "MPI_Allreduce(matrix-free residual)");
    const auto diffusive = diffusion.balance(field, boundary, law);
    const double convective_flux = -convective_residual_sum;
    const double defect = convective_flux+diffusive.outward_boundary_flux;
    // A case reference scale stays meaningful when each net flux vanishes (notably Pe=0).
    const double scale = cross_section*(std::abs(static_cast<double>(mass_flow.rho_u))+diffusion_coefficient/length);
    if (comm.rank() == 0)
        std::printf("outward scalar flux: convection=%.9e diffusion=%.9e total=%.9e\n"
                    "relative_balance_defect=%.6e matrix_free_residual_l2=%.6e\n",
            convective_flux, diffusive.outward_boundary_flux, defect,
            std::abs(defect)/scale, std::sqrt(global_square));
}

[[nodiscard]] int run(const std::array<std::string, 3>& paths, double requested_peclet,
    bool limited, krona::mpi::CommView comm)
{
    // All geometry/field owners stay alive through kernels, halo exchange and host output.
    const auto host_mesh = krona::io::solver_mesh::import_mesh_hdf5(paths[0], comm);
    Exec exec{};
    const krona::mesh::compute::MeshStorage<Exec> mesh{
        host_mesh, {.data = krona::mesh::compute::MeshData::FiniteVolume}, exec};
    const auto host_connectivity = krona::mesh::build_aux_connectivity(host_mesh, krona::mesh::AuxConnType::CellFaces);
    const auto host_geometry = krona::mesh::build_aux_geometry(host_mesh,
        krona::mesh::AuxGeomType::FaceNormalDistance | krona::mesh::AuxGeomType::FaceAreaOverNormalDistance);
    const krona::mesh::compute::AuxConnectivityStorage<Exec> connectivity{host_mesh, host_connectivity, exec};
    const krona::mesh::compute::AuxGeometryStorage<Exec> geometry{host_mesh, host_geometry, exec};
    const df::DiffusionPlan<Exec, Real> diffusion_plan{
        host_mesh, mesh.view(), connectivity.view(), geometry.view(), comm, exec};
    const cv::ConvectionPlan<Exec, Real> convection_plan{host_mesh, mesh.view(), connectivity.view(), comm, exec};
    const df::DiffusionOperator<Exec, Real> diffusion{diffusion_plan};
    const cv::ConvectionOperator<Exec, Real> convection{convection_plan};
    const krona::linalg::FaceAssemblyPlan<Exec> assembly{host_mesh, comm, exec};
    const bc::Manager<Exec, Real> boundary{host_mesh, mesh.view(), load_boundary(paths[1], comm), exec};

    krona::fields::FieldStorage<krona::StorageReal, 1, Exec> scalar{
        "scalar", boundary.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    scalar.fill(exec, krona::StorageReal{0});
    const auto field = scalar.view().component(0);
    krona::halo::Exchange<krona::StorageReal, Exec> halo{mesh.view(),
        {host_mesh.nb_ranks, host_mesh.send_offsets, host_mesh.recv_offsets}, comm, 0, exec,
        krona::halo::Transfer::HostStaged};
    halo.add_field(scalar.view());
    halo.prepare();
    std::unique_ptr<HighOrderState> high_order;
    if (limited) high_order = std::make_unique<HighOrderState>(
        convection_plan, geometry.view(), mesh.halo_plan(), comm, exec);

    const Real gamma = static_cast<Real>(diffusion_coefficient);
    const Real rho_u = static_cast<Real>(requested_peclet*static_cast<double>(gamma)/length);
    krona::mpi::require(std::isfinite(rho_u), "Scalar transport app: mass flow is not representable in ComputeReal");
    const auto patches = boundary.catalog().patches();
    krona::mpi::require(patches.size() == 3,
        "Scalar transport app: expected end patches 0/1 and impermeable side patch 2");
    const auto side = patches[2].range;
    const PrescribedMassFlow mass_flow{rho_u, side.begin, side.end};
    const ExactProfile exact{static_cast<double>(rho_u)*length/static_cast<double>(gamma)};
    const auto law = krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{gamma}};
    krona::linalg::CsrMatrix<Scalar, Exec> matrix{assembly.pattern()};
    krona::linalg::Vector<Scalar, Exec> rhs{matrix.distribution()}, solution{matrix.distribution()};
    const auto owned_count = static_cast<std::size_t>(host_mesh.n_own);
    Kokkos::View<Scalar*, typename Exec::memory_space> base_rhs{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "transport/base_rhs"), limited ? owned_count : 0};
    Kokkos::View<Real*, typename Exec::memory_space> residual{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "transport/full_residual"), owned_count};
    ts::AssemblyStatus<Exec> status{exec};
    krona::linalg::SolverOptions options;
    options.method = "GMRES";
    options.preconditioner.type = "AMG";
    options.relative_tolerance = reduced_precision ? 1.0e-5 : 1.0e-10;
    options.maximum_iterations = 2000;
    options.restart_length = 100;
    krona::linalg::Solver<decltype(matrix)> solver{matrix, options};

    exec.fence("Scalar transport app: setup complete");
    krona::mpi::check(MPI_Barrier(comm.native()), "MPI_Barrier(before assembly)");
    const double start = MPI_Wtime();
    const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
    // Reset exactly once. Both operators add to the same native coefficient and RHS views.
    Kokkos::deep_copy(exec, target.values, Scalar{0});
    Kokkos::deep_copy(exec, target.rhs, Scalar{0});
    status.begin_async();
    diffusion.add_system_async(target, boundary, status, law);
    convection.add_system_async(target, boundary, status, mass_flow);
    if (limited) Kokkos::deep_copy(exec, base_rhs, target.rhs);
    // Source density is zero, so no source contribution is needed.
    (void)status.check(comm);
    matrix.commit(); rhs.commit();
    solution.copy_from_component(field, exec);
    solution.commit();
    const double assembled = MPI_Wtime();
    const auto prepare = [&] {
        halo.start();
        halo.finish_async();
        boundary.apply_async(scalar.view(), law);
        if (high_order) high_order->update_async(scalar.view());
    };
    const auto correct_rhs = [&] {
        const auto local_rhs = rhs.overwrite(exec);
        const auto component = Kokkos::subview(local_rhs, Kokkos::ALL(), 0);
        Kokkos::deep_copy(exec, component, base_rhs);
        status.begin_async();
        convection.add_correction_async(component, boundary, status, mass_flow,
            high_order->reconstruction(scalar.view(), convection_plan));
        (void)status.check(comm);
        rhs.commit();
    };
    const auto measure_residual = [&] {
        if (high_order)
            return spatial_residual(convection, diffusion, field, boundary, law, mass_flow, residual, status,
                high_order->reconstruction(scalar.view(), convection_plan), comm, exec);
        return spatial_residual(convection, diffusion, field, boundary, law, mass_flow, residual, status,
            cv::NoReconstruction{}, comm, exec);
    };
    const auto result = solve_deferred(matrix, rhs, solution, solver, field, limited,
        prepare, correct_rhs, measure_residual, comm, exec);
    const double solved = MPI_Wtime();
    std::array<double, 2> local_times{assembled-start, solved-assembled}, global_times{};
    krona::mpi::check(MPI_Allreduce(local_times.data(), global_times.data(), 2, MPI_DOUBLE,
        MPI_MAX, comm.native()), "MPI_Allreduce(assembly/solve duration)");
    if (comm.rank() == 0)
        std::printf("execution=%s ranks=%d cells=%llu\n"
                    "Pe=%.9g rho=%.9g U=%.9g Gamma=%.9g scheme=%s\n"
                    "converged=%s corrections=%d linear_solves=%d total_krylov_iterations=%d\n"
                    "last_linear_relative_residual=%.6e full_relative_residual=%.6e full_threshold=%.6e\n"
                    "preconditioner_builds=%llu\n"
                    "assembly_seconds=%.6f transport_solve_seconds=%.6f (includes AMG/outer iterations)\n",
            Exec::name(), comm.size(), static_cast<unsigned long long>(host_mesh.n_cells_g),
            exact.peclet, density, static_cast<double>(rho_u)/density, static_cast<double>(gamma),
            limited ? "limited" : "upwind", result.converged ? "yes" : "no", result.corrections,
            result.linear_solves, result.total_krylov_iterations,
            result.linear.relative_residual, result.relative_residual(), result.residual_threshold,
            static_cast<unsigned long long>(solver.preconditioner().generation()),
            global_times[0], global_times[1]);
    if (!result.converged) return EXIT_FAILURE;

    // The authoritative solution is the relaxed field; the linear vector may hold an unrelaxed candidate.
    exec.fence("Scalar transport app: published field ready");
    if (high_order)
        report_balance(convection, diffusion, field, boundary, law, mass_flow, residual, status, comm, exec,
            high_order->reconstruction(scalar.view(), convection_plan));
    else report_balance(convection, diffusion, field, boundary, law, mass_flow, residual, status, comm, exec);

    auto values = scalar.create_host_buffer();
    Kokkos::deep_copy(exec, values, scalar.view().kokkos_view());
    exec.fence("Scalar transport app: visualization snapshot ready");
    const auto owned = static_cast<std::size_t>(host_mesh.n_own);
    const auto cells = static_cast<std::size_t>(host_mesh.n_cells);
    const auto first_boundary = static_cast<std::size_t>(host_mesh.n_inner_faces);
    const auto boundary_faces = static_cast<std::size_t>(host_mesh.n_faces)-first_boundary;
    std::vector<double> reference(owned), error(owned), boundary_reference(boundary_faces), boundary_error(boundary_faces);
    std::array<double, 3> sums{};
    double maximum = 0, minimum_value = std::numeric_limits<double>::infinity(), maximum_value = -minimum_value;
    for (std::size_t i = 0; i < owned; ++i) {
        krona::mpi::require(host_mesh.cell_type[i] == krona::mesh::CellType::HEXA,
            "Scalar transport app: exact cell averages require the supplied Cartesian HEXA mesh");
        double x0 = std::numeric_limits<double>::infinity(), x1 = -x0;
        for (auto k = host_mesh.cell_nodes_offsets[i]; k < host_mesh.cell_nodes_offsets[i+1]; ++k) {
            const double x = static_cast<double>(host_mesh.node_x[static_cast<std::size_t>(host_mesh.cell_nodes[k])]);
            x0 = std::min(x0, x); x1 = std::max(x1, x);
        }
        reference[i] = exact.average(x0, x1);
        error[i] = static_cast<double>(values(i))-reference[i];
        if (!std::isfinite(error[i])) krona::mpi::fatal("Scalar transport app: nonfinite scalar");
        const double volume = static_cast<double>(host_mesh.cell_volume[i]);
        sums[0] += volume; sums[1] += volume*std::abs(error[i]); sums[2] += volume*error[i]*error[i];
        maximum = std::max(maximum, std::abs(error[i]));
        minimum_value = std::min(minimum_value, static_cast<double>(values(i)));
        maximum_value = std::max(maximum_value, static_cast<double>(values(i)));
    }
    for (std::size_t b = 0; b < boundary_faces; ++b) {
        boundary_reference[b] = exact.value(static_cast<double>(host_mesh.face_centroid_x[first_boundary+b]));
        boundary_error[b] = static_cast<double>(values(cells+b))-boundary_reference[b];
    }
    std::array<double, 3> global{};
    const std::array<double, 3> local_extrema{maximum, -minimum_value, maximum_value};
    std::array<double, 3> global_extrema{};
    krona::mpi::check(MPI_Allreduce(sums.data(), global.data(), 3, MPI_DOUBLE, MPI_SUM, comm.native()),
        "MPI_Allreduce(error integrals)");
    krona::mpi::check(MPI_Allreduce(local_extrema.data(), global_extrema.data(), 3, MPI_DOUBLE, MPI_MAX, comm.native()),
        "MPI_Allreduce(maximum error)");
    krona::mpi::require(global[0] > 0, "Scalar transport app: mesh has no positive owned volume");
    if (comm.rank() == 0)
        std::printf("scalar error vs exact cell averages: L1=%.6e L2=%.6e Linf=%.6e\n"
                    "owned scalar range: [%.9e, %.9e]\n",
            global[1]/global[0], std::sqrt(global[2]/global[0]), global_extrema[0], -global_extrema[1], global_extrema[2]);
    const std::array cell_fields{
        vtk::field("scalar", std::span{values.data(), owned}),
        vtk::field("scalar_exact", std::span{reference}), vtk::field("scalar_error", std::span{error})};
    auto* boundary_values = values.data();
    if (cells != 0) boundary_values += cells;
    const std::array boundary_fields{
        vtk::field("scalar", std::span{boundary_values, boundary_faces}),
        vtk::field("scalar_exact", std::span{boundary_reference}),
        vtk::field("scalar_error", std::span{boundary_error})};
    vtk::write_vtu(host_mesh, paths[2], "exponential_bar", comm,
        {.cell = cell_fields, .boundary = boundary_fields});
    if (comm.rank() == 0) std::printf("VTK datasets written to %s (stem: exponential_bar)\n", paths[2].c_str());
    return EXIT_SUCCESS;
}

[[nodiscard]] int execute(int argc, char* const* argv, krona::mpi::CommView comm)
{
    int action = 0;
    if (comm.rank() == 0) {
        if (argc == 1 || (argc == 2 && std::string_view{argv[1]} == "--help")) action = 1;
        else if (argc < 4 || argc > 6) action = 2;
    }
    krona::mpi::check(MPI_Bcast(&action, 1, MPI_INT, 0, comm.native()), "MPI_Bcast(CLI action)");
    if (action != 0) {
        if (comm.rank() == 0) usage(action == 1 ? stdout : stderr);
        return action == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    std::array<std::string, 3> paths;
    std::string peclet_text, scheme;
    if (comm.rank() == 0) {
        for (std::size_t i = 0; i < paths.size(); ++i) paths[i] = argv[i+1];
        peclet_text = argc >= 5 ? argv[4] : "10";
        scheme = argc == 6 ? argv[5] : "limited";
    }
    for (auto& path : paths) {
        broadcast_text(path, comm);
        krona::mpi::require(!path.empty(), "Scalar transport app: empty path argument");
    }
    broadcast_text(peclet_text, comm);
    broadcast_text(scheme, comm);
    krona::mpi::require(scheme == "limited" || scheme == "upwind",
        "Scalar transport app: scheme must be limited or upwind");
    char* end = nullptr;
    errno = 0;
    const double peclet = std::strtod(peclet_text.c_str(), &end);
    krona::mpi::require(end != peclet_text.c_str() && end == peclet_text.c_str()+peclet_text.size() &&
        errno != ERANGE && std::isfinite(peclet), "Scalar transport app: Pe must be a finite number");
    return run(paths, peclet, scheme == "limited", comm);
}
} // namespace

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    try {
        Kokkos::ScopeGuard kokkos{argc, argv};
        Tpetra::ScopeGuard tpetra{&argc, &argv};
        try { return execute(argc, argv, krona::mpi::CommView::world()); }
        catch (const std::exception& error) { krona::mpi::fatal(error.what()); }
        catch (...) { krona::mpi::fatal("Unhandled scalar transport application exception"); }
    } catch (const std::exception& error) { krona::mpi::fatal(error.what()); }
    catch (...) { krona::mpi::fatal("Scalar transport runtime initialization/finalization failed"); }
    return EXIT_FAILURE;
}