// Fully three-dimensional Gaussian transport; BDF2 and limited linear reconstruction by default.
#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/transport/scalar/transient/operator.hpp>

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
#include <locale>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Scalar = krona::SolverReal;
using Storage = krona::StorageReal;
using Space = typename Exec::memory_space;
namespace bc = krona::boundary::scalar;
namespace ts = krona::transport::scalar;
namespace cv = ts::convection;
namespace df = ts::diffusion;
namespace tt = ts::transient;
namespace nt = krona::numerics::time;
namespace vtk = krona::io::vtk;
namespace rec = krona::numerics::reconstruction;
namespace grad = krona::numerics::gradient;

constexpr double length = 1.0;
constexpr double cross_section = 1.0;
constexpr double density = 1.0;
constexpr double velocity = 1.0;
constexpr double diffusion_coefficient = 0.005;
constexpr int maximum_corrections = 200;
constexpr double correction_relaxation = 0.8;
constexpr bool reduced_precision = std::is_same_v<Storage, float> ||
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
        krona::mesh::device::AuxGeometryView<Space> geometry,
        krona::mesh::host::HaloPlanView halo_plan, krona::mpi::CommView comm, const Exec& exec)
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

/// @brief Signed integrated flux for a uniform prescribed three-dimensional velocity.
struct PrescribedMassFlow {
    Real rho_u = static_cast<Real>(density*velocity);
    Real rho_v = static_cast<Real>(density*0.4);
    Real rho_w = static_cast<Real>(density*0.2);
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const
    { return (rho_u*c.nx+rho_v*c.ny+rho_w*c.nz)*c.area; }
};

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
/// measure the full current space-time operator; none of them rebuilds the matrix.
template<class Matrix, class Field, class Prepare, class Correct, class Residual>
IterationResult solve_deferred(const Matrix& matrix, krona::linalg::Vector<Scalar, Exec>& rhs,
    krona::linalg::Vector<Scalar, Exec>& solution, krona::linalg::Solver<Matrix>& solver,
    Field field, bool limited, bool verbose, Prepare prepare, Correct correct_rhs, Residual measure_residual,
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
    if (verbose && comm.rank() == 0) {
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
        if (verbose && comm.rank() == 0) {
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


/// @brief Separable isotropic 3-D Gaussian; host reference and exact cell integrals.
struct ExactGaussian {
    double speed = velocity, speed_y = 0.4, speed_z = 0.2;
    double diffusivity = diffusion_coefficient/density;
    static constexpr double initial_sigma = 0.08;
    static constexpr std::array<double,3> initial_centre{0.25,0.35,0.4};

    [[nodiscard]] double value(double x,double y,double z,double time) const
    {
        const double variance=initial_sigma*initial_sigma+2.0*diffusivity*time;
        const double dx=x-initial_centre[0]-speed*time;
        const double dy=y-initial_centre[1]-speed_y*time;
        const double dz=z-initial_centre[2]-speed_z*time;
        const double amplitude=initial_sigma/std::sqrt(variance);
        return amplitude*amplitude*amplitude*std::exp(-(dx*dx+dy*dy+dz*dz)/(2.0*variance));
    }
    [[nodiscard]] double average(const std::array<double,3>& lo,
        const std::array<double,3>& hi,double time) const
    {
        const double scale=std::sqrt(2.0*(initial_sigma*initial_sigma+2.0*diffusivity*time));
        const std::array<double,3> speeds{speed,speed_y,speed_z};
        double result=1.0;
        for(std::size_t d=0;d<3;++d){
            const double centre=initial_centre[d]+speeds[d]*time;
            const double a=(lo[d]-centre)/scale,b=(hi[d]-centre)/scale;
            const double difference=a>=0?std::erfc(a)-std::erfc(b)
                :b<=0?std::erfc(-b)-std::erfc(-a):std::erf(b)-std::erf(a);
            result*=initial_sigma*std::sqrt(std::acos(-1.0)/2.0)*difference/(hi[d]-lo[d]);
        }
        return result;
    }
};

/// @brief Application-specific Dirichlet condition, accepted by the existing custom-condition API.
/// The stored physical time is shared by all consumers; no library profile extension is required.
struct GaussianDirichlet {
    Real time=0, ux=0, uy=0, uz=0, diffusivity=0;
    Real sigma=1, x0=0, y0=0, z0=0;
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real value(const Context& c) const
    {
        const Real variance=sigma*sigma+Real{2}*diffusivity*time;
        const Real dx=static_cast<Real>(c.x)-x0-ux*time;
        const Real dy=static_cast<Real>(c.y)-y0-uy*time;
        const Real dz=static_cast<Real>(c.z)-z0-uz*time;
        const Real amplitude=sigma/Kokkos::sqrt(variance);
        return amplitude*amplitude*amplitude*Kokkos::exp(-(dx*dx+dy*dy+dz*dz)/(Real{2}*variance));
    }
    template<class R>
    KOKKOS_INLINE_FUNCTION bc::Relation<R> operator()(const krona::boundary::FaceContext<R>& c,R increment=0) const
    { return bc::FixedValue<R>{static_cast<R>(value(c))}(c,increment); }
};

/// @brief Convection's explicit inflow adapter reads the same condition used by diffusion/gradients.
struct GaussianInflow {
    GaussianDirichlet profile;
    template<class Context,class Condition>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c,const Condition&) const
    { return profile.value(c); }
};

struct Settings {
    double dt = 0.005, end_time = 0.35;
    int output_every = 5;
    bool limited = true, bdf2 = true;
};

void usage(std::FILE* stream)
{
    std::fputs(
        "KRONA 3-D transient Gaussian convection-diffusion example\n\n"
        "Usage:\n"
        "  mpirun -np N krona_scalar_transient_gaussian_3d <mesh.h5> <boundary.toml> <output-dir>\n"
        "      [dt=0.005] [end_time=0.35] [output_every=5] [limited|upwind] [bdf2|euler]\n\n"
        "Use examples/transport/scalar/gaussian_3d/{mesh,boundary}.toml.\n"
        "Defaults: limited linear reconstruction with Barth-Jespersen, BDF2 after Euler startup.\n"
        "rho=1, velocity=(1,0.4,0.2), Gamma=0.005, sigma0=0.08.\n"
        "Initial centre=(0.25,0.35,0.4); cube [0,1]^3; exact spatial/time boundary profile.\n"
        "N must match the saved mesh partition. Kokkos accepts --kokkos-* options.\n", stream);
}

template<class Boundary>
void update_boundary(Boundary& boundary,GaussianInflow& inflow,const ExactGaussian& exact,double time)
{
    inflow.profile={static_cast<Real>(time),static_cast<Real>(exact.speed),static_cast<Real>(exact.speed_y),
        static_cast<Real>(exact.speed_z),static_cast<Real>(exact.diffusivity),
        static_cast<Real>(ExactGaussian::initial_sigma),static_cast<Real>(ExactGaussian::initial_centre[0]),
        static_cast<Real>(ExactGaussian::initial_centre[1]),static_cast<Real>(ExactGaussian::initial_centre[2])};
    boundary.set(0,inflow.profile);
}

template<class Field, class Boundary, class Law, class Output, class Reconstruction,
    class Derivative, class Mass, class History>
double full_residual(const cv::ConvectionOperator<Exec,Real>& convection,
    const df::DiffusionOperator<Exec,Real>& diffusion, const tt::TransientOperator<Exec,Real>& temporal,
    Field field, Output output, const Boundary& boundary, Law law, PrescribedMassFlow mass_flow,
    Reconstruction reconstruction, GaussianInflow inflow, Derivative derivative, Mass mass, History history,
    ts::AssemblyStatus<Exec>& status, krona::mpi::CommView comm, const Exec& exec)
{
    Kokkos::deep_copy(exec,output,Real{0});
    status.begin_async();
    convection.add_residual_async(field,output,boundary,status,mass_flow,reconstruction,inflow);
    diffusion.add_residual_async(field,output,boundary,status,law);
    temporal.add_residual_async(field,output,status,derivative,mass,history);
    (void)status.check(comm);
    return global_l2(output,comm,exec);
}

/// @brief Publish a time index after the existing writer has closed the referenced datasets.
void write_series(const std::filesystem::path& directory,
    const std::vector<std::pair<double,std::string>>& frames, bool boundary)
{
    std::ofstream out(directory/(boundary?"gaussian_3d_boundary.pvd":"gaussian_3d.pvd"),std::ios::trunc);
    krona::mpi::require(out.is_open(),"Gaussian app: cannot open PVD time index");
    out.imbue(std::locale::classic());
    out.precision(std::numeric_limits<double>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n  <Collection>\n";
    for(const auto& [time,stem]:frames)
        out << "    <DataSet timestep=\"" << time << "\" group=\"\" part=\"0\" file=\""
            << stem << (boundary?"_bnd.pvtu":".pvtu") << "\"/>\n";
    out << "  </Collection>\n</VTKFile>\n";
    out.close();
    krona::mpi::require(!out.fail(),"Gaussian app: cannot finish PVD time index");
}

int run(const std::array<std::string,3>& paths, Settings settings, krona::mpi::CommView comm)
{
    const auto host_mesh = krona::io::solver_mesh::import_mesh_hdf5(paths[0],comm);
    Exec exec{};
    const krona::mesh::device::MeshStorage<Exec> mesh{
        host_mesh,{.data=krona::mesh::device::MeshData::FiniteVolume},exec};
    const auto host_connectivity = krona::mesh::host::build_aux_connectivity(host_mesh,krona::mesh::AuxConnType::CellFaces);
    const auto host_geometry = krona::mesh::host::build_aux_geometry(host_mesh,
        krona::mesh::AuxGeomType::FaceNormalDistance|krona::mesh::AuxGeomType::FaceAreaOverNormalDistance);
    const krona::mesh::device::AuxConnectivityStorage<Exec> connectivity{host_connectivity,exec};
    const krona::mesh::device::AuxGeometryStorage<Exec> geometry{host_geometry,exec};
    const df::DiffusionPlan<Exec,Real> diffusion_plan{
        host_mesh,mesh.view(),connectivity.view(),geometry.view(),comm,exec};
    const cv::ConvectionPlan<Exec,Real> convection_plan{host_mesh,mesh.view(),connectivity.view(),comm,exec};
    const df::DiffusionOperator<Exec,Real> diffusion{diffusion_plan};
    const cv::ConvectionOperator<Exec,Real> convection{convection_plan};
    const krona::linalg::FaceAssemblyPlan<Exec> assembly{host_mesh,comm,exec};
    bc::Manager<Exec,Real,GaussianDirichlet> boundary{host_mesh,mesh.view(),load_boundary(paths[1],comm),exec};
    const auto patches = boundary.catalog().patches();
    krona::mpi::require(patches.size()==1,"Gaussian 3-D app: expected one exact_boundary patch containing all six sides");
    const PrescribedMassFlow mass_flow{};
    const Real gamma=static_cast<Real>(diffusion_coefficient);
    const ExactGaussian exact{static_cast<double>(mass_flow.rho_u)/density,
        static_cast<double>(mass_flow.rho_v)/density,static_cast<double>(mass_flow.rho_w)/density,
        static_cast<double>(gamma)/density};
    GaussianInflow inflow{};
    const auto law = krona::transport::IsotropicDiffusion{krona::transport::UniformCoefficient<Real>{gamma}};
    const auto owned_count = static_cast<std::size_t>(host_mesh.n_own);
    const auto cells = static_cast<std::size_t>(host_mesh.n_cells);
    const auto first_boundary = static_cast<std::size_t>(host_mesh.n_inner_faces);
    const auto boundary_count = static_cast<std::size_t>(host_mesh.n_faces)-first_boundary;
    const tt::TransientOperator<Exec,Real> temporal{owned_count,exec};
    const auto mass = tt::DensityVolumeMass{mesh.view().geometry.cell_volume,static_cast<Real>(density)};

    krona::fields::FieldStorage<Storage,1,Exec> scalar{
        "scalar",boundary.required_field_size(),krona::fields::FieldLocation::Custom,exec};
    scalar.fill(exec,Storage{0});
    const auto field = scalar.view().component(0);
    const auto owned_field = Kokkos::subview(field,std::pair<std::size_t,std::size_t>{0,owned_count});
    auto snapshot = scalar.create_host_buffer(); // Reused for all output snapshots and initialization.
    Kokkos::deep_copy(snapshot,Storage{0});
    std::vector<std::array<double,3>> lower(owned_count),upper(owned_count);
    std::vector<double> reference(owned_count),error(owned_count);
    std::vector<double> boundary_reference(boundary_count),boundary_error(boundary_count);
    const double geometry_tolerance = std::max(1.0e-10,128.0*std::numeric_limits<krona::GeometryReal>::epsilon());
    for(std::size_t i=0;i<owned_count;++i){
        krona::mpi::require(host_mesh.cell_type[i]==krona::mesh::CellType::HEXA,
            "Gaussian app: analytical cell averages require the supplied Cartesian HEXA mesh");
        const auto begin = host_mesh.cell_nodes_offsets[i], end = host_mesh.cell_nodes_offsets[i+1];
        krona::mpi::require(end-begin==8,"Gaussian app: expected eight hexahedron vertices");
        std::array<double,3> lo{INFINITY,INFINITY,INFINITY}, hi{-INFINITY,-INFINITY,-INFINITY};
        for(auto k=begin;k<end;++k){
            const auto node = static_cast<std::size_t>(host_mesh.cell_nodes[k]);
            const std::array<double,3> p{static_cast<double>(host_mesh.node_x[node]),
                static_cast<double>(host_mesh.node_y[node]),static_cast<double>(host_mesh.node_z[node])};
            for(std::size_t d=0;d<3;++d){lo[d]=std::min(lo[d],p[d]);hi[d]=std::max(hi[d],p[d]);}
        }
        for(std::size_t d=0;d<3;++d)
            krona::mpi::require(std::isfinite(lo[d])&&std::isfinite(hi[d])&&hi[d]>lo[d],"Gaussian app: invalid cell bounds");
        for(auto k=begin;k<end;++k){
            const auto node = static_cast<std::size_t>(host_mesh.cell_nodes[k]);
            const std::array<double,3> p{static_cast<double>(host_mesh.node_x[node]),
                static_cast<double>(host_mesh.node_y[node]),static_cast<double>(host_mesh.node_z[node])};
            for(std::size_t d=0;d<3;++d)
                krona::mpi::require(std::min(std::abs(p[d]-lo[d]),std::abs(p[d]-hi[d]))<=geometry_tolerance,
                    "Gaussian app: reference requires axis-aligned cells");
        }
        const double box=(hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);
        krona::mpi::require(std::abs(static_cast<double>(host_mesh.cell_volume[i])-box)<=geometry_tolerance*box,
            "Gaussian app: cell volume differs from Cartesian bounding-box volume");
        krona::mpi::require(lo[0]>=-geometry_tolerance&&hi[0]<=length+geometry_tolerance&&
            lo[1]>=-geometry_tolerance&&hi[1]<=1.0+geometry_tolerance&&
            lo[2]>=-geometry_tolerance&&hi[2]<=1.0+geometry_tolerance,"Gaussian app: unexpected domain");
        lower[i]=lo;upper[i]=hi;
        snapshot(i)=static_cast<Storage>(exact.average(lower[i],upper[i],0.0));
    }
    Kokkos::deep_copy(exec,scalar.view().kokkos_view(),snapshot);
    Kokkos::View<Storage*,Space> previous(Kokkos::view_alloc(exec,Kokkos::WithoutInitializing,"gaussian/previous"),owned_count);
    Kokkos::View<Storage*,Space> older(Kokkos::view_alloc(exec,Kokkos::WithoutInitializing,"gaussian/older"),owned_count);
    Kokkos::deep_copy(exec,previous,owned_field);
    Kokkos::deep_copy(exec,older,owned_field);
    // R(0) supplies the affine base RHS on steps whose matrix can be reused.
    Kokkos::View<Storage*,Space> zero_field(Kokkos::view_alloc(exec,Kokkos::WithoutInitializing,"gaussian/zero"),boundary.required_field_size());
    Kokkos::deep_copy(exec,zero_field,Storage{0});
    krona::halo::Exchange<Storage,Exec> halo{mesh.view(),
        {host_mesh.nb_ranks,host_mesh.send_offsets,host_mesh.recv_offsets},comm,0,exec,krona::halo::Transfer::HostStaged};
    halo.add_field(scalar.view());halo.prepare();
    std::unique_ptr<HighOrderState> high_order;
    if(settings.limited)high_order=std::make_unique<HighOrderState>(convection_plan,geometry.view(),mesh.halo_plan(),comm,exec);

    krona::linalg::CsrMatrix<Scalar,Exec> matrix{assembly.pattern()};
    krona::linalg::Vector<Scalar,Exec> rhs{matrix.distribution()},solution{matrix.distribution()};
    Kokkos::View<Scalar*,Space> base_rhs(Kokkos::view_alloc(exec,Kokkos::WithoutInitializing,"gaussian/base_rhs"),settings.limited?owned_count:0);
    Kokkos::View<Real*,Space> residual(Kokkos::view_alloc(exec,Kokkos::WithoutInitializing,"gaussian/residual"),owned_count);
    ts::AssemblyStatus<Exec> status{exec};
    krona::linalg::SolverOptions options;
    options.method="GMRES";options.preconditioner.type="AMG";
    options.relative_tolerance=reduced_precision?1.e-5:1.e-10;
    options.maximum_iterations=2000;options.restart_length=100;
    krona::linalg::Solver<decltype(matrix)> solver{matrix,options};
    const auto prepare = [&]{
        halo.start();halo.finish_async();boundary.apply_async(scalar.view(),law);
        if(high_order)high_order->update_async(scalar.view());
    };
    update_boundary(boundary,inflow,exact,0.0);prepare();
    exec.fence("Gaussian app: initial state ready");
    double total_assembly=0,total_solve=0,total_output=0;
    int total_solves=0,total_iterations=0;
    std::vector<std::pair<double,std::string>> frames;
    const auto write_output = [&](double time,int step,double full_relative_residual){
        const double begin=MPI_Wtime();
        Kokkos::deep_copy(exec,snapshot,scalar.view().kokkos_view());
        exec.fence("Gaussian app: output snapshot ready");
        std::array<double,5> local{}; // volume, |error|, error^2, scalar integral, exact integral
        double linf=0,minimum=INFINITY,maximum=-INFINITY;
        for(std::size_t i=0;i<owned_count;++i){
            reference[i]=exact.average(lower[i],upper[i],time);
            const double value=static_cast<double>(snapshot(i));error[i]=value-reference[i];
            krona::mpi::require(std::isfinite(error[i]),"Gaussian app: nonfinite solution");
            const double volume=static_cast<double>(host_mesh.cell_volume[i]);
            local[0]+=volume;local[1]+=volume*std::abs(error[i]);local[2]+=volume*error[i]*error[i];
            local[3]+=volume*value;local[4]+=volume*reference[i];
            linf=std::max(linf,std::abs(error[i]));minimum=std::min(minimum,value);maximum=std::max(maximum,value);
        }
        for(std::size_t b=0;b<boundary_count;++b){
            const auto f=first_boundary+b;
            boundary_reference[b]=exact.value(static_cast<double>(host_mesh.face_centroid_x[f]),
                static_cast<double>(host_mesh.face_centroid_y[f]),static_cast<double>(host_mesh.face_centroid_z[f]),time);
            boundary_error[b]=static_cast<double>(snapshot(cells+b))-boundary_reference[b];
        }
        std::array<double,5> global{};
        const std::array<double,3> extremes{linf,-minimum,maximum};std::array<double,3> global_extremes{};
        krona::mpi::check(MPI_Allreduce(local.data(),global.data(),5,MPI_DOUBLE,MPI_SUM,comm.native()),"MPI_Allreduce(Gaussian error)");
        krona::mpi::check(MPI_Allreduce(extremes.data(),global_extremes.data(),3,MPI_DOUBLE,MPI_MAX,comm.native()),"MPI_Allreduce(Gaussian extrema)");
        krona::mpi::require(global[0]>0&&std::abs(global[0]-length*cross_section)<=geometry_tolerance*length*cross_section,
            "Gaussian app: expected the complete unit cube");
        const std::string stem="gaussian_3d_"+std::to_string(step);
        const std::array cell_fields{vtk::field("scalar",std::span{snapshot.data(),owned_count}),
            vtk::field("scalar_exact",std::span{reference}),vtk::field("scalar_error",std::span{error})};
        auto* boundary_values=snapshot.data();if(cells!=0)boundary_values+=cells;
        const std::array boundary_fields{vtk::field("scalar",std::span{boundary_values,boundary_count}),
            vtk::field("scalar_exact",std::span{boundary_reference}),vtk::field("scalar_error",std::span{boundary_error})};
        vtk::write_vtu(host_mesh,paths[2],stem,comm,{.cell=cell_fields,.boundary=boundary_fields});
        if(comm.rank()==0){frames.emplace_back(time,stem);write_series(paths[2],frames,false);write_series(paths[2],frames,true);}
        const double local_seconds=MPI_Wtime()-begin;double seconds=0;
        krona::mpi::check(MPI_Allreduce(&local_seconds,&seconds,1,MPI_DOUBLE,MPI_MAX,comm.native()),"MPI_Allreduce(output duration)");
        total_output+=seconds;
        if(comm.rank()==0)std::printf("output step=%d t=%.9g L1=%.6e L2=%.6e Linf=%.6e range=[%.6e,%.6e]\n"
            "  scalar_integral=%.9e exact_integral=%.9e integral_error=%.6e full_relative_residual=%.6e output_seconds=%.6f\n",
            step,time,global[1]/global[0],std::sqrt(global[2]/global[0]),global_extremes[0],-global_extremes[1],global_extremes[2],
            global[3],global[4],global[3]-global[4],full_relative_residual,seconds);
    };
    if(comm.rank()==0)std::printf("execution=%s ranks=%d cells=%llu scheme=%s time_scheme=%s dt=%.9g end_time=%.9g\n"
        "rho=%.9g velocity=(%.9g,%.9g,%.9g) Gamma=%.9g centre=(0.25,0.35,0.4) sigma0=%.9g\n",
        Exec::name(),comm.size(),static_cast<unsigned long long>(host_mesh.n_cells_g),settings.limited?"limited":"upwind",
        settings.bdf2?"bdf2":"euler",settings.dt,settings.end_time,density,exact.speed,exact.speed_y,exact.speed_z,
        static_cast<double>(gamma),ExactGaussian::initial_sigma);
    write_output(0.0,0,0.0);

    double time=0,previous_dt=settings.dt;
    Real cached_weight=0;
    bool matrix_ready=false;
    int step=0;
    while(time<settings.end_time){
        ++step;
        const double remaining=settings.end_time-time;
        const double dt=std::min(settings.dt,remaining);
        const double next_time=remaining<=settings.dt?settings.end_time:time+dt;
        krona::mpi::require(dt>0&&next_time>time,"Gaussian app: time step cannot advance time");
        const bool use_bdf=settings.bdf2&&step>1;
        update_boundary(boundary,inflow,exact,next_time);
        const auto previous_level=tt::TimeLevel{mass,previous};
        const auto older_level=tt::TimeLevel{mass,older};
        IterationResult result;
        bool rebuilt=false;
        double assembled_at=0,solved_at=0;
        const double begin=MPI_Wtime();
        const auto advance = [&](auto derivative,auto history){
            // Constant geometry, mass, flow and diffusivity; Dirichlet values affect only b.
            // Rebuild A only when the storage diagonal changes (startup or a shortened final step).
            rebuilt=!matrix_ready||derivative.weights[0]!=cached_weight;
            if(rebuilt){
                const auto target=ts::make_system_view(matrix,rhs,assembly,exec);
                Kokkos::deep_copy(exec,target.values,Scalar{0});Kokkos::deep_copy(exec,target.rhs,Scalar{0});
                status.begin_async();
                convection.add_system_async(target,boundary,status,mass_flow,cv::NoReconstruction{},inflow);
                diffusion.add_system_async(target,boundary,status,law);
                temporal.add_system_async(target,status,derivative,mass,history);
                if(settings.limited)Kokkos::deep_copy(exec,base_rhs,target.rhs);
            }else{
                const auto local_rhs=rhs.overwrite(exec);
                const auto component=Kokkos::subview(local_rhs,Kokkos::ALL(),0);
                Kokkos::deep_copy(exec,component,Scalar{0});
                status.begin_async();
                // For the frozen affine low-order operator, b = R(phi=0).
                // This reuses library closures, signs and history handling without touching A.
                convection.add_residual_async(zero_field,component,boundary,status,mass_flow,cv::NoReconstruction{},inflow);
                diffusion.add_residual_async(zero_field,component,boundary,status,law);
                temporal.add_residual_async(zero_field,component,status,derivative,mass,history);
                if(settings.limited)Kokkos::deep_copy(exec,base_rhs,component);
            }
            (void)status.check(comm);matrix.commit();rhs.commit();
            cached_weight=derivative.weights[0];matrix_ready=true;
            solution.copy_from_component(field,exec);solution.commit();
            assembled_at=MPI_Wtime();
            const auto correct_rhs = [&]{
                const auto local_rhs=rhs.overwrite(exec);
                const auto component=Kokkos::subview(local_rhs,Kokkos::ALL(),0);
                Kokkos::deep_copy(exec,component,base_rhs);
                status.begin_async();
                convection.add_correction_async(component,boundary,status,mass_flow,
                    high_order->reconstruction(scalar.view(),convection_plan));
                (void)status.check(comm);rhs.commit();
            };
            const auto measure_residual = [&]{
                if(high_order)return full_residual(convection,diffusion,temporal,field,residual,boundary,law,mass_flow,
                    high_order->reconstruction(scalar.view(),convection_plan),inflow,derivative,mass,history,status,comm,exec);
                return full_residual(convection,diffusion,temporal,field,residual,boundary,law,mass_flow,
                    cv::NoReconstruction{},inflow,derivative,mass,history,status,comm,exec);
            };
            result=solve_deferred(matrix,rhs,solution,solver,field,settings.limited,false,
                prepare,correct_rhs,measure_residual,comm,exec);
            solved_at=MPI_Wtime();
        };
        if(use_bdf)advance(nt::Bdf2{}.coefficients(static_cast<Real>(dt),static_cast<Real>(previous_dt)),
            tt::time_history(previous_level,older_level));
        else advance(nt::BackwardEuler{}.coefficients(static_cast<Real>(dt)),tt::time_history(previous_level));
        std::array<double,2> local_times{assembled_at-begin,solved_at-assembled_at},global_times{};
        krona::mpi::check(MPI_Allreduce(local_times.data(),global_times.data(),2,MPI_DOUBLE,MPI_MAX,comm.native()),"MPI_Allreduce(step times)");
        total_assembly+=global_times[0];total_solve+=global_times[1];
        total_solves+=result.linear_solves;total_iterations+=result.total_krylov_iterations;
        if(comm.rank()==0)std::printf("step=%d t=%.9g dt=%.9g method=%s matrix=%s solves=%d krylov=%d corrections=%d relative_residual=%.6e\n",
            step,next_time,dt,use_bdf?"bdf2":"euler",rebuilt?"rebuilt":"reused",result.linear_solves,
            result.total_krylov_iterations,result.corrections,result.relative_residual());
        if(!result.converged){
            if(comm.rank()==0)std::fprintf(stderr,"Gaussian app: step %d failed; history was not advanced\n",step);
            return EXIT_FAILURE;
        }
        // Accept only the fully converged step. Copies are ordered after all history readers.
        Kokkos::deep_copy(exec,older,previous);Kokkos::deep_copy(exec,previous,owned_field);
        previous_dt=dt;time=next_time;
        if(step%settings.output_every==0||time==settings.end_time)write_output(time,step,result.relative_residual());
    }
    exec.fence("Gaussian app: accepted history ready");
    if(comm.rank()==0)std::printf("completed steps=%d linear_solves=%d total_krylov_iterations=%d preconditioner_builds=%llu\n"
        "assembly_seconds=%.6f solve_seconds=%.6f output_seconds=%.6f\n"
        "Open %s/gaussian_3d.pvd (volume) or gaussian_3d_boundary.pvd (boundary) in ParaView.\n",
        step,total_solves,total_iterations,static_cast<unsigned long long>(solver.preconditioner().generation()),
        total_assembly,total_solve,total_output,paths[2].c_str());
    return EXIT_SUCCESS;
}

double positive_number(const std::string& text)
{
    char* end=nullptr;errno=0;
    const double value=std::strtod(text.c_str(),&end);
    krona::mpi::require(end!=text.c_str()&&end==text.c_str()+text.size()&&errno!=ERANGE&&std::isfinite(value)&&value>0,
        "Gaussian app: dt and end_time must be finite positive numbers");
    return value;
}
int execute(int argc,char* const* argv,krona::mpi::CommView comm)
{
    int action=0;
    if(comm.rank()==0){
        if(argc==1||(argc==2&&std::string_view(argv[1])=="--help"))action=1;
        else if(argc<4||argc>9)action=2;
    }
    krona::mpi::check(MPI_Bcast(&action,1,MPI_INT,0,comm.native()),"MPI_Bcast(CLI action)");
    if(action){if(comm.rank()==0)usage(action==1?stdout:stderr);return action==1?EXIT_SUCCESS:EXIT_FAILURE;}
    std::array<std::string,3> paths;
    std::array<std::string,5> arguments{"0.005","0.35","5","limited","bdf2"};
    if(comm.rank()==0){
        for(std::size_t i=0;i<paths.size();++i)paths[i]=argv[i+1];
        for(int i=4;i<argc;++i)arguments[static_cast<std::size_t>(i-4)]=argv[i];
    }
    for(auto& path:paths){broadcast_text(path,comm);krona::mpi::require(!path.empty(),"Gaussian app: empty path");}
    for(auto& arg:arguments)broadcast_text(arg,comm);
    Settings settings;settings.dt=positive_number(arguments[0]);settings.end_time=positive_number(arguments[1]);
    krona::mpi::require(settings.end_time/settings.dt<static_cast<double>(std::numeric_limits<int>::max()-2),
        "Gaussian app: too many time steps for the step counter");
    char* end=nullptr;errno=0;
    const long interval=std::strtol(arguments[2].c_str(),&end,10);
    krona::mpi::require(end!=arguments[2].c_str()&&end==arguments[2].c_str()+arguments[2].size()&&
        errno!=ERANGE&&interval>0&&interval<=std::numeric_limits<int>::max(),"Gaussian app: output_every must be a positive integer");
    settings.output_every=static_cast<int>(interval);
    krona::mpi::require(arguments[3]=="limited"||arguments[3]=="upwind","Gaussian app: scheme must be limited or upwind");
    krona::mpi::require(arguments[4]=="bdf2"||arguments[4]=="euler","Gaussian app: time scheme must be bdf2 or euler");
    settings.limited=arguments[3]=="limited";settings.bdf2=arguments[4]=="bdf2";
    return run(paths,settings,comm);
}
} // namespace

int main(int argc,char** argv)
{
    krona::mpi::Runtime runtime{argc,argv,krona::mpi::ThreadLevel::funneled};
    try{
        Kokkos::ScopeGuard kokkos{argc,argv};Tpetra::ScopeGuard tpetra{&argc,&argv};
        try{return execute(argc,argv,krona::mpi::CommView::world());}
        catch(const std::exception& error){krona::mpi::fatal(error.what());}
        catch(...){krona::mpi::fatal("Unhandled Gaussian application exception");}
    }catch(const std::exception& error){krona::mpi::fatal(error.what());}
    catch(...){krona::mpi::fatal("Gaussian runtime initialization/finalization failed");}
    return EXIT_FAILURE;
}