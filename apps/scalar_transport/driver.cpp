#include "driver.hpp"
#include "workspace.hpp"
#include "io.hpp"
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/transient/operator.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <locale>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace krona::apps::scalar_transport {
namespace {
using Storage = StorageReal;
namespace tt = ts::transient;
namespace vtk = io::vtk;
template<class View>
struct SumSquares {
    View values;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, double& sum) const
    { const double value = static_cast<double>(values(i)); sum += value*value; }
};
template<class View>
double norm(View values, const Exec& exec, mpi::CommView comm)
{
    double local = 0, global = 0;
    Kokkos::parallel_reduce("scalar_transport/residual_norm",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>(exec, 0, values.extent(0)), SumSquares<View>{values}, local);
    mpi::check(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm.native()), "MPI_Allreduce(residual norm)");
    mpi::require(std::isfinite(global), "Scalar transport: nonfinite full residual norm");
    return std::sqrt(global);
}
struct Statistics { double l1 = 0, l2 = 0, linf = 0, integral = 0, minimum = 0, maximum = 0; };
Statistics statistics(const mesh::host::MeshPart& m, std::span<const Storage> values,
    std::span<const double> exact, mpi::CommView comm)
{
    std::array<double, 4> local{}, global{};
    std::array<double, 3> extrema{0, -INFINITY, -INFINITY}, all_extrema{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double value = static_cast<double>(values[i]), volume = static_cast<double>(m.cell_volume[i]);
        const double error = exact.empty() ? 0 : value-exact[i];
        mpi::require(std::isfinite(value) && std::isfinite(error), "Scalar transport: nonfinite solution or reference");
        local[0] += volume; local[1] += volume*std::abs(error); local[2] += volume*error*error; local[3] += volume*value;
        extrema[0] = std::max(extrema[0], std::abs(error)); extrema[1] = std::max(extrema[1], -value); extrema[2] = std::max(extrema[2], value);
    }
    mpi::check(MPI_Allreduce(local.data(), global.data(), 4, MPI_DOUBLE, MPI_SUM, comm.native()), "MPI_Allreduce(solution statistics)");
    mpi::check(MPI_Allreduce(extrema.data(), all_extrema.data(), 3, MPI_DOUBLE, MPI_MAX, comm.native()), "MPI_Allreduce(solution extrema)");
    mpi::require(global[0] > 0 && std::isfinite(global[0]) && std::isfinite(global[1]) &&
        std::isfinite(global[2]) && std::isfinite(global[3]), "Scalar transport: invalid global statistics");
    return {global[1]/global[0], std::sqrt(global[2]/global[0]), all_extrema[0], global[3], -all_extrema[1], all_extrema[2]};
}
void write_series(const std::filesystem::path& directory, const std::vector<std::pair<double, int>>& frames, bool boundary)
{
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(directory/(boundary ? "scalar_boundary.pvd" : "scalar.pvd"), std::ios::trunc);
    out.imbue(std::locale::classic()); out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\"><Collection>\n";
    for (const auto& [time, step] : frames)
        out << "<DataSet timestep=\"" << time << "\" group=\"\" part=\"0\" file=\"scalar_" << step
            << (boundary ? "_bnd.pvtu" : ".pvtu") << "\"/>\n";
    out << "</Collection></VTKFile>\n"; out.close();
}
} // namespace

int run(const Configuration& c, Verbosity verbosity, mpi::CommView comm)
{
    const auto host = io::solver_mesh::import_mesh_hdf5(c.mesh_file, comm);
    mpi::require(host.n_cells_g > 0, "Scalar transport: the global mesh is empty");
    Exec exec{};
    const mesh::device::MeshStorage<Exec> device_mesh(host, {.data = mesh::device::MeshData::FiniteVolume}, exec);
    const auto m = device_mesh.view();
    const auto host_connectivity = mesh::host::build_aux_connectivity(host, mesh::AuxConnType::CellFaces);
    auto metrics=mesh::AuxGeomType::None;
    if(c.diffusion_enabled)metrics=metrics|mesh::AuxGeomType::FaceNormalDistance|mesh::AuxGeomType::FaceAreaOverNormalDistance;
    if(c.nonorthogonal_correction)metrics=metrics|mesh::AuxGeomType::FaceNonOrthogonalityVector|mesh::AuxGeomType::FaceInterpolationWeights;
    if(c.needs_gradients())metrics=metrics|gr::requirements(c.gradient).geometry;
    const auto host_geometry=mesh::host::build_aux_geometry(host,metrics);
    const mesh::device::AuxConnectivityStorage<Exec> connectivity(host_connectivity, exec);
    const mesh::device::AuxGeometryStorage<Exec> geometry(host_geometry, exec);
    const boundary::BoundaryLayout<Exec, Real> layout(host, m, exec);
    Boundary boundary(layout);
    const auto initial_profile=make_profile(c);
    const auto profile=make_profile(c,true);
    configure_boundary(boundary,bc::parse_config(read_text(c.boundary_file,comm),c.field),profile);
    const SpatialEquation spatial(c, m, connectivity.view(), geometry.view(), comm, exec);
    const auto owned = static_cast<std::size_t>(host.n_own);
    fields::FieldStorage<Storage, 1, Exec> scalar(c.field, boundary.required_field_size(), fields::FieldLocation::CellExtnd, exec);
    const auto field = scalar.view().component(0);
    const auto owned_field = Kokkos::subview(field, std::pair<std::size_t, std::size_t>{0, owned});
    auto snapshot = Kokkos::create_mirror(Kokkos::WithoutInitializing, field);
    Kokkos::deep_copy(snapshot, Storage{0});
    std::vector<CellBox> boxes(owned);
    for(std::size_t i=0;i<owned;++i) {
        boxes[i].lo={static_cast<double>(host.cell_centroid_x[i]),static_cast<double>(host.cell_centroid_y[i]),static_cast<double>(host.cell_centroid_z[i])};
        boxes[i].hi=boxes[i].lo;
    }
    // std::visit is host-side and independent of the linear/time solver types.
    const auto prepare_boxes=[&](const auto& p) {
        using P = std::remove_cvref_t<decltype(p)>;
        if constexpr (std::is_same_v<P, GaussianProfile>) {
            const Real width = computation_value(p.width);
            mpi::require(std::isfinite(width*width) && width*width > Real{0}, "Scalar transport: Gaussian width squared is invalid in computation precision");
            for (std::size_t i = 0; i < owned; ++i) boxes[i]=cartesian_box(host, i);
        }
    };
    std::visit(prepare_boxes,initial_profile);
    if(c.verification)std::visit(prepare_boxes,profile);
    std::visit([&](const auto& p) {
        for (std::size_t i = 0; i < owned; ++i) {
            const auto& box=boxes[i];
            snapshot(i) = static_cast<Storage>(p.average(box.lo, box.hi, 0));
            mpi::require(std::isfinite(snapshot(i)), "Scalar transport: initial condition is unrepresentable in field storage");
        }
    }, initial_profile);
    Kokkos::deep_copy(exec, field, snapshot);
    std::optional<halo::Exchange<Storage, Exec>> exchange;
    if (c.convection || c.diffusion_enabled) {
        exchange.emplace(m, device_mesh.halo_plan(), comm, 71, exec, halo::Transfer::HostStaged);
        exchange->add_field(scalar.view()); exchange->prepare();
    }
    NumericsWorkspace workspace(c,m,connectivity.view(),geometry.view(),device_mesh.halo_plan(),boundary,comm,exec);
    const auto publish=[&](Real at) {
        if(exchange)exchange->exchange();
        verify_boundary(c,boundary,profile,at);
        return workspace.prepare(scalar.view(),boundary,spatial.law(),spatial.scheme(),at);
    };
    (void)publish(Real{0});
    const linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
    linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
    linalg::Vector<SolverReal, Exec> rhs(matrix.distribution()), solution(matrix.distribution());
    linalg::SolverOptions options;
    options.method = c.linear.method; options.preconditioner.type = c.linear.preconditioner;
    options.relative_tolerance = c.linear.relative_tolerance; options.absolute_tolerance = c.linear.absolute_tolerance;
    options.maximum_iterations = c.linear.maximum_iterations; options.restart_length = c.linear.restart_length;
    options.verify_residual = c.linear.verify_residual;
    linalg::Solver<decltype(matrix)> linear(matrix, options);
    const tt::TransientOperator<Exec, Real> temporal(owned, exec);
    const auto mass = tt::DensityVolumeMass{m.geometry.cell_volume, computation_value(c.density)};
    Kokkos::View<Storage*,Space> previous("scalar/accepted",c.mode==Mode::Transient?owned:0);
    Kokkos::View<Storage*,Space> older("scalar/older",c.mode==Mode::Transient&&c.time_scheme==ts::TimeScheme::Bdf2?owned:0);
    Kokkos::View<Storage*,Space> unrelaxed("scalar/previous_iterate",c.outer.relaxation!=1?owned:0);
    Kokkos::View<SolverReal*,Space> base_rhs("scalar/base_rhs",owned);
    if(c.mode==Mode::Transient)Kokkos::deep_copy(exec,previous,owned_field);
    Kokkos::View<AccumulationReal*, Space> residual("scalar/residual", owned);
    ts::AssemblyStatus<Exec> status(exec);
    std::vector<double> reference(c.verification ? owned : 0), errors(reference.size());
    std::vector<std::pair<double, int>> frames;
    double final_l2 = 0;
    const std::string exact_name=c.field+"_exact",error_name=c.field+"_error";
    const auto report = [&](double time, int step, double full_residual) {
        mpi::guarded([&] {
            Kokkos::deep_copy(exec, snapshot, field); exec.fence("Scalar transport: output snapshot ready");
            if (c.verification) std::visit([&](const auto& p) {
                for (std::size_t i = 0; i < owned; ++i) {
                    const auto& box=boxes[i];
                    reference[i] = p.average(box.lo, box.hi, time);
                    errors[i] = static_cast<double>(snapshot(i))-reference[i];
                }
            }, profile);
            const auto stats = statistics(host, std::span<const Storage>{snapshot.data(), owned}, reference, comm);
            final_l2 = stats.l2;
            if (comm.rank() == 0 && verbosity != Verbosity::QUIET) {
                std::printf("step=%d time=%.12g", step, time);
                if (step != 0) std::printf(" residual=%.6e", full_residual);
                std::printf(" range=[%.6e,%.6e] integral=%.12g", stats.minimum, stats.maximum, stats.integral);
                if (c.verification) std::printf(" L1=%.6e L2=%.6e Linf=%.6e", stats.l1, stats.l2, stats.linf);
                std::fputc('\n', stdout); std::fflush(stdout);
            }
            if (c.output_enabled) {
                std::vector<vtk::Field> fields{vtk::field(c.field, std::span{snapshot.data(), owned})};
                if (c.verification) {
                    fields.push_back(vtk::field(exact_name, std::span{reference})); fields.push_back(vtk::field(error_name, std::span{errors}));
                }
                const auto* boundary_begin=host.n_cells==0?snapshot.data():snapshot.data()+host.n_cells;
                const std::array boundary_fields{vtk::field(c.field,std::span{boundary_begin,static_cast<std::size_t>(host.n_faces-host.n_inner_faces)})};
                vtk::write_vtu(host,c.output_directory,"scalar_"+std::to_string(step),comm,{.cell=fields,.boundary=boundary_fields});
                if (comm.rank() == 0) {
                    frames.emplace_back(time, step); write_series(c.output_directory, frames, false); write_series(c.output_directory, frames, true);
                    std::ofstream out;
                    out.exceptions(std::ios::failbit | std::ios::badbit);
                    out.open(c.output_directory/"history.csv", step == 0 ? std::ios::trunc : std::ios::app);
                    out.imbue(std::locale::classic()); out << std::setprecision(std::numeric_limits<double>::max_digits10);
                    if (step == 0) out << "step,time,residual,integral,minimum,maximum" << (c.verification ? ",L1,L2,Linf\n" : "\n");
                    out << step << ',' << time << ',';
                    if (step != 0) out << full_residual;
                    out << ',' << stats.integral << ',' << stats.minimum << ',' << stats.maximum;
                    if (c.verification) out << ',' << stats.l1 << ',' << stats.l2 << ',' << stats.linf;
                    out << '\n'; out.close();
                }
            }
        });
    };
    if(comm.rank()==0 && verbosity!=Verbosity::QUIET)
        std::printf("scalar_transport: ranks=%d cells=%llu field=%s mode=%s gradients=%s reconstruction=%d correction=%s\n",
            comm.size(),static_cast<unsigned long long>(host.n_cells_g),c.field.c_str(),c.mode==Mode::Steady?"steady":"transient",
            c.needs_gradients()?"on":"off",static_cast<int>(c.reconstruction.method),c.nonorthogonal_correction?"on":"off");
    report(0,0,0);
    double time=0,previous_dt=0;
    int step=0;
    const auto values=fields::pack(scalar.view());
    const auto gradients=workspace.gradient_pack();
    const auto factors=workspace.factor_pack();
    const auto correction=workspace.correction();
    while(c.mode==Mode::Steady?step==0:time<c.end_time) {
        mpi::require(step<c.maximum_steps,"Scalar transport: maximum_steps reached");
        double next=time;
        if(c.mode==Mode::Transient) {
            next=std::min(c.end_time,time+c.dt);
            if(c.end_time-next<=16*std::numeric_limits<double>::epsilon()*std::max(std::abs(c.end_time),std::abs(next)))next=c.end_time;
            mpi::require(next>time,"Scalar transport: time step does not advance time");
        }
        const Real at=computation_value(next);
        // The independent time stage instantiates BE/BDF2 only around the temporal operator.
        const auto with_time=[&](auto&& add) {
            if(c.mode==Mode::Steady)return;
            if(c.time_scheme==ts::TimeScheme::Bdf2 && step>0)
                add(numerics::time::Bdf2{}.coefficients(computation_value(next-time),computation_value(previous_dt)),
                    tt::time_history(tt::TimeLevel{mass,previous},tt::TimeLevel{mass,older}));
            else add(numerics::time::BackwardEuler{}.coefficients(computation_value(next-time)),
                    tt::time_history(tt::TimeLevel{mass,previous}));
        };
        // Freeze the upwind/orthogonal matrix for this step. Only RHS changes below.
        const auto target=ts::make_system_view(matrix,rhs,assembly,exec);
        Kokkos::deep_copy(exec,target.values,SolverReal{0});Kokkos::deep_copy(exec,target.rhs,SolverReal{0});
        status.begin_async();spatial.add_base_async(target,boundary,status,at);
        with_time([&](auto weights,auto history){temporal.add_system_async(target,status,weights,mass,history);});
        (void)status.check(comm);matrix.commit();rhs.commit();
        const auto frozen=Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0);
        Kokkos::deep_copy(exec,base_rhs,frozen);rhs.commit();
        const auto frozen_revision=matrix.revision();
        (void)publish(at);
        const auto full_residual=[&] {
            Kokkos::deep_copy(exec,residual,AccumulationReal{0});status.begin_async();
            spatial.add_residual_async(field,residual,boundary,status,values,gradients,factors,correction,at);
            with_time([&](auto weights,auto history){temporal.add_residual_async(field,residual,status,weights,mass,history);});
            (void)status.check(comm);return norm(residual,exec,comm);
        };
        // Reference norm is frozen too, including homogeneous-RHS problems.
        const double scale=std::max(norm(base_rhs,exec,comm),full_residual());
        const double tolerance=std::max(c.outer.absolute_tolerance,c.outer.relative_tolerance*scale);
        const double feedback_tolerance=std::max(c.outer.relative_tolerance,64*std::max(
            static_cast<double>(std::numeric_limits<Storage>::epsilon()),static_cast<double>(std::numeric_limits<Real>::epsilon())));
        double full=0;bool accepted=false;int total_linear=0;
        for(int outer=1;outer<=c.outer.maximum_iterations;++outer) {
            const auto b=Kokkos::subview(rhs.overwrite(exec),Kokkos::ALL(),0);
            Kokkos::deep_copy(exec,b,base_rhs);status.begin_async();
            spatial.add_correction_async(b,boundary,status,values,gradients,factors,correction,at);
            (void)status.check(comm);rhs.commit();
            mpi::require(matrix.revision()==frozen_revision,"Scalar transport: correction modified the frozen matrix");
            if(c.outer.relaxation!=1)Kokkos::deep_copy(exec,unrelaxed,owned_field);
            solution.copy_from_component(field,exec);
            const auto result=linear.solve(rhs,solution);total_linear+=result.iterations;
            if(!result.converged) {
                if(comm.rank()==0)std::fprintf(stderr,"scalar_transport: linear solve failed at step %d, outer %d; step not accepted\n",step+1,outer);
                return EXIT_FAILURE;
            }
            solution.copy_to_component(field,exec);
            if(c.outer.relaxation!=1) {
                const Real omega=computation_value(c.outer.relaxation);const auto old=unrelaxed;
                Kokkos::parallel_for("scalar/relax",Kokkos::RangePolicy<Exec,Kokkos::IndexType<std::size_t>>(exec,0,owned),
                    KOKKOS_LAMBDA(std::size_t i){field(i)=static_cast<Storage>((Real{1}-omega)*static_cast<Real>(old(i))+omega*static_cast<Real>(field(i)));});
            }
            const double feedback=publish(at);full=full_residual();
            if(comm.rank()==0 && verbosity==Verbosity::DETAILED)
                std::printf("step=%d outer=%d linear=%d residual=%.6e target=%.6e boundary_feedback=%.6e\n",step+1,outer,result.iterations,full,tolerance,feedback);
            if(full<=tolerance && feedback<=feedback_tolerance) {accepted=true;break;}
        }
        if(!accepted) {
            if(comm.rank()==0)std::fprintf(stderr,"scalar_transport: outer iteration failed at step %d: residual=%.6e target=%.6e; step not accepted\n",step+1,full,tolerance);
            return EXIT_FAILURE;
        }
        // Accepted states only: outer iterates never become temporal history.
        if(c.mode==Mode::Transient) {
            if(c.time_scheme==ts::TimeScheme::Bdf2)Kokkos::deep_copy(exec,older,previous);
            Kokkos::deep_copy(exec,previous,owned_field);previous_dt=next-time;
        }
        time=next;++step;
        if(comm.rank()==0 && verbosity!=Verbosity::QUIET)
            std::printf("solve step=%d linear_iterations=%d full_residual=%.6e\n",step,total_linear,full);
        if(step%c.output_interval==0 || c.mode==Mode::Steady || time==c.end_time)report(time,step,full);
    }
    const bool passed = !c.maximum_l2_error || final_l2 <= *c.maximum_l2_error;
    if (!passed && comm.rank() == 0) std::fprintf(stderr, "scalar_transport: final L2 error %.6e exceeds %.6e\n", final_l2, *c.maximum_l2_error);
    exec.fence("Scalar transport: calculation complete");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
} // namespace krona::apps::scalar_transport
