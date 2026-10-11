// Steady heat conduction in an insulated rectangular bar; T_exact(x) = 300 + 100*x.
#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/io/vtu/vtu.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/transport/scalar/steady_diffusion.hpp>

#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
namespace bc = krona::boundary::scalar;
namespace ts = krona::transport::scalar;
namespace vtk = krona::io::vtk;

constexpr double conductivity = 2.0;
constexpr double initial_temperature = 350.0;

/// @brief Exact temperature for the supplied bar geometry and boundary configuration.
[[nodiscard]] double exact_temperature(double x) noexcept { return 300.0 + 100.0*x; }

void usage(std::FILE* stream)
{
    std::fputs(
        "KRONA steady scalar diffusion case\n\n"
        "Usage:\n"
        "  mpirun -np N krona_scalar_diffusion <mesh.h5> <boundary.toml> <output-dir>\n"
        "  krona_scalar_diffusion --help\n\n"
        "Use examples/transport/scalar/linear_bar/mesh.toml and boundary.toml.\n"
        "N must match the saved mesh partition. Kokkos accepts --kokkos-* options.\n"
        "This application fixes conductivity=2, source=0 and T_exact=300+100*x.\n"
        "Changing the physical case also requires updating its exact solution.\n", stream);
}

/// @brief Broadcasts host text; the root value is authoritative on all ranks.
void broadcast_text(std::string& text, krona::mpi::CommView comm)
{
    if (comm.rank() == 0 && text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        krona::mpi::fatal("Scalar diffusion app: text exceeds MPI INT_MAX");
    int count = comm.rank() == 0 ? static_cast<int>(text.size()) : 0;
    krona::mpi::check(MPI_Bcast(&count, 1, MPI_INT, 0, comm.native()), "MPI_Bcast(text length)");
    if (comm.rank() != 0) text.resize(static_cast<std::size_t>(count));
    krona::mpi::check(MPI_Bcast(text.data(), count, MPI_CHAR, 0, comm.native()), "MPI_Bcast(text)");
}

/// @brief Reads once on rank zero; each rank subsequently parses the same TOML text.
[[nodiscard]] bc::Configuration load_boundary(const std::filesystem::path& path,
                                              krona::mpi::CommView comm)
{
    std::string text;
    if (comm.rank() == 0) {
        std::ifstream input{path, std::ios::binary};
        if (!input) krona::mpi::fatal("Scalar diffusion app: cannot open boundary file");
        text.assign(std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{});
        if (input.bad()) krona::mpi::fatal("Scalar diffusion app: cannot read boundary file");
    }
    broadcast_text(text, comm);
    return bc::parse_config(text, "temperature");
}

/// @brief Runs the complete mesh/field/boundary/linear-solver/halo/output pipeline.
[[nodiscard]] int run(const std::array<std::string, 3>& paths, krona::mpi::CommView comm)
{
    // MeshPart remains live: CPU compute arrays may borrow its buffers, and VTU uses it directly.
    const auto host_mesh = krona::io::solver_mesh::import_mesh_hdf5(paths[0], comm);
    Exec exec{};
    const krona::mesh::compute::MeshStorage<Exec> mesh{
        host_mesh, {.data = krona::mesh::compute::MeshData::FiniteVolume}, exec};
    const auto config = load_boundary(paths[1], comm);
    const bc::Manager<Exec, Real> boundary{host_mesh, mesh.view(), config, exec};

    // Extended storage: owned cells, MPI ghosts, then one face-centroid value per boundary face.
    krona::fields::FieldStorage<krona::StorageReal, 1, Exec> temperature{
        "temperature", boundary.required_field_size(), krona::fields::FieldLocation::Custom, exec};
    temperature.fill(exec, static_cast<krona::StorageReal>(initial_temperature));

    ts::SteadyDiffusionOptions options;
    options.linear.relative_tolerance = std::is_same_v<krona::SolverReal, float> ? 2.0e-6 : 1.0e-8;
    options.linear.maximum_iterations = 2000;
    options.linear.restart_length = 100;
    // Keep the default orthogonality check and host-staged MPI transport.
    ts::SteadyDiffusion<> solver{host_mesh, mesh.view(), temperature.view().component(0),
                                 comm, exec, options};
    const ts::UniformSource<Real> source{Real{0}};
    const bc::UniformDiffusivity<Real> diffusion{static_cast<Real>(conductivity)};

    exec.fence("Scalar diffusion app: setup complete");
    krona::mpi::check(MPI_Barrier(comm.native()), "MPI_Barrier(before solve)");

    const double start = MPI_Wtime();
    const auto result = solver.solve(boundary, source, diffusion);
    exec.fence("Scalar diffusion app: solve measurement complete");
    
    const double local_seconds = MPI_Wtime() - start;
    double solve_seconds = 0;

    krona::mpi::check(
        MPI_Allreduce(&local_seconds, &solve_seconds, 1,
                    MPI_DOUBLE, MPI_MAX, comm.native()),
        "MPI_Allreduce(solve duration)");

    if (comm.rank() == 0)
        std::printf("execution=%s ranks=%d cells=%llu\n"
                    "converged=%s iterations=%d relative_residual=%.6e\nsolve_seconds=%.6f seconds\n",
            Exec::name(), comm.size(), static_cast<unsigned long long>(host_mesh.n_cells_g),
            result.converged ? "yes" : "no", result.iterations, result.relative_residual, solve_seconds);
    if (!result.converged) return EXIT_FAILURE;

    const auto balance = solver.balance(boundary, source, diffusion);

    // The writer is host-only. Allocate a snapshot even on CPU; no device pointer reaches VTU.
    auto values = temperature.create_host_buffer();
    Kokkos::deep_copy(exec, values, temperature.view().kokkos_view());
    exec.fence("Scalar diffusion app: visualization snapshot ready");

    const auto owned = static_cast<std::size_t>(host_mesh.n_own);
    const auto cells = static_cast<std::size_t>(host_mesh.n_cells);
    const auto first_boundary = static_cast<std::size_t>(host_mesh.n_inner_faces);
    const auto boundary_faces = static_cast<std::size_t>(host_mesh.n_faces)-first_boundary;
    std::vector<double> exact(owned), error(owned), boundary_exact(boundary_faces), boundary_error(boundary_faces);
    std::array<double, 3> sums{}; // Owned volume, volume-weighted absolute error and squared error.
    double maximum = 0;
    for (std::size_t i = 0; i < owned; ++i) {
        exact[i] = exact_temperature(static_cast<double>(host_mesh.cell_centroid_x[i]));
        error[i] = static_cast<double>(values(i))-exact[i];
        if (!std::isfinite(error[i])) krona::mpi::fatal("Scalar diffusion app: nonfinite temperature");
        const double volume = static_cast<double>(host_mesh.cell_volume[i]);
        sums[0] += volume;
        sums[1] += volume*std::abs(error[i]);
        sums[2] += volume*error[i]*error[i];
        maximum = std::max(maximum, std::abs(error[i]));
    }
    for (std::size_t b = 0; b < boundary_faces; ++b) {
        boundary_exact[b] = exact_temperature(static_cast<double>(host_mesh.face_centroid_x[first_boundary+b]));
        boundary_error[b] = static_cast<double>(values(cells+b))-boundary_exact[b];
    }
    std::array<double, 3> global{};
    double global_maximum = 0;
    krona::mpi::check(MPI_Allreduce(sums.data(), global.data(), 3, MPI_DOUBLE, MPI_SUM, comm.native()),
                      "MPI_Allreduce(error integrals)");
    krona::mpi::check(MPI_Allreduce(&maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX, comm.native()),
                      "MPI_Allreduce(maximum error)");
    if (!(global[0] > 0)) krona::mpi::fatal("Scalar diffusion app: mesh has no positive owned volume");
    if (comm.rank() == 0)
        std::printf("temperature error [K]: L1=%.6e L2=%.6e Linf=%.6e\n"
                    "source=%.6e W outward_boundary_flux=%.6e W balance_defect=%.6e W\n",
            global[1]/global[0], std::sqrt(global[2]/global[0]), global_maximum,
            balance.source_integral, balance.outward_boundary_flux, balance.defect);

    // These arrays are already trimmed: owned cells and physical boundary faces, respectively.
    const std::array cell_fields{
        vtk::field("temperature", std::span{values.data(), owned}),
        vtk::field("temperature_exact", std::span{exact}),
        vtk::field("temperature_error", std::span{error})};
    auto* boundary_values = values.data();
    if (cells != 0) boundary_values += cells; // Avoid arithmetic on a null pointer on an empty rank.
    const std::array boundary_fields{
        vtk::field("temperature", std::span{boundary_values, boundary_faces}),
        vtk::field("temperature_exact", std::span{boundary_exact}),
        vtk::field("temperature_error", std::span{boundary_error})};
    vtk::write_vtu(host_mesh, paths[2], "linear_bar", comm,
                   {.cell = cell_fields, .boundary = boundary_fields});
    if (comm.rank() == 0)
        std::printf("VTK datasets written to %s (stem: linear_bar)\n", paths[2].c_str());
    return EXIT_SUCCESS;
}

/// @brief Uses rank zero's arguments; every rank takes the same collective control path.
[[nodiscard]] int execute(int argc, char* const* argv, krona::mpi::CommView comm)
{
    int action = 0; // Run, help, invalid arguments.
    if (comm.rank() == 0) {
        if (argc == 1 || (argc == 2 && std::string_view{argv[1]} == "--help")) action = 1;
        else if (argc != 4) action = 2;
    }
    krona::mpi::check(MPI_Bcast(&action, 1, MPI_INT, 0, comm.native()), "MPI_Bcast(CLI action)");
    if (action != 0) {
        if (comm.rank() == 0) usage(action == 1 ? stdout : stderr);
        return action == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    std::array<std::string, 3> paths;
    if (comm.rank() == 0)
        for (std::size_t i = 0; i < paths.size(); ++i) paths[i] = argv[i+1];
    for (auto& path : paths) {
        broadcast_text(path, comm);
        if (path.empty()) krona::mpi::fatal("Scalar diffusion app: empty path argument");
    }
    return run(paths, comm);
}

} // namespace

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    try {
        Kokkos::ScopeGuard kokkos{argc, argv};
        Tpetra::ScopeGuard tpetra{&argc, &argv};
        // Abort inside the guard scope: a rank-local exception must not strand peers in collectives.
        try { return execute(argc, argv, krona::mpi::CommView::world()); }
        catch (const std::exception& error) { krona::mpi::fatal(error.what()); }
        catch (...) { krona::mpi::fatal("Unhandled scalar diffusion application exception"); }
    } catch (const std::exception& error) { krona::mpi::fatal(error.what()); }
    catch (...) { krona::mpi::fatal("Scalar diffusion runtime initialization/finalization failed"); }
    return EXIT_FAILURE;
}
