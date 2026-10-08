// Distributed smoke test for MeshPart -> MeshStorage -> MeshView.
#include <krona/execution/types.hpp>
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mpi/collective_status.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/runtime.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>

namespace {

using Exec = krona::execution::ExecutionSpace;
using Memory = typename Exec::memory_space;
using MeshView = krona::mesh::compute::MeshView<Memory>;
using MeshStorage = krona::mesh::compute::MeshStorage<Exec>;
using MeshData = krona::mesh::compute::MeshData;
using Index = krona::LocalIndex;
using Real = krona::AccumulationReal;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<Index>>;
using Clock = std::chrono::steady_clock;

enum class Verbosity : int { QUIET, SUMMARY, DETAILED };
enum class CliAction : int { RUN, HELP, ERROR };

struct Options {
    std::string input;
    Verbosity verbosity = Verbosity::QUIET;
    bool all_data = false;
};

struct ParseResult {
    Options options;
    CliAction action = CliAction::RUN;
    std::string_view error;
    std::string_view argument;
};

void usage(std::FILE* stream) noexcept
{
    std::fputs(
        "KRONA compute mesh smoke test\n\n"
        "Usage:\n"
        "  mpirun -np N krona_plug <mesh.h5> [options]\n"
        "  krona_plug --help\n\n"
        "Options:\n"
        "  --all-data          Load all compute groups (default: FiniteVolume).\n"
        "  -v, --verbose       Summary diagnostics.\n"
        "  -vv                 Detailed diagnostics, including Kokkos configuration.\n"
        "  -h, --help          Show this help.\n"
        "  --                  End options; the next argument is the input path.\n\n"
        "Kokkos consumes its own --kokkos-* options before this parser runs.\n"
        "Use the same number of MPI ranks as in the saved mesh partition.\n"
        "This test checks geometry and face-cell links; it performs no halo exchange.\n\n",
        stream);
}

[[nodiscard]] ParseResult parse_arguments(int argc, char* const* argv)
{
    ParseResult result;
    if (argc <= 1) { result.action = CliAction::HELP; return result; }
    bool positional_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view token{argv[i]};
        if (!positional_only && token == "--") { positional_only = true; continue; }
        if (!positional_only && (token == "-h" || token == "--help")) {
            result.action = CliAction::HELP;
            return result;
        }
        if (!positional_only && (token == "-v" || token == "--verbose")) {
            result.options.verbosity = std::max(result.options.verbosity, Verbosity::SUMMARY);
            continue;
        }
        if (!positional_only && token == "-vv") {
            result.options.verbosity = Verbosity::DETAILED;
            continue;
        }
        if (!positional_only && token == "--all-data") {
            result.options.all_data = true;
            continue;
        }
        if (!positional_only && token.size() > 1 && token.front() == '-') {
            result.error = "Unknown option";
        } else if (token.empty()) {
            result.error = "Input path must not be empty";
        } else if (!result.options.input.empty()) {
            result.error = "Unexpected positional argument";
        } else {
            result.options.input = token;
            continue;
        }
        result.action = CliAction::ERROR;
        result.argument = token;
        return result;
    }
    if (result.options.input.empty()) {
        result.action = CliAction::ERROR;
        result.error = "Expected an input HDF5 mesh path";
    }
    return result;
}

void broadcast_options(Options& options, krona::mpi::CommView comm)
{
    std::array<int, 3> header{};
    if (comm.rank() == 0) {
        if (options.input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            krona::mpi::fatal("Input path exceeds MPI 3.1 INT_MAX");
        header = {static_cast<int>(options.input.size()),
                  static_cast<int>(options.verbosity), options.all_data ? 1 : 0};
    }
    krona::mpi::broadcast(comm, std::span{header});

    // This preparation has no MPI calls; all ranks reach the status check.
    krona::mpi::LocalError local_error{};
    try {
        if (comm.rank() != 0) options.input.resize(static_cast<std::size_t>(header[0]));
    } catch (const std::exception& error) {
        krona::mpi::set_local_error(local_error, error.what());
    } catch (...) {
        krona::mpi::set_local_error(local_error, "Unhandled input path allocation exception");
    }
    krona::mpi::require_collective_success(comm, local_error, "Allocating input path");
    krona::mpi::broadcast(comm, std::span{options.input});
    options.verbosity = static_cast<Verbosity>(header[1]);
    options.all_data = header[2] != 0;
}

// Custom reduction results are plain data, also stored in rank-zero Kokkos Views.
struct CellStatistics {
    Real volume;
    Real minimum;
    Real maximum;
    std::uint64_t invalid;
};

struct CheckCells {
    using value_type = CellStatistics;
    MeshView mesh;

    KOKKOS_INLINE_FUNCTION void init(value_type& result) const
    {
        result = {Real{0}, std::numeric_limits<Real>::max(), Real{0}, 0};
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const
    {
        dst.volume += src.volume;
        dst.minimum = dst.minimum < src.minimum ? dst.minimum : src.minimum;
        dst.maximum = dst.maximum > src.maximum ? dst.maximum : src.maximum;
        dst.invalid += src.invalid;
    }

    KOKKOS_INLINE_FUNCTION void operator()(Index cell, value_type& result) const
    {
        const Real volume = static_cast<Real>(mesh.geometry.cell_volume(cell));
        const bool valid = Kokkos::isfinite(volume) && volume > Real{0} &&
            Kokkos::isfinite(mesh.geometry.cell_centroid_x(cell)) &&
            Kokkos::isfinite(mesh.geometry.cell_centroid_y(cell)) &&
            Kokkos::isfinite(mesh.geometry.cell_centroid_z(cell));
        if (!valid) { ++result.invalid; return; }
        // Validate ghosts too, but count volume only for owned cells.
        if (cell >= mesh.sizes.n_own) return;
        result.volume += volume;
        result.minimum = result.minimum < volume ? result.minimum : volume;
        result.maximum = result.maximum > volume ? result.maximum : volume;
    }
};

struct FaceStatistics {
    std::uint64_t invalid;
    std::uint64_t boundary;
    std::uint64_t interface_copies;
};

struct CheckFaces {
    using value_type = FaceStatistics;
    MeshView mesh;

    KOKKOS_INLINE_FUNCTION void init(value_type& result) const { result = {0, 0, 0}; }

    KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const
    {
        dst.invalid += src.invalid;
        dst.boundary += src.boundary;
        dst.interface_copies += src.interface_copies;
    }

    KOKKOS_INLINE_FUNCTION void operator()(Index face, value_type& result) const
    {
        const auto owner = mesh.connectivity.face_owner(face);
        const auto neigh = mesh.connectivity.face_neigh(face);
        const Real area = static_cast<Real>(mesh.geometry.face_area(face));
        const Real nx = static_cast<Real>(mesh.geometry.face_normal_x(face));
        const Real ny = static_cast<Real>(mesh.geometry.face_normal_y(face));
        const Real nz = static_cast<Real>(mesh.geometry.face_normal_z(face));
        const Real normal_squared = nx * nx + ny * ny + nz * nz;
        // MeshPart specifies unit normals; use the precision of their storage.
        constexpr Real tolerance = Real{256} * std::numeric_limits<krona::GeometryReal>::epsilon();
        bool valid = owner >= 0 && owner < mesh.sizes.n_own &&
            neigh >= Index{-1} && neigh < mesh.sizes.n_cells && neigh != owner &&
            Kokkos::isfinite(area) && area > Real{0} &&
            Kokkos::isfinite(normal_squared) && Kokkos::abs(normal_squared - Real{1}) <= tolerance &&
            Kokkos::isfinite(mesh.geometry.face_centroid_x(face)) &&
            Kokkos::isfinite(mesh.geometry.face_centroid_y(face)) &&
            Kokkos::isfinite(mesh.geometry.face_centroid_z(face));
        if (mesh.sizes.faces_grouped) {
            const bool grouped = face < mesh.sizes.n_internal_faces ? neigh >= 0 && neigh < mesh.sizes.n_own :
                face < mesh.sizes.n_inner_faces ? neigh >= mesh.sizes.n_own : neigh == Index{-1};
            valid = valid && grouped;
        }
        if (!valid) ++result.invalid;
        if (neigh == Index{-1}) ++result.boundary;
        else if (neigh >= mesh.sizes.n_own) ++result.interface_copies;
    }
};

[[nodiscard]] double elapsed(Clock::time_point start) noexcept
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] int run_plug(const Options& options, krona::mpi::CommView comm)
{
    const bool report = comm.rank() == 0 && options.verbosity != Verbosity::QUIET;
    const bool detailed = report && options.verbosity == Verbosity::DETAILED;
    const auto total_start = Clock::now();

    if (report) {
        std::fprintf(stderr, "KRONA plug: input=%s ranks=%d execution=%s memory=%s data=%s\n",
            options.input.c_str(), comm.size(), Exec::name(), Memory::name(),
            options.all_data ? "All" : "FiniteVolume");
        std::fprintf(stderr, "  geometry=%zu bytes accumulation=%zu bytes local-index=%zu bytes\n",
            sizeof(krona::GeometryReal), sizeof(Real), sizeof(Index));
        std::fprintf(stderr, "  mesh arrays=%s\n",
            MeshStorage::borrows_host ? "borrowed from MeshPart" : "owned compute copies");
    }
    if (detailed) Kokkos::print_configuration(std::cerr, true);

    // Collective importer: exceptions propagate to the application's fatal handler.
    const auto read_start = Clock::now();
    const auto mesh_part = krona::io::solver_mesh::import_mesh_hdf5(
        std::filesystem::path{options.input}, comm);
    const double read_time = elapsed(read_start);

    // One explicit execution-space instance is used for uploads, kernels and copies.
    Exec exec{};
    const auto storage_start = Clock::now();
    const MeshStorage storage(mesh_part,
        {.data = options.all_data ? MeshData::All : MeshData::FiniteVolume}, exec);
    const auto mesh = storage.view();
    const double storage_time = elapsed(storage_start);

    // Allocate before dispatch. CPU mirrors alias these results; GPU mirrors are
    // small host allocations. MeshPart stays live for the entire borrowed CPU view.
    Kokkos::View<CellStatistics, Memory> cell_result(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "plug/cell_result"));
    Kokkos::View<FaceStatistics, Memory> face_result(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "plug/face_result"));
    auto host_cells = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, cell_result);
    auto host_faces = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, face_result);

    const auto compute_start = Clock::now();
    try {
        // A result View permits asynchronous dispatch; a host scalar would require
        // its result before parallel_reduce returns. init/join also handle empty ranks.
        Kokkos::parallel_reduce("plug/check_cells",
            Policy(exec, 0, mesh.sizes.n_cells), CheckCells{mesh}, cell_result);
        Kokkos::parallel_reduce("plug/check_faces",
            Policy(exec, 0, mesh.sizes.n_faces), CheckFaces{mesh}, face_result);

        // Explicit-instance deep_copy is ordered with the preceding kernels.
        Kokkos::deep_copy(exec, host_cells, cell_result);
        Kokkos::deep_copy(exec, host_faces, face_result);
        exec.fence("KRONA plug: host statistics ready for MPI");
    } catch (...) {
        // Do not destroy results or borrowed descriptors while work is pending.
        exec.fence("KRONA plug: drain failed checks");
        throw;
    }
    const double compute_time = elapsed(compute_start);
    const auto cells = host_cells();
    const auto faces = host_faces();

    // MPI sees host data only; this works with CUDA-aware and ordinary MPI alike.
    std::array<std::uint64_t, 6> counts{
        static_cast<std::uint64_t>(mesh.sizes.n_own),
        static_cast<std::uint64_t>(mesh.sizes.n_cells - mesh.sizes.n_own),
        cells.invalid, faces.invalid, faces.boundary, faces.interface_copies};
    krona::mpi::allreduce_in_place<krona::mpi::Reduction::sum>(comm, std::span{counts});
    const Real volume = krona::mpi::allreduce<krona::mpi::Reduction::sum>(comm, cells.volume);
    const Real minimum = krona::mpi::allreduce<krona::mpi::Reduction::min>(comm, cells.minimum);
    const Real maximum = krona::mpi::allreduce<krona::mpi::Reduction::max>(comm, cells.maximum);

    if (counts[2] != 0 || counts[3] != 0) {
        if (comm.rank() == 0)
            std::fprintf(stderr, "KRONA plug: invalid cell copies=%" PRIu64 " invalid faces=%" PRIu64 "\n",
                counts[2], counts[3]);
        krona::mpi::fatal("MeshView geometry/connectivity check failed");
    }
    if (counts[0] != mesh_part.n_cells_g)
        krona::mpi::fatal("Owned cell count differs from saved global count");
    if (!Kokkos::isfinite(volume) || (counts[0] != 0 && volume <= Real{0}))
        krona::mpi::fatal("Global owned volume is not finite and positive");

    // Report maxima of local durations. These are startup diagnostics, not a benchmark.
    std::array<double, 4> times{read_time, storage_time, compute_time, elapsed(total_start)};
    if (options.verbosity != Verbosity::QUIET) {
        krona::mpi::allreduce_in_place<krona::mpi::Reduction::max>(comm, std::span{times});
        if (report) {
            std::fprintf(stderr,
                "  owned cells=%" PRIu64 " ghost copies=%" PRIu64 " boundary faces=%" PRIu64
                " interface face copies=%" PRIu64 "\n"
                "  owned volume=%.17g cell volume range=[%.17g, %.17g]\n"
                "  max local seconds: read=%.6f storage=%.6f checks+copy=%.6f total=%.6f\n"
                "KRONA plug: checks passed\n",
                counts[0], counts[1], counts[4], counts[5], static_cast<double>(volume),
                counts[0] != 0 ? static_cast<double>(minimum) : 0.0,
                counts[0] != 0 ? static_cast<double>(maximum) : 0.0,
                times[0], times[1], times[2], times[3]);
        }
    }
    // Results and storage are destroyed here, before the application's Kokkos guard.
    return EXIT_SUCCESS;
}

[[nodiscard]] int execute(int argc, char* const* argv, krona::mpi::CommView comm)
{
    ParseResult parsed;
    krona::mpi::LocalError local_error{};
    try {
        if (comm.rank() == 0) parsed = parse_arguments(argc, argv);
    } catch (const std::exception& error) {
        krona::mpi::set_local_error(local_error, error.what());
    } catch (...) {
        krona::mpi::set_local_error(local_error, "Unhandled argument parsing exception");
    }
    krona::mpi::require_collective_success(comm, local_error, "Parsing command line");

    int action = static_cast<int>(parsed.action);
    krona::mpi::broadcast(comm, action);
    if (action == static_cast<int>(CliAction::HELP)) {
        if (comm.rank() == 0) usage(stdout);
        return EXIT_SUCCESS;
    }
    if (action == static_cast<int>(CliAction::ERROR)) {
        if (comm.rank() == 0) {
            constexpr std::size_t limit = 512;
            std::fprintf(stderr, "KRONA plug: %.*s",
                static_cast<int>(std::min(parsed.error.size(), limit)), parsed.error.data());
            if (!parsed.argument.empty())
                std::fprintf(stderr, ": '%.*s'%s",
                    static_cast<int>(std::min(parsed.argument.size(), limit)), parsed.argument.data(),
                    parsed.argument.size() > limit ? " [truncated]" : "");
            std::fputs("\n\n", stderr);
            usage(stderr);
        }
        return EXIT_FAILURE;
    }
    broadcast_options(parsed.options, comm);
    return run_plug(parsed.options, comm);
}

// Abort immediately on an unexpected rank-local exception. Other ranks may be
// inside the importer or an MPI collective; do not try another status collective.
[[nodiscard]] int execute_safely(int argc, char* const* argv, krona::mpi::CommView comm) noexcept
{
    try {
        return execute(argc, argv, comm);
    } catch (const std::exception& error) {
        krona::mpi::fatal(error.what());
    } catch (...) {
        krona::mpi::fatal("Unhandled plug exception");
    }
}

} // namespace

int main(int argc, char** argv)
{
    // All MPI calls run on the initializing thread; kernels never call MPI.
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
    try {
        // Initializes every enabled Kokkos backend and consumes --kokkos-* flags.
        // Normal shutdown order: views/storage -> Kokkos -> communicator -> MPI.
        Kokkos::ScopeGuard kokkos{argc, argv};
        return execute_safely(argc, argv, communicator.view());
    } catch (const std::exception& error) {
        krona::mpi::fatal(error.what());
    } catch (...) {
        krona::mpi::fatal("Unhandled Kokkos initialization/finalization exception");
    }
}