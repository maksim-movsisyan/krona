#include "scalar_transport/cli.hpp"
#include "scalar_transport/config.hpp"
#include "scalar_transport/driver.hpp"
#include "scalar_transport/io.hpp"
#include "scalar_transport/profiles.hpp"
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <Tpetra_Core.hpp>
#include <cstdlib>
#include <string>

namespace app = krona::apps::scalar_transport;
int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    return krona::mpi::guarded([&] {
        auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
        const auto comm = communicator.view();
        app::ParseResult parsed;
        if (comm.rank() == 0) parsed = app::parse_arguments(argc, argv);
        int action = static_cast<int>(parsed.action);
        krona::mpi::check(MPI_Bcast(&action, 1, MPI_INT, 0, comm.native()), "MPI_Bcast(CLI action)");
        if (action == static_cast<int>(app::CliAction::HELP)) {
            if (comm.rank() == 0) app::usage(stdout);
            return EXIT_SUCCESS;
        }
        if (action == static_cast<int>(app::CliAction::ERROR)) {
            if (comm.rank() == 0) { std::fprintf(stderr, "scalar_transport: %s\n\n", parsed.error.c_str()); app::usage(stderr); }
            return EXIT_FAILURE;
        }
        std::string path;
        if (comm.rank() == 0) path = std::filesystem::absolute(parsed.options.case_file).lexically_normal().string();
        app::broadcast_text(path, comm);
        int settings[2]{static_cast<int>(parsed.options.verbosity), parsed.options.check_config ? 1 : 0};
        krona::mpi::check(MPI_Bcast(settings, 2, MPI_INT, 0, comm.native()), "MPI_Bcast(CLI settings)");
        const auto config = app::parse_config(app::read_text(path, comm), path);
        (void)app::make_profile(config);
        (void)app::make_profile(config,true);
        const auto verbosity = static_cast<app::Verbosity>(settings[0]);
        if (comm.rank() == 0 && (settings[1] || verbosity == app::Verbosity::DETAILED)) {
            std::printf("case=%s\nmesh=%s\nboundary=%s\noutput=%s\n", path.c_str(), config.mesh_file.string().c_str(),
                config.boundary_file.string().c_str(), config.output_directory.string().c_str());
        }
        if (comm.rank() == 0 && verbosity == app::Verbosity::DETAILED) {
            std::printf("density=%.12g velocity=(%.12g,%.12g,%.12g) Gamma=%.12g source=%.12g convection=%s diffusion=%s\n",
                config.density, config.velocity[0], config.velocity[1], config.velocity[2], config.diffusion, config.source,
                config.convection ? "on" : "off", config.diffusion_enabled ? "on" : "off");
            std::printf("dt=%.12g end_time=%.12g solver=%s preconditioner=%s tolerance=%.6g\n",
                config.dt, config.end_time, config.linear.method.c_str(), config.linear.preconditioner.c_str(), config.linear.relative_tolerance);
        }
        if (settings[1]) { if (comm.rank() == 0) std::puts("Application settings valid (mesh/BC files, geometry and native solver setup have not been checked)."); return EXIT_SUCCESS; }
        Kokkos::ScopeGuard kokkos{argc, argv};
        Tpetra::ScopeGuard tpetra{&argc, &argv, comm.native()};
        return krona::mpi::guarded([&] { return app::run(config, verbosity, comm); });
    });
}
