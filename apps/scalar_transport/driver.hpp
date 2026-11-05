#pragma once
#include "config.hpp"
#include "cli.hpp"
#include <krona/mpi/communicator.hpp>
namespace krona::apps::scalar_transport {
int run(const Configuration& config, Verbosity verbosity, mpi::CommView comm);
}
