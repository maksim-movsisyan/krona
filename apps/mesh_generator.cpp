#include "mesh_generator/driver.hpp"

#include <krona/mpi/communicator.hpp>
#include <krona/mpi/runtime.hpp>

/// @brief Owns MPI resources while the application driver executes collectively.
int main(int argc, char **argv) {
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());

    // The driver handles errors before these collective resources are destroyed.
    return krona::apps::mesh_generator::run(argc, argv, communicator.view());
}
