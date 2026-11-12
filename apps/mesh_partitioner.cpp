#include "mesh_partitioner/driver.hpp"

#include <krona/mpi/communicator.hpp>
#include <krona/mpi/runtime.hpp>

/// @brief Owns MPI resources while the partitioner driver executes collectively. 
int main(int argc, char** argv) {
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
    return krona::apps::mesh_partitioner::run(argc, argv, communicator.view());
}
