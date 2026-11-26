#include "flow_steady/driver.hpp"

#include <krona/mpi/runtime.hpp>

/// @brief Owns MPI while the steady flow driver executes collectively.
int main(int argc, char** argv) {
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
    return krona::apps::flow_steady::run(argc, argv, communicator.view());
}
