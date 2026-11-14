#include "scalar_transport/driver.hpp"

#include <krona/mpi/runtime.hpp>

/// @brief Owns MPI while the scalar transport driver executes collectively.
int main(int argc, char** argv) {
    krona::mpi::Runtime runtime{argc, argv, krona::mpi::ThreadLevel::funneled};
    auto communicator = krona::mpi::Comm::duplicate(krona::mpi::CommView::world());
    return krona::apps::scalar_transport::run(argc, argv, communicator.view());
}
