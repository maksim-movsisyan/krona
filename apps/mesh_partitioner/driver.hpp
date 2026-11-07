#pragma once

#include <krona/mpi/communicator.hpp>

namespace krona::apps::mesh_partitioner {

/// @brief Selects a common CLI action and runs the partitioning pipeline on all ranks.
[[nodiscard]] int run(int argc, char* const* argv, mpi::CommView comm) noexcept;

} // namespace krona::apps::mesh_partitioner
