#pragma once

#include <krona/mpi/communicator.hpp>

namespace krona::apps::mesh_generator {

/// @brief Selects a common CLI action and runs the generator on all ranks.
/// MPI must be active; comm must be live. Errors are guarded before returning.
[[nodiscard]] int run(int argc, char *const *argv, mpi::CommView comm) noexcept;

} // namespace krona::apps::mesh_generator
