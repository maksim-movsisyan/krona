#pragma once

#include <krona/mpi/communicator.hpp>

namespace krona::apps::scalar_transport {

/// @brief Selects a common CLI action and runs the transport solver on all ranks.
[[nodiscard]] int run(int argc, char* const* argv, mpi::CommView comm) noexcept;

} // namespace krona::apps::scalar_transport
