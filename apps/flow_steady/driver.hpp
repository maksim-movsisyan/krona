#pragma once

#include <krona/mpi/communicator.hpp>

namespace krona::apps::flow_steady {

/// @brief Selects a common CLI action and runs the selected steady compressible solver on all ranks.
[[nodiscard]] int run(int argc, char* const* argv, mpi::CommView comm) noexcept;

} // namespace krona::apps::flow_steady
