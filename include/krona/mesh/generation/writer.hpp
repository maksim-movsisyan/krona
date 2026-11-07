#pragma once

#include <krona/mpi/communicator.hpp>
#include <string>

#include <krona/mesh/generation/config.hpp>
#include <krona/mesh/generation/topology.hpp>

namespace krona::mesh::generation {

/// @brief Writes HEXA_8 cells and outward boundary quads collectively to CGNS.
/// Blocking; config must match topo, and all ranks must use the same output path.
void write(const std::string &filepath, const GeneratorConfig &config, const MacroTopology &topo,
           mpi::CommView comm) noexcept;

} // namespace krona::mesh::generation