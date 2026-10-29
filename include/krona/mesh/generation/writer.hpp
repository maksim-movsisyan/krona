#pragma once

#include <krona/mpi/communicator.hpp>
#include <string>

#include "krona/mesh/generation/config.hpp"
#include "krona/mesh/generation/topology.hpp"

namespace krona::mesh::generation {

// Writes the complete unstructured mesh to CGNS collectively across all MPI ranks
void write(const std::string& filepath,
           const GeneratorConfig& config,
           const MacroTopology& topo,
           mpi::CommView comm) noexcept;

} // namespace krona::mesh::generation 