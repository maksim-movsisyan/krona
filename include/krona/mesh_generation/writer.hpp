#pragma once

#include <mpi.h>
#include <string>

#include "krona/mesh_generation/config.hpp"
#include "krona/mesh_generation/topology.hpp"

namespace krona::mesh_generation {

// Writes the complete unstructured mesh to CGNS collectively across all MPI ranks
void write(const std::string& filepath,
           const GeneratorConfig& config,
           const MacroTopology& topo,
           MPI_Comm comm);

} // namespace krona::mesh_generation