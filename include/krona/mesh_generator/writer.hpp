#pragma once

#include <mpi.h>
#include <string>

#include "krona/mesh_generator/config.hpp"
#include "krona/mesh_generator/topology.hpp"

namespace krona::mesh_generator {

// Writes the complete unstructured mesh to CGNS collectively across all MPI ranks
void write(const std::string& filepath,
           const GeneratorConfig& config,
           const MacroTopology& topo,
           MPI_Comm comm);

} // namespace krona::mesh_generator