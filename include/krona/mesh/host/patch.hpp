#pragma once

#include <krona/mesh/types.hpp>
#include <string>

namespace krona::mesh::host {

/// @brief Stores source patch labels independently of solver boundary conditions. 
struct Patch {
    std::string name;
    std::string family;
    std::string source_type; // Import label, not a solver boundary condition.
};

} // namespace krona::mesh::host
