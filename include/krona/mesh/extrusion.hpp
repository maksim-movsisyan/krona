#pragma once

#include "krona/mesh/raw_mesh.hpp"
#include "krona/mpi/communicator.hpp"

#include <cstddef>
#include <limits>
#include <string>

namespace krona::mesh
{

struct ExtrusionOptions
{
    double thickness_fraction = 0.1;
    double planarity_tolerance =
        64.0 * static_cast<double>(std::numeric_limits<GeometryReal>::epsilon());
    std::size_t batch_elements = 65536;
    std::string bottom_patch = "Z_MIN";
    std::string top_patch = "Z_MAX";
};

struct ExtrusionInfo
{
    GeometryReal thickness = 0;
    int normal_axis = 2; // 0=X, 1=Y, 2=Z.
    int normal_sign = 1;
};

// Collective, in-place extrusion of an axis-aligned planar TRI/QUAD import snapshot.
// h = thickness_fraction * max(in-plane bbox spans); original coordinates are preserved.
// XY -> +Z, XZ -> -Y, YZ -> +X. Bottom IDs stay unchanged; top IDs are shifted by N.
// Cell ownership is preserved; nodes and boundary ordinals are balanced again.
// Options/metadata must agree across ranks. Errors abort comm.abort_scope().
[[nodiscard]] ExtrusionInfo extrude_one_layer(RawMesh &mesh, mpi::CommView comm,
                                              const ExtrusionOptions &options = {});

} // namespace krona::mesh
