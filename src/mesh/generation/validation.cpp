#include <krona/mesh/generation/config.hpp>
#include <krona/mesh/generation/tfi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace krona::mesh::generation {
namespace {

/// @brief Reports an invalid generator configuration and terminates the MPI job.
[[noreturn]] void fail(const std::string &message) 
{
    mpi::fatal("Mesh generator: " + message);
}

} // namespace

/// @brief Checks physical geometry, block values and patch-name contracts locally.
void validate_generator_config(const GeneratorConfig &cfg) noexcept 
{
    mpi::guarded([&] {
        mpi::require(!cfg.vertices.empty() && !cfg.blocks.empty(), "Mesh needs vertices and blocks");
        mpi::require(std::isfinite(cfg.scale) && cfg.scale > 0, "Mesh scale must be finite and positive");
        for (const auto &v : cfg.vertices)
            mpi::require(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z),
                         "Mesh coordinates must be finite after scaling");
        const std::size_t num_vertices = cfg.vertices.size();
        std::vector<bool> used(num_vertices, false);
        std::set<std::array<std::size_t, 8>> block_vertices;

        // Validate local block geometry before building shared topology.
        for (const auto &block : cfg.blocks) {
            auto unique = block.vertices;
            std::sort(unique.begin(), unique.end());
            mpi::require(std::adjacent_find(unique.begin(), unique.end()) == unique.end(),
                         "A block must reference eight distinct vertices");
            mpi::require(block_vertices.insert(unique).second,
                         "Duplicate blocks reference the same eight vertices");

            std::array<Vec3, 8> corners;
            for (std::size_t i = 0; i < 8; ++i) {
                if (block.vertices[i] >= num_vertices)
                    fail("block '" + block.name + "': vertex index is out of bounds");
                used[block.vertices[i]] = true;
                corners[i] = cfg.vertices[block.vertices[i]];
            }

            // Check the trilinear map at all corners, edge/face centres and the centre.
            // This is a sampling check, not a proof of positivity everywhere.
            for (double u : {0.0, 0.5, 1.0})
                for (double v : {0.0, 0.5, 1.0})
                    for (double w : {0.0, 0.5, 1.0})
                        if (!TFI::positive_jacobian(corners, u, v, w))
                            fail("block '" + block.name + "': inverted or degenerate trilinear map; check HEXA_8 order");

            for (std::size_t d = 0; d < 3; ++d) {
                mpi::require(to_string(block.grading_type[d]) != "unknown", "Invalid grading type");
                mpi::require(block.cells[d] > 0 && block.cells[d] < std::numeric_limits<std::size_t>::max(),
                             "Cell counts must be positive and leave room for an endpoint");
                if (!std::isfinite(block.grading[d]) || (block.grading_type[d] != GradingType::UNIFORM && block.grading[d] <= 0))
                    fail("block '" + block.name + "': grading must be finite and positive for nonuniform distributions");
            }
        }
        mpi::require(std::all_of(used.begin(), used.end(), [](bool v) { return v; }),
                     "Every macro vertex must be used by a block");

        // CGNS section names are validated before creating an output file.
        std::set<std::string> names;
        for (const auto &patch : cfg.boundaries) {
            if (patch.name.empty() || patch.name.size() > 32 || patch.name == "." || patch.name == ".." ||
                patch.name.find('/') != std::string::npos || patch.name.find('\0') != std::string::npos ||
                patch.name == "Volume_HEXA" || patch.name == "GridCoordinates" || patch.name == "ZoneBC" ||
                patch.name == "ZoneType" || !names.insert(patch.name).second)
                fail("invalid, duplicate or reserved CGNS boundary name '" + patch.name + "'");
            mpi::require(!patch.faces.empty(), "Boundary patch must contain faces");
            for (const auto &face : patch.faces) {
                auto unique = face.vertices;
                std::sort(unique.begin(), unique.end());
                if (unique.back() >= num_vertices ||
                    std::adjacent_find(unique.begin(), unique.end()) != unique.end())
                    fail("patch '" + patch.name + "': face needs four distinct valid vertex indices");
            }
        }
    });
}

} // namespace krona::mesh::generation
