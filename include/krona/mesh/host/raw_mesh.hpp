#pragma once

#include "krona/config.hpp"
#include "krona/distribution/block.hpp"
#include <krona/mesh/element.hpp>
#include <krona/mesh/host/patch.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace krona::mesh::host
{

struct ElementTable
{
    std::vector<CellType> types;
    std::vector<std::size_t> offsets{0};
    std::vector<GlobalIndex> node_ids;

    /// @brief Returns the number of elements in this CSR table.
    [[nodiscard]] std::size_t size() const noexcept { return types.size(); }
    
    /// @brief Borrows an element's oriented global-node connectivity.
    [[nodiscard]] std::span<const GlobalIndex> nodes(std::size_t element) const noexcept
    {
        assert(element < size() && offsets.size() == size() + 1);
        // Connectivity stays in global IDs independently of coordinate ownership.
        return std::span<const GlobalIndex>{node_ids}.subspan(
            offsets[element], offsets[element + 1] - offsets[element]);
    }
};

struct Coordinates
{
    std::vector<GeometryReal> x, y, z;

    /// @brief Returns the number of locally owned coordinate triples.
    [[nodiscard]] std::size_t size() const noexcept { return x.size(); }
};

struct BoundaryTable
{
    ElementTable elements;
    std::vector<GlobalIndex> global_ids; // Sparse source element IDs, zero-based.
    std::vector<PatchId> patch_ids;      // May contain invalid_patch_id.

    /// @brief Returns the number of locally owned boundary elements.
    [[nodiscard]] std::size_t size() const noexcept { return elements.size(); }
};

struct ImportDistribution
{
    distribution::BlockDistribution<GlobalIndex> cells;
    distribution::BlockDistribution<GlobalIndex> nodes;
    distribution::BlockDistribution<GlobalIndex> boundary_ordinals; // Dense import order, not source IDs.
    int rank = 0;
};

// Import snapshot. Block ownership applies until migration or topology changes.
// Cell/node IDs are zero-based; coordinates contain only the owned node slice.
struct RawMesh
{
    int cell_dimension = 0;
    int physical_dimension = 0;
    ImportDistribution distribution;
    ElementTable cells;
    Coordinates coordinates;
    BoundaryTable boundary;
    std::vector<Patch> patches; // Small replicated metadata; never BC point lists.

    /// @brief Resolves a cell ID while this import's block ownership is unchanged.
    [[nodiscard]] GlobalIndex global_cell_id(LocalIndex local) const noexcept
    {
        assert(local >= 0 && static_cast<std::size_t>(local) < cells.size());
        return distribution.cells.begin(distribution.rank) + static_cast<GlobalIndex>(local);
    }
    /// @brief Resolves a node ID while this import's block ownership is unchanged.
    [[nodiscard]] GlobalIndex global_node_id(LocalIndex local) const noexcept
    {
        assert(local >= 0 && static_cast<std::size_t>(local) < coordinates.size());
        return distribution.nodes.begin(distribution.rank) + static_cast<GlobalIndex>(local);
    }
};

} // namespace krona::mesh::host