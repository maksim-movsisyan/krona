#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <krona/mesh/generation/config.hpp>
#include <krona/mpi/communicator.hpp>

namespace krona::mesh::generation {

/// @brief Identifies a node in one block's integer lattice.
struct NodeSource {
    std::size_t block_idx{0};
    std::size_t i{0};
    std::size_t j{0};
    std::size_t k{0};
};

/// @brief Stores a unique edge in increasing macro-vertex ID order.
struct MacroEdge {
    std::size_t v0{0};
    std::size_t v1{0};
    std::size_t num_cells{0};
    std::uint64_t global_start_id{0};
};

/// @brief Stores a unique face in the first block's parametric orientation.
struct MacroFaceDesc {
    std::array<std::size_t, 4> vertices{};
    std::size_t incident_blocks{1};
    std::size_t cells_u{0};
    std::size_t cells_v{0};
    std::uint64_t global_start_id{0};
};

/// @brief Owns shared node numbering and compact coordinate descriptors.
class MacroTopology {
public:
    /// @brief Builds and validates topology locally; comm selects the warning rank.
    void build(const GeneratorConfig &config, mpi::CommView comm) noexcept;

    /// @brief Returns the unique one-based ID of a valid block lattice node.
    [[nodiscard]] std::uint64_t get_node_id(std::size_t block_idx, std::size_t i, std::size_t j,
                                            std::size_t k) const noexcept;

    /// @brief Evaluates a node using the unchanged configuration used by build.
    [[nodiscard]] Vec3 evaluate_node_coord(std::uint64_t global_node_id,
                                           const GeneratorConfig &config) const noexcept;

    /// @brief Evaluates a consecutive one-based node interval into caller arrays.
    /// The configuration must match build; each output holds at least count entries.
    void evaluate_node_coords(std::uint64_t first_node_id, std::size_t count, const GeneratorConfig &config,
                              double *x, double *y, double *z) const noexcept;

    /// @brief Returns a zero-based global cell's one-based HEXA_8 node IDs.
    [[nodiscard]] std::array<std::uint64_t, 8> get_cell_nodes(std::uint64_t global_cell_id) const noexcept;

    /// @brief Returns a valid block cell's one-based HEXA_8 node IDs.
    [[nodiscard]] std::array<std::uint64_t, 8>
    get_block_cell_nodes(std::size_t block_idx, std::size_t i, std::size_t j, std::size_t k) const noexcept;

    /// @brief Decodes a zero-based global cell ID into block and lattice indices.
    void get_cell_block_and_ijk(std::uint64_t global_cell_id, std::size_t &block_idx, std::size_t &i,
                                std::size_t &j, std::size_t &k) const noexcept;

    /// @brief Borrows validated user patches followed by any automatic patch.
    [[nodiscard]] const std::vector<BoundaryPatch> &boundary_patches() const noexcept {
        return boundary_patches_;
    }

    /// @brief Returns the total number of unique nodes.
    [[nodiscard]] std::uint64_t total_nodes() const noexcept { return total_nodes_; }
    /// @brief Returns the total number of volume cells.
    [[nodiscard]] std::uint64_t total_cells() const noexcept { return total_cells_; }

private:
    /// @brief Resets storage and resolves each block's shared entities.
    void build_blocks(const GeneratorConfig &config);

    /// @brief Allocates deterministic one-based node-ID intervals.
    void assign_node_ids(std::size_t num_macro_vertices);

    /// @brief Builds coordinate descriptors and checks shared grading.
    void build_node_sources(const GeneratorConfig &config);

    /// @brief Validates patches and collects omitted exterior faces.
    void build_boundary_patches(const GeneratorConfig &config, mpi::CommView comm);

    std::uint64_t total_nodes_{0};
    std::uint64_t total_cells_{0};

    std::vector<MacroEdge> edges_;
    std::vector<MacroFaceDesc> faces_;
    std::vector<BoundaryPatch> boundary_patches_;

    /// @brief Stores one block's shared-entity references and node distributions.
    struct BlockTopo {
        std::array<std::size_t, 8> vertices{};
        std::array<std::pair<std::size_t, bool>, 12> edges;
        /// @brief Maps block face axes to the canonical face orientation.
        struct FaceRef {
            std::size_t index{0};
            bool swap_uv{false};
            bool reverse_u{false};
            bool reverse_v{false};
        };
        std::array<FaceRef, 6> faces;
        std::uint64_t vol_start_id{0};
        std::size_t vol_count{0};
        std::vector<double> dist_x;
        std::vector<double> dist_y;
        std::vector<double> dist_z;
    };
    std::vector<BlockTopo> blocks_;
    std::vector<std::uint64_t> cell_offsets_;
    // One descriptor per nonempty macro edge, face, or block interior.
    // No storage proportional to the number of generated nodes.
    /// @brief Describes one consecutive node-ID interval with a lattice box.
    struct NodeRange {
        std::uint64_t first{0};
        std::uint64_t count{0};
        std::size_t block_idx{0};
        std::array<std::size_t, 3> origin{};
        std::array<std::size_t, 3> shape{1, 1, 1};
        bool reversed{false};
        std::size_t axis{0};
    };
    std::vector<NodeSource> vertex_sources_;
    std::vector<NodeRange> node_ranges_;

    /// @brief Locates the compact descriptor containing a valid node ID.
    [[nodiscard]] const NodeRange &find_node_range(std::uint64_t id) const noexcept;

    /// @brief Reuses an edge or inserts its first occurrence.
    std::size_t find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells);
    /// @brief Reuses a face or inserts its first occurrence.
    std::size_t find_or_add_face(const std::array<std::size_t, 4> &quad_v, std::size_t cu, std::size_t cv);
};

} // namespace krona::mesh::generation