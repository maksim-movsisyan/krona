#include <krona/mesh/generation/topology.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <krona/mesh/generation/hex.hpp>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>

#include <krona/mesh/generation/grading.hpp>
#include <krona/mesh/generation/tfi.hpp>
#include <krona/mpi/error.hpp>

namespace krona::mesh::generation {

namespace {

/// @brief Reports an invalid topology and terminates the MPI job.
[[noreturn]] void fail(const std::string &what) 
{
    mpi::fatal("topology: " + what);
}

/// @brief Largest supported count, with room for the writer's one-past-end ID.
constexpr auto max_id = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - 1;
/// @brief Adds mesh counts without overflowing signed CGNS numbering.
std::uint64_t add(std::uint64_t a, std::uint64_t b) 
{
    mpi::require(a <= max_id && b <= max_id - a, "Mesh ID total exceeds signed 64-bit range");
    return a + b;
}
/// @brief Multiplies mesh extents within size_t and signed CGNS limits.
std::uint64_t product(std::size_t a, std::size_t b) {
    const auto value = mpi::checked_product(a, b, "Mesh size overflows size_t");
    mpi::require(value <= max_id, "Mesh size exceeds signed 64-bit range");
    return value;
}
} // namespace

/// @brief Builds deterministic node IDs and validates shared interfaces locally.
void MacroTopology::build(const GeneratorConfig &config, mpi::CommView comm) noexcept {
    mpi::guarded([&] {
        mpi::require(!comm.is_null(), "Topology construction requires a live communicator");
        validate_generator_config(config);

        build_blocks(config);
        assign_node_ids(config.vertices.size());
        build_node_sources(config);
        build_boundary_patches(config, comm);
    });
}

/// @brief Resets topology and resolves block edges, faces and grading.
void MacroTopology::build_blocks(const GeneratorConfig &config) {
    const std::size_t num_blocks = config.blocks.size();
    const std::size_t num_macro_vertices = config.vertices.size();

    // build() may be called more than once on the same object so clear all data first.
    edges_.clear();
    faces_.clear();
    boundary_patches_.clear();
    blocks_.clear();
    node_ranges_.clear();
    vertex_sources_.assign(num_macro_vertices, NodeSource{});
    blocks_.resize(num_blocks);
    cell_offsets_.assign(num_blocks + 1, 0);

    // count total number of cells and create offsets (global numeration)
    total_cells_ = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        cell_offsets_[b] = total_cells_;
        const auto &blk = config.blocks[b];
        total_cells_ =
            add(total_cells_, product(static_cast<std::size_t>(product(blk.cells[0], blk.cells[1])), blk.cells[2]));
    } // end loop over all blocks
    cell_offsets_[num_blocks] = total_cells_;

    // build macro block topology 
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const auto &blk = config.blocks[b];
        auto &bt = blocks_[b];
        bt.vertices = blk.vertices;

        // loop over all block edges
        for (std::size_t e = 0; e < 12; ++e) {
            // get edge vertices (directed) and number of cells among each edge
            const std::size_t u = blk.vertices[hex::edges[e].first];
            const std::size_t v = blk.vertices[hex::edges[e].second];
            const std::size_t cells = (e < 4) ? blk.cells[0] : ((e < 8) ? blk.cells[1] : blk.cells[2]);

            const std::size_t edge_idx = find_or_add_edge(u, v, cells);

            // write edge index and weather edge is reversed 
            bt.edges[e] = {edge_idx, (u > v)};
        } // end loop over all block edges

        // loop over all block faces
        for (std::size_t f = 0; f < 6; ++f) {
            // get vertices forming a face and face layout
            const auto q_verts = hex::face_vertices(blk.vertices, f);
            const auto &layout = hex::layouts[f];

            // get number of cells along local 2-d axes
            const auto cu = blk.cells[layout.u];
            const auto cv = blk.cells[layout.v];

            auto &ref = bt.faces[f];
            ref.index = find_or_add_face(q_verts, cu, cv);

            // get face by its index (new if it was just add)
            const auto &canonical = faces_[ref.index];

            // find position of 0 and 1 vertices of curent face in canonical face
            const auto a = static_cast<std::size_t>(
                std::find(canonical.vertices.begin(), canonical.vertices.end(), q_verts[0]) - canonical.vertices.begin());
            const auto b1 = static_cast<std::size_t>(
                std::find(canonical.vertices.begin(), canonical.vertices.end(), q_verts[1]) - canonical.vertices.begin());

            // The canonical u-axis runs from corner 0 to corner 1.
            // If the current face's local u-axis runs along canonical edge
            // 1 <-> 2 or 0 <-> 3, it follows the canonical v-axis, so swap u and v.
            const bool a_u = a == 1 || a == 2;
            const bool b_u = b1 == 1 || b1 == 2;
            ref.swap_uv = a_u == b_u;

            // If the starting corner is on the canonical u = 1 side,
            // count canonical u in the reverse direction after swapping axes.
            ref.reverse_u = a_u;

            // Likewise, corners 2 and 3 lie on the canonical v = 1 side.
            ref.reverse_v = a >= 2;

            mpi::require(canonical.cells_u == (ref.swap_uv ? cv : cu) &&
                             canonical.cells_v == (ref.swap_uv ? cu : cv),
                         "Shared face cell counts disagree after axis mapping");
        } // end loop over all block faces

        // get number of INNER nodes (witout cells, located on block vertices/edges/faces)
        bt.vol_count = static_cast<std::size_t>(
            product(static_cast<std::size_t>(product(blk.cells[0] - 1, blk.cells[1] - 1)), blk.cells[2] - 1));

        // creating distributions along edges
        bt.dist_x = Grading::compute_distribution(blk.cells[0], blk.grading_type[0], blk.grading[0]);
        bt.dist_y = Grading::compute_distribution(blk.cells[1], blk.grading_type[1], blk.grading[1]);
        bt.dist_z = Grading::compute_distribution(blk.cells[2], blk.grading_type[2], blk.grading[2]);
    } // end loop over all macro blocks
}

/// @brief Numbers vertices, edge interiors, face interiors and block interiors.
void MacroTopology::assign_node_ids(std::size_t num_macro_vertices) {
    const auto num_blocks = blocks_.size();

    // Allocate 1-based (!!) unique IDs
    std::uint64_t current_id = 1;

    // Strata 0: Macro-vertices [1, num_macro_vertices]
    current_id = add(current_id, num_macro_vertices);

    // Strata 1: Unique Edges
    for (auto &edge : edges_) {
        // get number of inner nodes, except forming vertices
        const std::size_t count = (edge.num_cells > 1) ? (edge.num_cells - 1) : 0;
        if (count > 0) {
            edge.global_start_id = current_id;
            current_id = add(current_id, count);
        }
    } // end loop over all edges

    // Strata 2: Unique Faces
    for (auto &face : faces_) {
        // get number of inner nodes, except forming vertices and edges
        const std::size_t count = (face.cells_u > 1 && face.cells_v > 1)
                                      ? static_cast<std::size_t>(product(face.cells_u - 1, face.cells_v - 1))
                                      : 0;
        if (count > 0) {
            face.global_start_id = current_id;
            current_id = add(current_id, count);
        }
    } // end loop over all faces

    // Strata 3: Block Volumes
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (blocks_[b].vol_count > 0) {
            blocks_[b].vol_start_id = current_id;
            current_id = add(current_id, blocks_[b].vol_count);
        }
    } // end loop over all blocks

    total_nodes_ = current_id - 1;
}

/// @brief Builds compact coordinate ranges and checks shared edge distributions.
void MacroTopology::build_node_sources(const GeneratorConfig &config) {
    const auto num_blocks = config.blocks.size();

    // Construct compact coordinate sources; shared edges/faces keep their
    // first owner, which defines canonical node ordering.
    std::vector<NodeRange> edge_sources(edges_.size());
    std::vector<NodeRange> face_sources(faces_.size());
    node_ranges_.reserve(edges_.size() + faces_.size() + num_blocks);

    // loop over all blocks
    for (std::size_t b = 0; b < num_blocks; ++b) {
        // get block references (from topology and config)
        const auto &blk = config.blocks[b];
        const auto &bt = blocks_[b];
        const std::array<std::size_t, 3> n = {blk.cells[0], blk.cells[1], blk.cells[2]};

        // loop over all block vertices
        for (std::size_t c = 0; c < 8; ++c) {
            // Shared vertices are overwritten by later blocks.
            // Keep one block-local source for evaluating each macro vertex's coordinates.
            vertex_sources_[blk.vertices[c]] = {b, hex::corners[c][0] * n[0], hex::corners[c][1] * n[1],
                                                hex::corners[c][2] * n[2]};
        } // end loop over all block vertices

        // loop over all block edges
        for (std::size_t e = 0; e < 12; ++e) {
            const auto [idx, reversed] = bt.edges[e];
            if (edges_[idx].num_cells <= 1) continue;

            NodeRange r;
            r.first = edges_[idx].global_start_id;
            r.count = edges_[idx].num_cells - 1;
            r.block_idx = b;
            r.reversed = reversed;
            r.axis = e / 4;

            // Locate the edge's starting corner in the block's (i,j,k) node lattice.
            const auto &corner = hex::corners[hex::edges[e].first];
            for (std::size_t d = 0; d < 3; ++d)
                r.origin[d] = corner[d] * n[d];

            // Skip the corner: the range contains only interior edge nodes.
            const std::size_t axis = e / 4;
            r.origin[axis] = 1;
            r.shape[axis] = n[axis] - 1;

            // Add node range or check consistency with already existed
            if (edge_sources[idx].count == 0) {
                edge_sources[idx] = r;
            } else {
                const auto &prior = edge_sources[idx];
                const auto &old = blocks_[prior.block_idx];
                const auto old_axis = prior.axis;
                const auto &old_dist = old_axis == 0 ? old.dist_x : (old_axis == 1 ? old.dist_y : old.dist_z);
                const auto &new_dist = axis == 0 ? bt.dist_x : (axis == 1 ? bt.dist_y : bt.dist_z);
                for (std::size_t t = 1; t < edges_[idx].num_cells; ++t) {
                    const double a = prior.reversed ? 1 - old_dist[old_dist.size() - 1 - t] : old_dist[t];
                    const double c = reversed ? 1 - new_dist[new_dist.size() - 1 - t] : new_dist[t];
                    if (std::abs(a - c) > 64 * std::numeric_limits<double>::epsilon())
                        fail("Shared edge has incompatible grading (accounting for reversed orientation)");
                }
            }
        } // end loop over all block edges

        // loop over all block ffaces
        for (std::size_t f = 0; f < 6; ++f) {
            // get macro face reference
            const auto idx = bt.faces[f].index;
            const auto &face = faces_[idx];

            if (face.cells_u <= 1 || face.cells_v <= 1) continue;

            NodeRange r;
            r.first = face.global_start_id;
            r.count = (face.cells_u - 1) * (face.cells_v - 1);
            r.block_idx = b;
            r.origin = {1, 1, 1};
            r.shape = {n[0] - 1, n[1] - 1, n[2] - 1};

            const auto normal = hex::layouts[f].normal;
            const bool upper = hex::layouts[f].upper;
            r.origin[normal] = upper ? n[normal] : 0;
            r.shape[normal] = 1;

            // The first block defines canonical face axes and ID ordering.
            if (face_sources[idx].count == 0)
                face_sources[idx] = r;
        } // end loop over all block faces

        if (bt.vol_count > 0) {
            NodeRange r;
            r.first = bt.vol_start_id;
            r.count = bt.vol_count;
            r.block_idx = b;
            r.origin = {1, 1, 1};
            r.shape = {n[0] - 1, n[1] - 1, n[2] - 1};
            node_ranges_.push_back(r);
        }
    } // end loop over all blocks

    // add sources to node_ranges_
    for (const auto &r : edge_sources)
        if (r.count > 0) node_ranges_.push_back(r);

    for (const auto &r : face_sources)
        if (r.count > 0) node_ranges_.push_back(r);
    
    // sort by global node indices
    std::sort(node_ranges_.begin(), node_ranges_.end(),
              [](const NodeRange &a, const NodeRange &b) { return a.first < b.first; });
}

/// @brief Validates explicit patches and groups unassigned exterior faces.
void MacroTopology::build_boundary_patches(const GeneratorConfig &config, mpi::CommView comm) {
    // Resolve patch selections against unique macro faces once.
    std::map<hex::FaceKey, const MacroFaceDesc *> lookup;
    for (const auto &face : faces_)
        lookup.emplace(hex::face_key(face.vertices), &face);

    // check if all faces was assigned
    std::set<hex::FaceKey> assigned;
    for (const auto &patch : config.boundaries) {
        for (const auto &mf : patch.faces) {
            const auto key = hex::face_key(mf.vertices);
            const auto it = lookup.find(key);
            if (it == lookup.end())
                fail("patch '" + patch.name + "' references a face absent from the blocks");
            const auto &face = *it->second;
            mpi::require(face.incident_blocks == 1, "A boundary patch references an internal block face");
            mpi::require(assigned.insert(key).second, "A boundary face is assigned more than once");
            mpi::require(hex::same_perimeter(mf.vertices, face.vertices),
                         "Boundary vertices must follow the face perimeter");
        }
    }

    // Preserve explicit assignments and collect only unassigned exterior faces.
    boundary_patches_ = config.boundaries;
    BoundaryPatch automatic;
    std::uint64_t automatic_quads = 0;
    for (const auto &face : faces_) {
        if (face.incident_blocks != 1)
            continue;
        const auto key = hex::face_key(face.vertices);
        if (assigned.contains(key))
            continue;
        automatic.faces.push_back({face.vertices});
        automatic_quads = add(automatic_quads, product(face.cells_u, face.cells_v));
    }
    if (!automatic.faces.empty()) {
        automatic.name = "patch_automatic";
        const auto name_in_use = [&](std::string_view name) {
            return std::any_of(boundary_patches_.begin(), boundary_patches_.end(),
                               [&](const BoundaryPatch &patch) { return patch.name == name; });
        };
        for (std::size_t suffix = 1; name_in_use(automatic.name); ++suffix) {
            automatic.name = "patch_automatic_" + std::to_string(suffix);
            mpi::require(automatic.name.size() <= 32, "Automatic patch name exceeds CGNS name limit");
        }
        boundary_patches_.push_back(std::move(automatic));
        if (comm.rank() == 0) {
            const auto &patch = boundary_patches_.back();
            std::fprintf(stderr,
                         "[KRONA warning] %zu unassigned outer macro face(s) "
                         "(%llu boundary quads) were added to patch '%s'.\n",
                         patch.faces.size(), static_cast<unsigned long long>(automatic_quads),
                         patch.name.c_str());
        }
    }
}

/// @brief Finds a shared edge and requires a matching subdivision (u,v is vertices).
std::size_t MacroTopology::find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells) {
    // canonical orientation
    const std::size_t v_min = std::min(u, v);
    const std::size_t v_max = std::max(u, v);

    // loop over all existing edges
    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (edges_[i].v0 == v_min && edges_[i].v1 == v_max) {
            // check if number of cells are coincidence
            if (edges_[i].num_cells != cells) {
                fail("Shared edge (" + std::to_string(v_min) + "," + std::to_string(v_max) +
                     ") cell count mismatch: " + std::to_string(edges_[i].num_cells) + " vs " +
                     std::to_string(cells));
            }

            // return index of found edge
            return i;
        }
    } // end loop over all existing edges

    // if edge not found: add it and return index, global_start_id is pluged
    edges_.push_back({v_min, v_max, cells, 0});
    return edges_.size() - 1;
}

/// @brief Finds a shared face and checks its perimeter and incidence (cu and cv is number of cells along u,v axes).
std::size_t MacroTopology::find_or_add_face(const std::array<std::size_t, 4> &q_verts, std::size_t cu, std::size_t cv) {
    const auto sorted_q = hex::face_key(q_verts);

    // loop over all existing macro faces
    for (std::size_t i = 0; i < faces_.size(); ++i) {
        const auto existing = hex::face_key(faces_[i].vertices);
        if (existing == sorted_q) {
            mpi::require(++faces_[i].incident_blocks <= 2, "More than two blocks share a face");
            mpi::require(hex::same_perimeter(q_verts, faces_[i].vertices),
                         "Shared faces have different edge connectivity");
            // if face already exist return it index
            return i;
        }
    } // end loop over all existing macro faces

    // add new face, incident_block and global_start_id is pluged
    faces_.push_back({q_verts, 1, cu, cv, 0});
    return faces_.size() - 1;
}

/// @brief Maps a block lattice node to its unique one-based ID.
std::uint64_t MacroTopology::get_node_id(std::size_t b, std::size_t i, std::size_t j,
                                         std::size_t k) const noexcept {
    mpi::require(b < blocks_.size(), "Block index is out of bounds");
    const auto &bt = blocks_[b];
    mpi::require(i < bt.dist_x.size() && j < bt.dist_y.size() && k < bt.dist_z.size(),
                 "Block node index is out of bounds");

    // get total number of cells along each axis
    const std::size_t Nx = bt.dist_x.size() - 1;
    const std::size_t Ny = bt.dist_y.size() - 1;
    const std::size_t Nz = bt.dist_z.size() - 1;

    // check if node lying on block marko vertex/edge/face
    const bool on_x0 = (i == 0), on_x1 = (i == Nx);
    const bool on_y0 = (j == 0), on_y1 = (j == Ny);
    const bool on_z0 = (k == 0), on_z1 = (k == Nz);

    // check "boundary" factor of node
    const int bnd_count = (on_x0 || on_x1 ? 1 : 0) + (on_y0 || on_y1 ? 1 : 0) + (on_z0 || on_z1 ? 1 : 0);

    // 1. Macro-vertex
    if (bnd_count == 3) {
        const std::size_t corner = (on_z0 ? 0 : 4) + (on_y0 ? (on_x0 ? 0 : 1) : (on_x0 ? 3 : 2));
        return static_cast<std::uint64_t>(bt.vertices[corner] + 1);
    }

    // 2. Macro-edge
    if (bnd_count == 2) {
        std::size_t e = 0;
        std::size_t step = 0;
        std::size_t max_step = 0;

        if (!on_x0 && !on_x1) {
            step = i;
            max_step = Nx;
            if (on_y0 && on_z0)
                e = 0;
            else if (on_y1 && on_z0)
                e = 1;
            else if (on_y0 && on_z1)
                e = 2;
            else
                e = 3;
        } else if (!on_y0 && !on_y1) {
            step = j;
            max_step = Ny;
            if (on_x0 && on_z0)
                e = 4;
            else if (on_x1 && on_z0)
                e = 5;
            else if (on_x0 && on_z1)
                e = 6;
            else
                e = 7;
        } else {
            step = k;
            max_step = Nz;
            if (on_x0 && on_y0)
                e = 8;
            else if (on_x1 && on_y0)
                e = 9;
            else if (on_x1 && on_y1)
                e = 10;
            else
                e = 11;
        }

        const auto [edge_idx, is_reversed] = bt.edges[e];
        const auto &edge = edges_[edge_idx];
        const std::size_t interior_idx = is_reversed ? (max_step - step - 1) : (step - 1);
        return edge.global_start_id + interior_idx;
    }

    // 3. Macro-face
    if (bnd_count == 1) {
        std::size_t f = 0;
        std::size_t u = 0, v = 0;

        if (on_z0) {
            f = 0;
            u = i;
            v = j;
        } else if (on_z1) {
            f = 5;
            u = i;
            v = j;
        } else if (on_y0) {
            f = 1;
            u = i;
            v = k;
        } else if (on_y1) {
            f = 3;
            u = i;
            v = k;
        } else if (on_x0) {
            f = 4;
            u = j;
            v = k;
        } else {
            f = 2;
            u = j;
            v = k;
        }

        const auto &ref = bt.faces[f];
        const auto &face = faces_[ref.index];
        if (ref.swap_uv)
            std::swap(u, v);
        if (ref.reverse_u)
            u = face.cells_u - u;
        if (ref.reverse_v)
            v = face.cells_v - v;
        return face.global_start_id + (v - 1) * (face.cells_u - 1) + (u - 1);
    }

    // 4. Volume interior
    const std::uint64_t v_idx = static_cast<std::uint64_t>(k - 1) * (Nx - 1) * (Ny - 1) +
                                static_cast<std::uint64_t>(j - 1) * (Nx - 1) +
                                static_cast<std::uint64_t>(i - 1);
    return bt.vol_start_id + v_idx;
}

/// @brief Finds the compact coordinate descriptor containing a node ID.
const MacroTopology::NodeRange &MacroTopology::find_node_range(std::uint64_t id) const noexcept {
    auto it = std::upper_bound(node_ranges_.begin(), node_ranges_.end(), id,
                               [](std::uint64_t value, const NodeRange &r) { return value < r.first; });
    mpi::require(it != node_ranges_.begin(), "Node ID is outside topology ranges");
    --it;
    mpi::require(id - it->first < it->count, "Node ID is outside topology ranges");
    return *it;
}

/// @brief Evaluates one global node using its canonical block source.
Vec3 MacroTopology::evaluate_node_coord(std::uint64_t global_node_id,
                                        const GeneratorConfig &config) const noexcept {
    mpi::require(global_node_id > 0 && global_node_id <= total_nodes_, "Node ID is out of bounds");
    NodeSource src;
    if (global_node_id <= vertex_sources_.size()) {
        src = vertex_sources_[global_node_id - 1];
    } else {
        const auto &r = find_node_range(global_node_id);
        std::uint64_t q = global_node_id - r.first;
        if (r.reversed)
            q = r.count - 1 - q;
        src.block_idx = r.block_idx;
        src.i = r.origin[0] + static_cast<std::size_t>(q % r.shape[0]);
        q /= r.shape[0];
        src.j = r.origin[1] + static_cast<std::size_t>(q % r.shape[1]);
        src.k = r.origin[2] + static_cast<std::size_t>(q / r.shape[1]);
    }
    const auto &bt = blocks_[src.block_idx];
    const auto &blk = config.blocks[src.block_idx];
    std::array<Vec3, 8> corners;
    for (std::size_t c = 0; c < 8; ++c)
        corners[c] = config.vertices[blk.vertices[c]];
    return TFI::interpolate_hex(corners, bt.dist_x[src.i], bt.dist_y[src.j], bt.dist_z[src.k]);
}

/// @brief Evaluates a consecutive node interval while reusing range lookups.
void MacroTopology::evaluate_node_coords(std::uint64_t first_node_id, std::size_t count,
                                         const GeneratorConfig &config, double *x, double *y,
                                         double *z) const noexcept {
    if (count == 0)
        return;
    mpi::require(x && y && z, "Coordinate output buffers must be non-null");
    mpi::require(first_node_id > 0 && first_node_id <= total_nodes_ &&
                     count <= total_nodes_ - first_node_id + 1,
                 "Coordinate interval is out of bounds");
    std::size_t out = 0;
    while (out < count && first_node_id <= vertex_sources_.size()) {
        const auto p = evaluate_node_coord(first_node_id++, config);
        x[out] = p.x;
        y[out] = p.y;
        z[out] = p.z;
        ++out;
    }
    if (out == count)
        return;
    // One lookup per batch, then walk adjacent descriptors. Decode only the
    // first index of each span; advance i/j/k without integer division.
    const NodeRange *range = &find_node_range(first_node_id);
    while (out < count) {
        const auto &r = *range;
        const auto &bt = blocks_[r.block_idx];
        const auto &blk = config.blocks[r.block_idx];
        std::array<Vec3, 8> corners;
        for (std::size_t c = 0; c < 8; ++c)
            corners[c] = config.vertices[blk.vertices[c]];
        const std::uint64_t offset = first_node_id - r.first;
        const std::size_t span =
            static_cast<std::size_t>(std::min<std::uint64_t>(count - out, r.count - offset));
        std::uint64_t q = r.reversed ? r.count - 1 - offset : offset;
        std::size_t i = static_cast<std::size_t>(q % r.shape[0]);
        q /= r.shape[0];
        std::size_t j = static_cast<std::size_t>(q % r.shape[1]);
        std::size_t k = static_cast<std::size_t>(q / r.shape[1]);
        for (std::size_t t = 0; t < span; ++t, ++out) {
            const auto p = TFI::interpolate_hex(corners, bt.dist_x[r.origin[0] + i],
                                                bt.dist_y[r.origin[1] + j], bt.dist_z[r.origin[2] + k]);
            x[out] = p.x;
            y[out] = p.y;
            z[out] = p.z;
            if (!r.reversed) {
                if (++i == r.shape[0]) {
                    i = 0;
                    if (++j == r.shape[1]) {
                        j = 0;
                        ++k;
                    }
                }
            } else {
                if (i > 0)
                    --i;
                else {
                    i = r.shape[0] - 1;
                    if (j > 0)
                        --j;
                    else {
                        j = r.shape[1] - 1;
                        if (k > 0)
                            --k;
                    }
                }
            }
        }
        first_node_id += span;
        ++range;
    }
}

/// @brief Maps a zero-based global cell ID to block-local indices.
void MacroTopology::get_cell_block_and_ijk(std::uint64_t global_cell_id, std::size_t &block_idx,
                                           std::size_t &i, std::size_t &j, std::size_t &k) const noexcept {
    mpi::require(global_cell_id < total_cells_, "Cell ID must be zero-based and within the mesh");
    auto it = std::upper_bound(cell_offsets_.begin(), cell_offsets_.end(), global_cell_id);
    block_idx = static_cast<std::size_t>(std::distance(cell_offsets_.begin(), it) - 1);

    const std::uint64_t local_c = global_cell_id - cell_offsets_[block_idx];
    const auto &bt = blocks_[block_idx];
    const std::size_t Nx = bt.dist_x.size() - 1;
    const std::size_t Ny = bt.dist_y.size() - 1;

    k = static_cast<std::size_t>(local_c / (Nx * Ny));
    j = static_cast<std::size_t>((local_c % (Nx * Ny)) / Nx);
    i = static_cast<std::size_t>(local_c % Nx);
}

/// @brief Returns one block cell in CGNS HEXA_8 corner order (ijk - cell indices).
std::array<std::uint64_t, 8> MacroTopology::get_block_cell_nodes(std::size_t b, std::size_t i, std::size_t j,
                                                                 std::size_t k) const noexcept {
    mpi::require(b < blocks_.size(), "Block index is out of bounds");
    const auto &bt = blocks_[b];
    mpi::require(i < bt.dist_x.size() - 1 && j < bt.dist_y.size() - 1 && k < bt.dist_z.size() - 1,
                 "Block cell index is out of bounds");
                 
    // Apply the common HEXA_8 corner offsets to the cell origin.
    std::array<std::uint64_t, 8> nodes{};
    for (std::size_t corner = 0; corner < nodes.size(); ++corner) {
        const auto &offset = hex::corners[corner];
        nodes[corner] = get_node_id(b, i + offset[0], j + offset[1], k + offset[2]);
    }
    return nodes;
}

/// @brief Returns a global cell in CGNS HEXA_8 corner order.
std::array<std::uint64_t, 8> MacroTopology::get_cell_nodes(std::uint64_t global_cell_id) const noexcept {
    std::size_t b = 0, i = 0, j = 0, k = 0;
    get_cell_block_and_ijk(global_cell_id, b, i, j, k);
    return get_block_cell_nodes(b, i, j, k);
}

} // namespace krona::mesh::generation