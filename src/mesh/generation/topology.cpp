#include "krona/mesh/generation/topology.hpp"

#include <algorithm>
#include <cstdio>
#include <format>
#include <string>
#include <string_view>
#include <limits>
#include <set>
#include <cmath>

#include "krona/mesh/generation/grading.hpp"
#include "krona/mesh/generation/tfi.hpp"
#include "krona/mpi/error.hpp"

namespace krona::mesh::generation {

namespace {

[[noreturn]] void fail(const std::string& what) {
    mpi::fatal("topology: " + what);
}

// Keep totals representable by signed 64-bit CGNS IDs and leave room for
// the one-past-end ID used by the writer.
constexpr auto max_id = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - 1;
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    mpi::require(a <= max_id && b <= max_id - a, "Mesh ID total exceeds signed 64-bit range");
    return a + b;
}
std::uint64_t product(std::size_t a, std::size_t b) {
    const auto value = mpi::checked_product(a, b, "Mesh size overflows size_t");
    mpi::require(value <= max_id, "Mesh size exceeds signed 64-bit range");
    return value;
}
} // namespace

void MacroTopology::build(const GeneratorConfig& config, mpi::CommView comm) noexcept {
    mpi::guarded([&] {
        mpi::require(!comm.is_null(), "Topology construction requires a live communicator");
        validate_generator_config(config);
        const std::size_t num_blocks = config.blocks.size();
        const std::size_t num_macro_vertices = config.vertices.size();

        // build() may be called more than once on the same object.
        edges_.clear();
        faces_.clear();
        boundary_patches_.clear();
        blocks_.clear();
        node_ranges_.clear();
        vertex_sources_.assign(num_macro_vertices, NodeSource{});
        blocks_.resize(num_blocks);
        cell_offsets_.assign(num_blocks + 1, 0);

        total_cells_ = 0;
        for (std::size_t b = 0; b < num_blocks; ++b) {
            cell_offsets_[b] = total_cells_;
            const auto& blk = config.blocks[b];
            total_cells_ = add(total_cells_, product(static_cast<std::size_t>(product(blk.cells[0], blk.cells[1])), blk.cells[2]));
        }
        cell_offsets_[num_blocks] = total_cells_;

        // 12 edges in standard HEXA_8 (0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)...)
        // Ordered strictly from step 0 to step N along xi (0..3), eta (4..7), zeta (8..11)
        constexpr std::array<std::pair<std::size_t, std::size_t>, 12> hex_edge_pairs = {{
            {0, 1}, {3, 2}, {4, 5}, {7, 6}, // xi-direction
            {0, 3}, {1, 2}, {4, 7}, {5, 6}, // eta-direction
            {0, 4}, {1, 5}, {2, 6}, {3, 7}  // zeta-direction
        }};

        // 6 faces in parametric (u, v) order: (0,0), (1,0), (1,1), (0,1)
        constexpr std::array<std::array<std::size_t, 4>, 6> hex_face_quads = {{
            {0, 1, 2, 3}, // Face 0: bottom (zeta = 0, u = xi, v = eta)
            {0, 1, 5, 4}, // Face 1: front  (eta = 0,  u = xi, v = zeta)
            {1, 2, 6, 5}, // Face 2: right  (xi = 1,   u = eta, v = zeta)
            {3, 2, 6, 7}, // Face 3: back   (eta = 1,  u = xi, v = zeta)
            {0, 3, 7, 4}, // Face 4: left   (xi = 0,   u = eta, v = zeta)
            {4, 5, 6, 7}  // Face 5: top    (zeta = 1, u = xi, v = eta)
        }};

        for (std::size_t b = 0; b < num_blocks; ++b) {
            const auto& blk = config.blocks[b];
            auto& bt = blocks_[b];
            bt.vertices = blk.vertices;

            for (std::size_t e = 0; e < 12; ++e) {
                const std::size_t u = blk.vertices[hex_edge_pairs[e].first];
                const std::size_t v = blk.vertices[hex_edge_pairs[e].second];
                const std::size_t cells = (e < 4) ? blk.cells[0] : ((e < 8) ? blk.cells[1] : blk.cells[2]);

                const std::size_t edge_idx = find_or_add_edge(u, v, cells);
                bt.edges[e] = {edge_idx, (u > v)};
            }

            for (std::size_t f = 0; f < 6; ++f) {
                std::array<std::size_t, 4> q_verts;
                for (std::size_t i = 0; i < 4; ++i) {
                    q_verts[i] = blk.vertices[hex_face_quads[f][i]];
                }

                std::size_t cu = 0, cv = 0;
                if (f == 0 || f == 5)      { cu = blk.cells[0]; cv = blk.cells[1]; }
                else if (f == 1 || f == 3) { cu = blk.cells[0]; cv = blk.cells[2]; }
                else                       { cu = blk.cells[1]; cv = blk.cells[2]; }

                auto& ref = bt.faces[f];
                ref.index = find_or_add_face(q_verts, cu, cv);
                const auto& canonical = faces_[ref.index];
                const auto a = static_cast<std::size_t>(std::find(canonical.vertices.begin(), canonical.vertices.end(), q_verts[0]) - canonical.vertices.begin());
                const auto b1 = static_cast<std::size_t>(std::find(canonical.vertices.begin(), canonical.vertices.end(), q_verts[1]) - canonical.vertices.begin());
                const bool a_u = a == 1 || a == 2;
                const bool b_u = b1 == 1 || b1 == 2;
                ref.swap_uv = a_u == b_u;
                ref.reverse_u = a_u;
                ref.reverse_v = a >= 2;
                mpi::require(canonical.cells_u == (ref.swap_uv ? cv : cu) &&
                             canonical.cells_v == (ref.swap_uv ? cu : cv),
                             "Shared face cell counts disagree after axis mapping");
            }

            bt.vol_count = static_cast<std::size_t>(product(static_cast<std::size_t>(product(blk.cells[0] - 1, blk.cells[1] - 1)), blk.cells[2] - 1));
            bt.dist_x = Grading::compute_distribution(blk.cells[0], blk.grading_type[0], blk.grading[0]);
            bt.dist_y = Grading::compute_distribution(blk.cells[1], blk.grading_type[1], blk.grading[1]);
            bt.dist_z = Grading::compute_distribution(blk.cells[2], blk.grading_type[2], blk.grading[2]);
        }

        // Allocate 1-based unique IDs
        std::uint64_t current_id = 1;

        // Strata 0: Macro-vertices [1, num_macro_vertices]
        current_id = add(current_id, num_macro_vertices);

        // Strata 1: Unique Edges
        for (auto& edge : edges_) {
            const std::size_t count = (edge.num_cells > 1) ? (edge.num_cells - 1) : 0;
            if (count > 0) {
                edge.global_start_id = current_id;
                current_id = add(current_id, count);
            }
        }

        // Strata 2: Unique Faces
        for (auto& face : faces_) {
            const std::size_t count = (face.cells_u > 1 && face.cells_v > 1)
                                          ? static_cast<std::size_t>(product(face.cells_u - 1, face.cells_v - 1))
                                          : 0;
            if (count > 0) {
                face.global_start_id = current_id;
                current_id = add(current_id, count);
            }
        }

        // Strata 3: Block Volumes
        for (std::size_t b = 0; b < num_blocks; ++b) {
            if (blocks_[b].vol_count > 0) {
                blocks_[b].vol_start_id = current_id;
                current_id = add(current_id, blocks_[b].vol_count);
            }
        }

        total_nodes_ = current_id - 1;

        // Construct compact coordinate sources; shared edges/faces keep their
        // first owner, which defines canonical node ordering.
        std::vector<NodeRange> edge_sources(edges_.size());
        std::vector<NodeRange> face_sources(faces_.size());
        node_ranges_.reserve(edges_.size() + faces_.size() + num_blocks);

        constexpr std::array<std::array<std::size_t, 3>, 8> corner_ijk = {{
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        }};
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const auto& blk = config.blocks[b];
            const auto& bt = blocks_[b];
            const std::array<std::size_t, 3> n = {blk.cells[0], blk.cells[1], blk.cells[2]};
            for (std::size_t c = 0; c < 8; ++c) {
                vertex_sources_[blk.vertices[c]] = {b, corner_ijk[c][0] * n[0],
                                                      corner_ijk[c][1] * n[1],
                                                      corner_ijk[c][2] * n[2]};
            }
            for (std::size_t e = 0; e < 12; ++e) {
                const auto [idx, reversed] = bt.edges[e];
                if (edges_[idx].num_cells <= 1) continue;
                NodeRange r;
                r.first = edges_[idx].global_start_id;
                r.count = edges_[idx].num_cells - 1;
                r.block_idx = b;
                r.reversed = reversed;
                r.axis = e / 4;
                const auto& corner = corner_ijk[hex_edge_pairs[e].first];
                for (std::size_t d = 0; d < 3; ++d) r.origin[d] = corner[d] * n[d];
                const std::size_t axis = e / 4;
                r.origin[axis] = 1;
                r.shape[axis] = n[axis] - 1;
                if (edge_sources[idx].count == 0) {
                    edge_sources[idx] = r;
                } else {
                    const auto& prior = edge_sources[idx];
                    const auto& old = blocks_[prior.block_idx];
                    const auto old_axis = prior.axis;
                    const auto& old_dist = old_axis == 0 ? old.dist_x : (old_axis == 1 ? old.dist_y : old.dist_z);
                    const auto& new_dist = axis == 0 ? bt.dist_x : (axis == 1 ? bt.dist_y : bt.dist_z);
                    for (std::size_t t = 1; t < edges_[idx].num_cells; ++t) {
                        const double a = prior.reversed ? 1-old_dist[old_dist.size()-1-t] : old_dist[t];
                        const double c = reversed ? 1-new_dist[new_dist.size()-1-t] : new_dist[t];
                        if (std::abs(a-c) > 64 * std::numeric_limits<double>::epsilon())
                            fail("Shared edge has incompatible grading (accounting for reversed orientation)");
                    }
                }
            }
            for (std::size_t f = 0; f < 6; ++f) {
                const auto idx = bt.faces[f].index;
                const auto& face = faces_[idx];
                if (face.cells_u <= 1 || face.cells_v <= 1) continue;
                NodeRange r;
                r.first = face.global_start_id;
                r.count = (face.cells_u - 1) * (face.cells_v - 1);
                r.block_idx = b;
                r.origin = {1, 1, 1};
                r.shape = {n[0] - 1, n[1] - 1, n[2] - 1};
                const std::size_t normal = (f == 0 || f == 5) ? 2 : ((f == 1 || f == 3) ? 1 : 0);
                const bool upper = (f == 2 || f == 3 || f == 5);
                r.origin[normal] = upper ? n[normal] : 0;
                r.shape[normal] = 1;
                // The first block defines canonical face axes and ID ordering.
                if (face_sources[idx].count == 0) face_sources[idx] = r;
            }
            if (bt.vol_count > 0) {
                NodeRange r;
                r.first = bt.vol_start_id;
                r.count = bt.vol_count;
                r.block_idx = b;
                r.origin = {1, 1, 1};
                r.shape = {n[0] - 1, n[1] - 1, n[2] - 1};
                node_ranges_.push_back(r);
            }
        }
        for (const auto& r : edge_sources) if (r.count > 0) node_ranges_.push_back(r);
        for (const auto& r : face_sources) if (r.count > 0) node_ranges_.push_back(r);
        std::sort(node_ranges_.begin(), node_ranges_.end(),
                  [](const NodeRange& a, const NodeRange& b) { return a.first < b.first; });
        std::set<std::array<std::size_t, 4>> assigned;
        for (const auto& patch : config.boundaries) {
            for (const auto& mf : patch.faces) {
                auto key = mf.vertices;
                std::sort(key.begin(), key.end());
                bool found = false;
                for (const auto& face : faces_) {
                    auto candidate = face.vertices;
                    std::sort(candidate.begin(), candidate.end());
                    if (candidate != key) continue;
                    mpi::require(face.incident_blocks == 1, "A boundary patch references an internal block face");
                    mpi::require(assigned.insert(key).second, "A boundary face is assigned more than once");
                    bool cyclic = false;
                    for (std::size_t shift = 0; shift < 4; ++shift) {
                        bool forward = true, backward = true;
                        for (std::size_t v = 0; v < 4; ++v) {
                            forward &= mf.vertices[v] == face.vertices[(shift+v)%4];
                            backward &= mf.vertices[v] == face.vertices[(shift+4-v)%4];
                        }
                        cyclic |= forward || backward;
                    }
                    mpi::require(cyclic, "Boundary vertices must follow the face perimeter");
                    found = true;
                    break;
                }
                if (!found) fail("patch '" + patch.name + "' references a face absent from the blocks");
            }
        }

        // Preserve explicit assignments and collect only unassigned exterior faces.
        boundary_patches_ = config.boundaries;
        BoundaryPatch automatic;
        std::uint64_t automatic_quads = 0;
        for (const auto& face : faces_) {
            if (face.incident_blocks != 1) continue;
            auto key = face.vertices;
            std::sort(key.begin(), key.end());
            if (assigned.contains(key)) continue;
            automatic.faces.push_back({face.vertices});
            automatic_quads = add(automatic_quads, product(face.cells_u, face.cells_v));
        } 
        if (!automatic.faces.empty()) {
            automatic.name = "patch_automatic";
            const auto name_in_use = [&](std::string_view name) {
                return std::any_of(boundary_patches_.begin(), boundary_patches_.end(),
                                   [&](const BoundaryPatch& patch) { return patch.name == name; });
            };
            for (std::size_t suffix = 1; name_in_use(automatic.name); ++suffix) {
                automatic.name = "patch_automatic_" + std::to_string(suffix);
                mpi::require(automatic.name.size() <= 32, "Automatic patch name exceeds CGNS name limit");
            }
            boundary_patches_.push_back(std::move(automatic));
            if (comm.rank() == 0) {
                const auto& patch = boundary_patches_.back();
                std::fprintf(stderr,
                             "[KRONA warning] %zu unassigned outer macro face(s) "
                             "(%llu boundary quads) were added to patch '%s'.\n",
                             patch.faces.size(), static_cast<unsigned long long>(automatic_quads),
                             patch.name.c_str());
            }
        }
    });
}

std::size_t MacroTopology::find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells) {
    const std::size_t v_min = std::min(u, v);
    const std::size_t v_max = std::max(u, v);

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (edges_[i].v0 == v_min && edges_[i].v1 == v_max) {
            if (edges_[i].num_cells != cells) {
                fail(std::format("Shared edge ({},{}) cell count mismatch: {} vs {}",
                                       v_min, v_max, edges_[i].num_cells, cells));
            }
            return i;
        }
    }

    edges_.push_back({v_min, v_max, cells, 0});
    return edges_.size() - 1;
}

std::size_t MacroTopology::find_or_add_face(const std::array<std::size_t, 4>& q_verts,
                                            std::size_t cu, std::size_t cv) {
    auto sorted_q = q_verts;
    std::sort(sorted_q.begin(), sorted_q.end());

    for (std::size_t i = 0; i < faces_.size(); ++i) {
        auto existing = faces_[i].vertices;
        std::sort(existing.begin(), existing.end());
        if (existing == sorted_q) {
            mpi::require(++faces_[i].incident_blocks <= 2, "More than two blocks share a face");
            // Matching sets must also describe the same cyclic perimeter.
            const auto& canonical = faces_[i].vertices;
            bool cyclic = false;
            for (std::size_t shift = 0; shift < 4; ++shift) {
                bool forward = true, backward = true;
                for (std::size_t v = 0; v < 4; ++v) {
                    forward &= q_verts[v] == canonical[(shift+v)%4];
                    backward &= q_verts[v] == canonical[(shift+4-v)%4];
                }
                cyclic |= forward || backward;
            }
            mpi::require(cyclic, "Shared faces have different edge connectivity");
            return i;
        }
    }

    faces_.push_back({q_verts, 1, cu, cv, 0});
    return faces_.size() - 1;
}

std::uint64_t MacroTopology::get_node_id(std::size_t b, std::size_t i, std::size_t j, std::size_t k) const noexcept {
    mpi::require(b < blocks_.size(), "Block index is out of bounds");
    const auto& bt = blocks_[b];
    mpi::require(i < bt.dist_x.size() && j < bt.dist_y.size() && k < bt.dist_z.size(),
                 "Block node index is out of bounds");
    const std::size_t Nx = bt.dist_x.size() - 1;
    const std::size_t Ny = bt.dist_y.size() - 1;
    const std::size_t Nz = bt.dist_z.size() - 1;

    const bool on_x0 = (i == 0), on_x1 = (i == Nx);
    const bool on_y0 = (j == 0), on_y1 = (j == Ny);
    const bool on_z0 = (k == 0), on_z1 = (k == Nz);

    const int bnd_count = (on_x0 || on_x1 ? 1 : 0) +
                          (on_y0 || on_y1 ? 1 : 0) +
                          (on_z0 || on_z1 ? 1 : 0);

    // 1. Macro-vertex
    if (bnd_count == 3) {
        const std::size_t corner = (on_z0 ? 0 : 4) +
                                   (on_y0 ? (on_x0 ? 0 : 1) : (on_x0 ? 3 : 2));
        return static_cast<std::uint64_t>(bt.vertices[corner] + 1);
    }

    // 2. Macro-edge
    if (bnd_count == 2) {
        std::size_t e = 0;
        std::size_t step = 0;
        std::size_t max_step = 0;

        if (!on_x0 && !on_x1) {
            step = i; max_step = Nx;
            if (on_y0 && on_z0)      e = 0;
            else if (on_y1 && on_z0) e = 1;
            else if (on_y0 && on_z1) e = 2;
            else                     e = 3;
        } else if (!on_y0 && !on_y1) {
            step = j; max_step = Ny;
            if (on_x0 && on_z0)      e = 4;
            else if (on_x1 && on_z0) e = 5;
            else if (on_x0 && on_z1) e = 6;
            else                     e = 7;
        } else {
            step = k; max_step = Nz;
            if (on_x0 && on_y0)      e = 8;
            else if (on_x1 && on_y0) e = 9;
            else if (on_x1 && on_y1) e = 10;
            else                     e = 11;
        }

        const auto [edge_idx, is_reversed] = bt.edges[e];
        const auto& edge = edges_[edge_idx];
        const std::size_t interior_idx = is_reversed ? (max_step - step - 1) : (step - 1);
        return edge.global_start_id + interior_idx;
    }

    // 3. Macro-face
    if (bnd_count == 1) {
        std::size_t f = 0;
        std::size_t u = 0, v = 0;

        if (on_z0)      { f = 0; u = i; v = j; }
        else if (on_z1) { f = 5; u = i; v = j; }
        else if (on_y0) { f = 1; u = i; v = k; }
        else if (on_y1) { f = 3; u = i; v = k; }
        else if (on_x0) { f = 4; u = j; v = k; }
        else            { f = 2; u = j; v = k; }

        const auto& ref = bt.faces[f];
        const auto& face = faces_[ref.index];
        if (ref.swap_uv) std::swap(u, v);
        if (ref.reverse_u) u = face.cells_u - u;
        if (ref.reverse_v) v = face.cells_v - v;
        return face.global_start_id + (v - 1) * (face.cells_u - 1) + (u - 1);
    }

    // 4. Volume interior
    const std::uint64_t v_idx = static_cast<std::uint64_t>(k - 1) * (Nx - 1) * (Ny - 1) +
                                static_cast<std::uint64_t>(j - 1) * (Nx - 1) +
                                static_cast<std::uint64_t>(i - 1);
    return bt.vol_start_id + v_idx;
}

const MacroTopology::NodeRange& MacroTopology::find_node_range(std::uint64_t id) const noexcept {
    auto it = std::upper_bound(node_ranges_.begin(), node_ranges_.end(), id,
                              [](std::uint64_t value, const NodeRange& r) { return value < r.first; });
    mpi::require(it != node_ranges_.begin(), "Node ID is outside topology ranges");
    --it;
    mpi::require(id - it->first < it->count, "Node ID is outside topology ranges");
    return *it;
}

Vec3 MacroTopology::evaluate_node_coord(std::uint64_t global_node_id,
                                        const GeneratorConfig& config) const noexcept {
    mpi::require(global_node_id > 0 && global_node_id <= total_nodes_, "Node ID is out of bounds");
    NodeSource src;
    if (global_node_id <= vertex_sources_.size()) {
        src = vertex_sources_[global_node_id - 1];
    } else {
        const auto& r = find_node_range(global_node_id);
        std::uint64_t q = global_node_id - r.first;
        if (r.reversed) q = r.count - 1 - q;
        src.block_idx = r.block_idx;
        src.i = r.origin[0] + static_cast<std::size_t>(q % r.shape[0]);
        q /= r.shape[0];
        src.j = r.origin[1] + static_cast<std::size_t>(q % r.shape[1]);
        src.k = r.origin[2] + static_cast<std::size_t>(q / r.shape[1]);
    }
    const auto& bt = blocks_[src.block_idx];
    const auto& blk = config.blocks[src.block_idx];
    std::array<Vec3, 8> corners;
    for (std::size_t c = 0; c < 8; ++c) corners[c] = config.vertices[blk.vertices[c]];
    return TFI::interpolate_hex(corners, bt.dist_x[src.i], bt.dist_y[src.j], bt.dist_z[src.k]);
}

void MacroTopology::evaluate_node_coords(std::uint64_t first_node_id, std::size_t count,
                                         const GeneratorConfig& config,
                                         double* x, double* y, double* z) const noexcept {
    if (count == 0) return;
    mpi::require(x && y && z, "Coordinate output buffers must be non-null");
    mpi::require(first_node_id > 0 && first_node_id <= total_nodes_ &&
                 count <= total_nodes_ - first_node_id + 1, "Coordinate interval is out of bounds");
    std::size_t out = 0;
    while (out < count && first_node_id <= vertex_sources_.size()) {
        const auto p = evaluate_node_coord(first_node_id++, config);
        x[out] = p.x; y[out] = p.y; z[out] = p.z;
        ++out;
    }
    if (out == count) return;
    // One lookup per batch, then walk adjacent descriptors. Decode only the
    // first index of each span; advance i/j/k without integer division.
    const NodeRange* range = &find_node_range(first_node_id);
    while (out < count) {
        const auto& r = *range;
        const auto& bt = blocks_[r.block_idx];
        const auto& blk = config.blocks[r.block_idx];
        std::array<Vec3, 8> corners;
        for (std::size_t c = 0; c < 8; ++c) corners[c] = config.vertices[blk.vertices[c]];
        const std::uint64_t offset = first_node_id - r.first;
        const std::size_t span = static_cast<std::size_t>(
            std::min<std::uint64_t>(count - out, r.count - offset));
        std::uint64_t q = r.reversed ? r.count - 1 - offset : offset;
        std::size_t i = static_cast<std::size_t>(q % r.shape[0]);
        q /= r.shape[0];
        std::size_t j = static_cast<std::size_t>(q % r.shape[1]);
        std::size_t k = static_cast<std::size_t>(q / r.shape[1]);
        for (std::size_t t = 0; t < span; ++t, ++out) {
            const auto p = TFI::interpolate_hex(corners,
                bt.dist_x[r.origin[0] + i], bt.dist_y[r.origin[1] + j], bt.dist_z[r.origin[2] + k]);
            x[out] = p.x; y[out] = p.y; z[out] = p.z;
            if (!r.reversed) {
                if (++i == r.shape[0]) {
                    i = 0;
                    if (++j == r.shape[1]) { j = 0; ++k; }
                }
            } else {
                if (i > 0) --i;
                else {
                    i = r.shape[0] - 1;
                    if (j > 0) --j;
                    else { j = r.shape[1] - 1; if (k > 0) --k; }
                }
            }
        }
        first_node_id += span;
        ++range;
    }
}

void MacroTopology::get_cell_block_and_ijk(std::uint64_t global_cell_id,
                                           std::size_t& block_idx,
                                           std::size_t& i,
                                           std::size_t& j,
                                           std::size_t& k) const noexcept {
    mpi::require(global_cell_id < total_cells_, "Cell ID must be zero-based and within the mesh");
    auto it = std::upper_bound(cell_offsets_.begin(), cell_offsets_.end(), global_cell_id);
    block_idx = static_cast<std::size_t>(std::distance(cell_offsets_.begin(), it) - 1);

    const std::uint64_t local_c = global_cell_id - cell_offsets_[block_idx];
    const auto& bt = blocks_[block_idx];
    const std::size_t Nx = bt.dist_x.size() - 1;
    const std::size_t Ny = bt.dist_y.size() - 1;

    k = static_cast<std::size_t>(local_c / (Nx * Ny));
    j = static_cast<std::size_t>((local_c % (Nx * Ny)) / Nx);
    i = static_cast<std::size_t>(local_c % Nx);
}

std::array<std::uint64_t, 8> MacroTopology::get_block_cell_nodes(std::size_t b,
                                                                 std::size_t i,
                                                                 std::size_t j,
                                                                 std::size_t k) const noexcept {
    mpi::require(b < blocks_.size(), "Block index is out of bounds");
    const auto& bt = blocks_[b];
    mpi::require(i < bt.dist_x.size()-1 && j < bt.dist_y.size()-1 && k < bt.dist_z.size()-1,
                 "Block cell index is out of bounds");
    // Strictly conforms to CGNS/VTK HEXA_8 (bottom CCW seen from top, top CCW seen from top)
    return {
        get_node_id(b, i,     j,     k),
        get_node_id(b, i + 1, j,     k),
        get_node_id(b, i + 1, j + 1, k),
        get_node_id(b, i,     j + 1, k),
        get_node_id(b, i,     j,     k + 1),
        get_node_id(b, i + 1, j,     k + 1),
        get_node_id(b, i + 1, j + 1, k + 1),
        get_node_id(b, i,     j + 1, k + 1)
    };
}

std::array<std::uint64_t, 8> MacroTopology::get_cell_nodes(std::uint64_t global_cell_id) const noexcept {
    std::size_t b = 0, i = 0, j = 0, k = 0;
    get_cell_block_and_ijk(global_cell_id, b, i, j, k);
    return get_block_cell_nodes(b, i, j, k);
}

} // namespace krona::mesh::generation