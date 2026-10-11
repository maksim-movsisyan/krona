#include <krona/mesh/reorder.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

namespace krona::mesh {
namespace {

/// @brief Checks required array sizes once and permits absent metric arrays.
void check_shape(const MeshPart& mesh)
{
    if (mesh.n_own < 0 || mesh.n_cells < mesh.n_own || mesh.n_faces < 0)
        mpi::fatal("Invalid entity counts for mesh reordering");
    const auto cells = static_cast<std::size_t>(mesh.n_cells);
    const auto faces = static_cast<std::size_t>(mesh.n_faces);
    if (mesh.cell_type.size() != cells || mesh.cell_gid.size() != cells ||
        mesh.cell_donor.size() != cells || mesh.cell_nodes_offsets.size() != cells + 1 ||
        mesh.cell_nodes_offsets.front() != 0 || mesh.cell_nodes_offsets.back() != mesh.cell_nodes.size() ||
        mesh.face_owner.size() != faces || mesh.face_neigh.size() != faces ||
        mesh.face_type.size() != faces || mesh.face_patch.size() != faces ||
        mesh.face_local_index.size() != faces || mesh.face_nodes_offsets.size() != faces + 1 ||
        mesh.face_nodes_offsets.front() != 0 || mesh.face_nodes_offsets.back() != mesh.face_nodes.size())
        mpi::fatal("Mesh topology array sizes do not match reordering input counts");
    // Geometry may be computed before or after this stage.
    for (const auto* field : {&mesh.cell_centroid_x, &mesh.cell_centroid_y,
                              &mesh.cell_centroid_z, &mesh.cell_volume})
        if (!field->empty() && field->size() != cells) mpi::fatal("Invalid cell metric array size");
    for (const auto* field : {&mesh.face_centroid_x, &mesh.face_centroid_y, &mesh.face_centroid_z,
                              &mesh.face_normal_x, &mesh.face_normal_y, &mesh.face_normal_z, &mesh.face_area})
        if (!field->empty() && field->size() != faces) mpi::fatal("Invalid face metric array size");
}

struct OwnedGraph {
    std::vector<std::size_t> offsets;
    std::vector<LocalIndex> adjacency;
};

/// @brief Builds symmetric owned-owned adjacency with size_t storage offsets.
[[nodiscard]] OwnedGraph owned_graph(const MeshPart& mesh)
{
    OwnedGraph graph;
    graph.offsets.assign(static_cast<std::size_t>(mesh.n_own) + 1, 0);
    for (std::size_t f = 0; f < mesh.face_owner.size(); ++f) {
        const auto u = mesh.face_owner[f], v = mesh.face_neigh[f];
        assert(u >= 0 && u < mesh.n_own);
        if (v >= 0 && v < mesh.n_own) {
            assert(u != v);
            ++graph.offsets[static_cast<std::size_t>(u) + 1];
            ++graph.offsets[static_cast<std::size_t>(v) + 1];
        }
    }
    std::partial_sum(graph.offsets.begin(), graph.offsets.end(), graph.offsets.begin());
    graph.adjacency.resize(graph.offsets.back());
    auto cursor = graph.offsets;
    for (std::size_t f = 0; f < mesh.face_owner.size(); ++f) {
        const auto u = mesh.face_owner[f], v = mesh.face_neigh[f];
        if (v >= 0 && v < mesh.n_own) {
            graph.adjacency[cursor[static_cast<std::size_t>(u)]++] = v;
            graph.adjacency[cursor[static_cast<std::size_t>(v)]++] = u;
        }
    }
    return graph;
}

/// @brief Orders vertices by degree and uses the current local index to break ties.
[[nodiscard]] bool degree_less(const OwnedGraph& graph, LocalIndex a, LocalIndex b) noexcept
{
    const auto ai = static_cast<std::size_t>(a), bi = static_cast<std::size_t>(b);
    const auto da = graph.offsets[ai + 1] - graph.offsets[ai];
    const auto db = graph.offsets[bi + 1] - graph.offsets[bi];
    return da != db ? da < db : a < b;
}

/// @brief Chooses a component start with at most three pseudo-peripheral BFS passes.
[[nodiscard]] LocalIndex peripheral_start(const OwnedGraph& graph, LocalIndex root,
                                         std::vector<std::uint8_t>& state,
                                         std::vector<LocalIndex>& queue)
{
    std::size_t previous_depth = 0;
    for (int pass = 0; pass < 3; ++pass) {
        queue.clear(); queue.push_back(root);
        state[static_cast<std::size_t>(root)] = 1;
        std::size_t begin = 0, depth = 0;
        LocalIndex candidate = root;
        while (begin < queue.size()) {
            const auto end = queue.size();
            candidate = queue[begin];
            for (auto i = begin; i < end; ++i) {
                const auto u = queue[i];
                if (degree_less(graph, u, candidate)) candidate = u;
                const auto ui = static_cast<std::size_t>(u);
                for (auto k = graph.offsets[ui]; k < graph.offsets[ui + 1]; ++k) {
                    const auto v = graph.adjacency[k];
                    if (state[static_cast<std::size_t>(v)] == 0) {
                        state[static_cast<std::size_t>(v)] = 1;
                        queue.push_back(v);
                    }
                }
            }
            begin = end;
            if (begin < queue.size()) ++depth;
        }
        // Clear only this component's temporary marks; completed components stay marked 2.
        for (auto v : queue) state[static_cast<std::size_t>(v)] = 0;
        if (depth <= previous_depth) break;
        previous_depth = depth; root = candidate;
    }
    return root;
}

/// @brief Computes componentwise RCM using flat queues and reusable visitation storage.
[[nodiscard]] std::vector<LocalIndex> rcm_order(const MeshPart& mesh)
{
    const auto graph = owned_graph(mesh);
    const auto count = static_cast<std::size_t>(mesh.n_own);
    std::vector<std::uint8_t> state(count, 0);
    std::vector<LocalIndex> order, queue;
    order.reserve(count);
    for (LocalIndex seed = 0; seed < mesh.n_own; ++seed) {
        if (state[static_cast<std::size_t>(seed)] != 0) continue;
        const auto root = peripheral_start(graph, seed, state, queue);
        const auto component_begin = order.size();
        order.push_back(root); state[static_cast<std::size_t>(root)] = 2;
        // The result buffer also serves as the Cuthill-McKee BFS queue.
        for (auto head = component_begin; head < order.size(); ++head) {
            const auto u = static_cast<std::size_t>(order[head]);
            const auto neighbours_begin = order.size();
            for (auto k = graph.offsets[u]; k < graph.offsets[u + 1]; ++k) {
                const auto v = graph.adjacency[k];
                if (state[static_cast<std::size_t>(v)] == 0) {
                    state[static_cast<std::size_t>(v)] = 2;
                    order.push_back(v);
                }
            }
            std::sort(order.begin() + static_cast<std::ptrdiff_t>(neighbours_begin), order.end(),
                [&](LocalIndex a, LocalIndex b) { return degree_less(graph, a, b); });
        }
        std::reverse(order.begin() + static_cast<std::ptrdiff_t>(component_begin), order.end());
    }
    return order;
}

/// @brief Maps 21-bit coordinates to a 63-bit 3D Hilbert key using Skilling's transform.
[[nodiscard]] std::uint64_t hilbert_key(std::array<std::uint32_t, 3> axis) noexcept
{
    constexpr std::uint32_t high_bit = 1u << 20;
    // Undo axis exchanges, then encode the Hilbert transpose in Gray form.
    for (auto q = high_bit; q > 1; q >>= 1) {
        const auto mask = q - 1;
        for (std::size_t i = 0; i < 3; ++i) {
            if ((axis[i] & q) != 0) axis[0] ^= mask;
            else {
                const auto exchange = (axis[0] ^ axis[i]) & mask;
                axis[0] ^= exchange; axis[i] ^= exchange;
            }
        }
    }
    axis[1] ^= axis[0]; axis[2] ^= axis[1];
    std::uint32_t correction = 0;
    for (auto q = high_bit; q > 1; q >>= 1)
        if ((axis[2] & q) != 0) correction ^= q - 1;
    for (auto& value : axis) value ^= correction;
    std::uint64_t key = 0;
    // Interleave the transformed axes from the most significant bit.
    for (int bit = 20; bit >= 0; --bit)
        for (auto value : axis) key = (key << 1) | ((value >> bit) & 1u);
    return key;
}

/// @brief Sorts owned centroids along a uniformly scaled local Hilbert curve.
[[nodiscard]] std::vector<LocalIndex> hilbert_order(const MeshPart& mesh)
{
    const auto count = static_cast<std::size_t>(mesh.n_own);
    if (mesh.cell_centroid_x.empty() || mesh.cell_centroid_y.empty() || mesh.cell_centroid_z.empty())
        mpi::fatal("Hilbert ordering requires computed cell centroids");
    const std::array<const std::vector<GeometryReal>*, 3> coordinates{
        &mesh.cell_centroid_x, &mesh.cell_centroid_y, &mesh.cell_centroid_z};
    std::array<double, 3> lo, hi;
    lo.fill(std::numeric_limits<double>::infinity()); hi.fill(-std::numeric_limits<double>::infinity());
    for (std::size_t c = 0; c < count; ++c)
        for (std::size_t d = 0; d < 3; ++d) {
            const double value = static_cast<double>((*coordinates[d])[c]);
            if (!std::isfinite(value)) mpi::fatal("Hilbert ordering requires finite cell centroids");
            lo[d] = std::min(lo[d], value); hi[d] = std::max(hi[d], value);
        }
    const double extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    if (!std::isfinite(extent)) mpi::fatal("Hilbert centroid bounding extent overflow");
    constexpr std::uint32_t maximum = (1u << 21) - 1;
    // Divide by the extent before multiplying: tiny domains need no absolute cutoff.
    const auto quantize = [&](double value, std::size_t d) {
        const double scaled = extent > 0 ? ((value - lo[d]) / extent) * maximum : 0;
        return static_cast<std::uint32_t>(std::clamp(scaled, 0.0, static_cast<double>(maximum)));
    };
    std::vector<std::uint64_t> keys(count);
    std::vector<LocalIndex> order(count);
    for (std::size_t c = 0; c < count; ++c) {
        keys[c] = hilbert_key({quantize((*coordinates[0])[c], 0), quantize((*coordinates[1])[c], 1),
                              quantize((*coordinates[2])[c], 2)});
        order[c] = static_cast<LocalIndex>(c);
    }
    // GIDs make quantization ties independent of the preceding local ordering.
    std::sort(order.begin(), order.end(), [&](LocalIndex a, LocalIndex b) {
        const auto ai = static_cast<std::size_t>(a), bi = static_cast<std::size_t>(b);
        return std::tie(keys[ai], mesh.cell_gid[ai]) < std::tie(keys[bi], mesh.cell_gid[bi]);
    });
    return order;
}

/// @brief Reorders a vector prefix by cycles with one shared byte-per-row work buffer.
template<class T>
void permute(std::vector<T>& values, std::span<const LocalIndex> order,
             std::vector<std::uint8_t>& visited)
{
    if (values.empty()) return;
    visited.assign(order.size(), 0);
    for (std::size_t start = 0; start < order.size(); ++start) {
        if (visited[start] != 0 || order[start] == static_cast<LocalIndex>(start)) continue;
        const T saved = values[start];
        auto current = start;
        // Gather each cycle without allocating a second field or touching the ghost tail.
        while (static_cast<std::size_t>(order[current]) != start) {
            const auto old = static_cast<std::size_t>(order[current]);
            values[current] = values[old]; visited[current] = 1; current = old;
        }
        values[current] = saved; visited[current] = 1;
    }
}

/// @brief Rebuilds permuted CSR rows while keeping any unpermuted tail in place.
void permute_connectivity(std::vector<std::size_t>& offsets, std::vector<LocalIndex>& nodes,
                          std::span<const LocalIndex> order)
{
    std::vector<std::size_t> reordered_offsets(order.size() + 1, 0);
    for (std::size_t row = 0; row < order.size(); ++row) {
        const auto old = static_cast<std::size_t>(order[row]);
        reordered_offsets[row + 1] = reordered_offsets[row] + offsets[old + 1] - offsets[old];
    }
    std::vector<LocalIndex> reordered_nodes(reordered_offsets.back());
    for (std::size_t row = 0; row < order.size(); ++row) {
        const auto old = static_cast<std::size_t>(order[row]);
        std::copy_n(nodes.data() + offsets[old], offsets[old + 1] - offsets[old],
                    reordered_nodes.data() + reordered_offsets[row]);
    }
    if (order.size() + 1 == offsets.size()) {
        offsets.swap(reordered_offsets); nodes.swap(reordered_nodes);
    } else {
        // The owned total length is invariant, so every ghost row keeps its original offset.
        std::copy(reordered_offsets.begin(), reordered_offsets.end(), offsets.begin());
        std::copy(reordered_nodes.begin(), reordered_nodes.end(), nodes.begin());
    }
}

/// @brief Applies an owned-cell permutation and remaps face and halo-send cell references.
void reorder_cells(MeshPart& mesh, std::span<const LocalIndex> order)
{
    const auto owned = static_cast<std::size_t>(mesh.n_own);
    if (order.size() != owned) mpi::fatal("Owned-cell permutation must contain n_own entries");
    std::vector<LocalIndex> inverse(owned, invalid_local_index);
    bool changed = false;
    for (std::size_t cell = 0; cell < owned; ++cell) {
        const auto old = order[cell];
        if (old < 0 || old >= mesh.n_own || inverse[static_cast<std::size_t>(old)] != invalid_local_index)
            mpi::fatal("Owned-cell permutation is not a bijection");
        inverse[static_cast<std::size_t>(old)] = static_cast<LocalIndex>(cell);
        changed = changed || old != static_cast<LocalIndex>(cell);
    }
    if (!changed) return;
    // Only owned indices need an inverse map; ghost indices already have the right numbering.
    for (auto& cell : mesh.send_owned_local) {
        assert(cell >= 0 && cell < mesh.n_own);
        cell = inverse[static_cast<std::size_t>(cell)];
    }
    for (std::size_t f = 0; f < mesh.face_owner.size(); ++f) {
        assert(mesh.face_owner[f] >= 0 && mesh.face_owner[f] < mesh.n_own);
        mesh.face_owner[f] = inverse[static_cast<std::size_t>(mesh.face_owner[f])];
        if (mesh.face_neigh[f] >= 0 && mesh.face_neigh[f] < mesh.n_own)
            mesh.face_neigh[f] = inverse[static_cast<std::size_t>(mesh.face_neigh[f])];
    }
    std::vector<LocalIndex>{}.swap(inverse);
    permute_connectivity(mesh.cell_nodes_offsets, mesh.cell_nodes, order);
    std::vector<std::uint8_t> work;
    permute(mesh.cell_type, order, work); permute(mesh.cell_gid, order, work);
    permute(mesh.cell_donor, order, work);
    permute(mesh.cell_centroid_x, order, work); permute(mesh.cell_centroid_y, order, work);
    permute(mesh.cell_centroid_z, order, work); permute(mesh.cell_volume, order, work);
}

/// @brief Separates and sorts faces into owned-owned, owned-ghost and boundary ranges.
[[nodiscard]] std::vector<LocalIndex> face_order(MeshPart& mesh)
{
    std::array<std::size_t, 3> counts{};
    const auto category = [&](std::size_t f) {
        const auto neighbour = mesh.face_neigh[f];
        return neighbour == invalid_local_index ? 2u : neighbour < mesh.n_own ? 0u : 1u;
    };
    for (std::size_t f = 0; f < mesh.face_owner.size(); ++f) {
        const auto patch = mesh.face_patch[f];
        if (patch != invalid_patch_id &&
            (category(f) != 2 || patch < 0 || static_cast<std::size_t>(patch) >= mesh.patches.size()))
            mpi::fatal("Invalid boundary patch assignment during face grouping");
        ++counts[category(f)];
    }
    const auto internal = counts[0], inner = counts[0] + counts[1];
    std::array<std::size_t, 3> cursor{0, internal, inner};
    std::vector<LocalIndex> order(mesh.face_owner.size());
    for (std::size_t f = 0; f < order.size(); ++f)
        order[cursor[category(f)]++] = static_cast<LocalIndex>(f);
    const auto internal_less = [&](LocalIndex a, LocalIndex b) {
        const auto ai = static_cast<std::size_t>(a), bi = static_cast<std::size_t>(b);
        return std::tie(mesh.face_owner[ai], mesh.face_neigh[ai], a) <
               std::tie(mesh.face_owner[bi], mesh.face_neigh[bi], b);
    };
    const auto boundary_less = [&](LocalIndex a, LocalIndex b) {
        const auto ai = static_cast<std::size_t>(a), bi = static_cast<std::size_t>(b);
        const auto ap = mesh.face_patch[ai] == invalid_patch_id ? std::numeric_limits<PatchId>::max() : mesh.face_patch[ai];
        const auto bp = mesh.face_patch[bi] == invalid_patch_id ? std::numeric_limits<PatchId>::max() : mesh.face_patch[bi];
        return std::tie(ap, mesh.face_owner[ai], a) < std::tie(bp, mesh.face_owner[bi], b);
    };
    // Category branches stay outside the sort comparators; unlabelled boundaries go last.
    auto middle = order.begin() + static_cast<std::ptrdiff_t>(internal);
    auto boundary = order.begin() + static_cast<std::ptrdiff_t>(inner);
    std::sort(order.begin(), middle, internal_less);
    std::sort(middle, boundary, internal_less);
    std::sort(boundary, order.end(), boundary_less);
    mesh.n_internal_faces = static_cast<LocalIndex>(internal);
    mesh.n_inner_faces = static_cast<LocalIndex>(inner);
    return order;
}

/// @brief Rebuilds patch-face CSR directly from the already patch-sorted boundary range.
void rebuild_patches(MeshPart& mesh)
{
    mesh.patch_face_offsets.assign(mesh.patches.size() + 1, 0);
    for (auto f = static_cast<std::size_t>(mesh.n_inner_faces); f < mesh.face_patch.size(); ++f) {
        const auto patch = mesh.face_patch[f];
        if (patch != invalid_patch_id) ++mesh.patch_face_offsets[static_cast<std::size_t>(patch) + 1];
    }
    std::partial_sum(mesh.patch_face_offsets.begin(), mesh.patch_face_offsets.end(), mesh.patch_face_offsets.begin());
    mesh.patch_faces.clear(); mesh.patch_faces.reserve(mesh.patch_face_offsets.back());
    // Face order already matches patch order; no patch-count/cursor scratch arrays are needed.
    for (auto f = static_cast<std::size_t>(mesh.n_inner_faces); f < mesh.face_patch.size(); ++f)
        if (mesh.face_patch[f] != invalid_patch_id) mesh.patch_faces.push_back(static_cast<LocalIndex>(f));
}

/// @brief Moves all face metadata, metrics and oriented loops into their three sorted ranges.
void reorder_faces(MeshPart& mesh)
{
    const auto order = face_order(mesh);
    bool changed = false;
    for (std::size_t f = 0; f < order.size(); ++f) changed = changed || order[f] != static_cast<LocalIndex>(f);
    if (changed) {
        permute_connectivity(mesh.face_nodes_offsets, mesh.face_nodes, order);
        std::vector<std::uint8_t> work;
        permute(mesh.face_owner, order, work); permute(mesh.face_neigh, order, work);
        permute(mesh.face_type, order, work); permute(mesh.face_patch, order, work);
        permute(mesh.face_local_index, order, work);
        permute(mesh.face_centroid_x, order, work); permute(mesh.face_centroid_y, order, work);
        permute(mesh.face_centroid_z, order, work); permute(mesh.face_area, order, work);
        permute(mesh.face_normal_x, order, work); permute(mesh.face_normal_y, order, work);
        permute(mesh.face_normal_z, order, work);
    }
    rebuild_patches(mesh);
    mesh.faces_grouped = true;
}

} // namespace

/// @brief Selects a built-in owned-cell order and always completes local face grouping.
void reorder_local_mesh(MeshPart& mesh, ReorderMethod method)
{
    if (mpi::is_active()) mpi::require_main_thread();

    mpi::guarded([&]() {
        check_shape(mesh);
        if (method != ReorderMethod::NONE && method != ReorderMethod::RCM && method != ReorderMethod::HILBERT_SFC)
            mpi::fatal("Unknown mesh reordering method");
        mesh.faces_grouped = false;
        if (mesh.n_own > 1 && method != ReorderMethod::NONE) {
            const auto order = method == ReorderMethod::RCM ? rcm_order(mesh) : hilbert_order(mesh);
            reorder_cells(mesh, order);
        }
        reorder_faces(mesh);
    });
}

/// @brief Applies a custom owned permutation through the same topology/metric update path.
void reorder_local_mesh(MeshPart& mesh, std::span<const LocalIndex> owned_new_to_old)
{
    if (mpi::is_active()) mpi::require_main_thread();

    mpi::guarded([&](){
        check_shape(mesh);
        mesh.faces_grouped = false;
        reorder_cells(mesh, owned_new_to_old);
        reorder_faces(mesh);
    });
}

} // namespace krona::mesh
