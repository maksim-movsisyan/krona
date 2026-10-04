#include <krona/mesh/faces.hpp>

#include <krona/mpi/collectives.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace krona::mesh {
namespace {

/// @brief A cell half-face or an optional exterior-face label.
struct HalfFace {
    FaceKey key;
    GlobalIndex cell = invalid_cell_id; // Sentinel distinguishes boundary labels.
    PatchId patch = invalid_patch_id;
    std::uint8_t local_face = invalid_local_face;
};

/// @brief Reverse adjacency delivered to the owner of the larger cell ID.
struct DualEdge {
    GlobalIndex cell;
    GlobalIndex neighbour;
};

/// @brief Releases large temporary storage as soon as its stage is complete.
template<class T>
void release(std::vector<T>& data)
{
    // clear() would retain capacity and increase the next stage's peak memory.
    std::vector<T>{}.swap(data);
}

/// @brief Hashes the complete face key and avalanches regularly numbered nodes.
[[nodiscard]] std::uint64_t face_hash(const FaceKey& key) noexcept
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const auto node : key.nodes) {
        hash ^= static_cast<std::uint64_t>(node);
        hash *= UINT64_C(1099511628211);
    }
    // Mix the low bits before reducing to a rank and round.
    hash ^= hash >> 33;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash *= UINT64_C(0xc4ceb9fe1a85ec53);
    return hash ^ (hash >> 33);
}

/// @brief Builds a matching key from an oriented cell face.
[[nodiscard]] FaceKey cell_face_key(const RawMesh& mesh, std::size_t cell,
                                    std::size_t face) noexcept
{
    const auto type = static_cast<std::size_t>(mesh.cells.types[cell]);
    const auto nodes = mesh.cells.nodes(cell);
    const auto count = static_cast<std::size_t>(face_nodes[type][face]);
    std::array<GlobalIndex, 4> face_ids{};
    // The local face index retains orientation; only the matching key is sorted.
    for (std::size_t i = 0; i < count; ++i)
        face_ids[i] = nodes[static_cast<std::size_t>(face_table[type][face][i])];
    return make_face_key(std::span<const GlobalIndex>{face_ids}.first(count));
}

/// @brief Visits source keys with compact references instead of materializing records.
template<class Visit>
void visit_sources(const RawMesh& mesh, Visit&& visit)
{
    // Eight slots per cell encode all supported local face indices.
    for (std::size_t cell = 0; cell < mesh.cells.size(); ++cell) {
        const std::size_t type = static_cast<std::size_t>(mesh.cells.types[cell]);
        for (int face = 0; face < faces_per_type[type]; ++face)
            visit(cell_face_key(mesh, cell, static_cast<std::size_t>(face)),
                  cell * 8 + static_cast<std::size_t>(face));
    }
    const std::size_t boundary_begin = mesh.cells.size() * 8;
    for (std::size_t i = 0; i < mesh.boundary.size(); ++i)
        visit(make_face_key(mesh.boundary.elements.nodes(i)), boundary_begin + i);
}

/// @brief Expands one compact source reference when its communication round is due.
[[nodiscard]] HalfFace source_record(const RawMesh& mesh, std::size_t source)
{
    const auto boundary_begin = mesh.cells.size() * 8;
    if (source >= boundary_begin) {
        const auto i = source - boundary_begin;
        // Boundary IDs and coordinate ownership are irrelevant for face matching.
        return {make_face_key(mesh.boundary.elements.nodes(i)), invalid_cell_id,
                mesh.boundary.patch_ids[i], invalid_local_face};
    }
    const auto cell = source / 8, face = source % 8;
    const auto id = mesh.distribution.cells.begin(mesh.distribution.rank) +
                    static_cast<GlobalIndex>(cell);
    return {cell_face_key(mesh, cell, face), id, invalid_patch_id,
            static_cast<std::uint8_t>(face)};
}

/// @brief Builds packing cursors from record counts for each destination.
[[nodiscard]] std::vector<std::size_t> packing_offsets(std::span<const int> counts)
{
    std::vector<std::size_t> offsets(counts.size() + 1, 0);
    // MPI layouts are built later by the reusable collective workspace.
    for (std::size_t i = 0; i < counts.size(); ++i)
        offsets[i + 1] = offsets[i] + static_cast<std::size_t>(counts[i]);
    return offsets;
}

/// @brief Matches complete equal-key groups and rejects ambiguous topology or labels.
template<class Visit>
void visit_matches(std::span<const HalfFace> records,
                   const distribution::BlockDistribution<GlobalIndex>& ownership,
                   Visit&& visit)
{
    for (std::size_t begin = 0; begin < records.size();) {
        std::size_t end = begin + 1;
        while (end < records.size() && records[begin].key == records[end].key) ++end;

        const HalfFace* cells[2]{};
        const HalfFace* boundary = nullptr;
        int count = 0;
        for (std::size_t i = begin; i < end; ++i) {
            const auto& record = records[i];
            if (record.cell == invalid_cell_id) {
                if (boundary) mpi::fatal("Duplicate boundary face records");
                boundary = &record;
            } else {
                if (count == 2) mpi::fatal("Non-manifold face shared by more than two cells");
                cells[count++] = &record;
            }
        }
        if (count == 0) mpi::fatal("Boundary element does not match any cell face");
        if (count == 2) {
            if (cells[0]->cell == cells[1]->cell)
                mpi::fatal("A cell contains duplicate faces");
            if (boundary) mpi::fatal("Boundary element matches an interior face");
            // Unique face ownership follows the smaller global cell ID.
            if (cells[0]->cell > cells[1]->cell) std::swap(cells[0], cells[1]);
        }
        const auto& a = *cells[0];
        const auto b = count == 2 ? cells[1]->cell : invalid_cell_id;
        const FaceRecord face{a.key, a.cell, b, boundary ? boundary->patch : invalid_patch_id,
                              a.local_face, count == 2 ? cells[1]->local_face : invalid_local_face};
        visit(face, ownership.owner(a.cell), count == 2 ? ownership.owner(b) : -1);
        begin = end;
    }
}

/// @brief Fills bounded CSR rows directly while communication rounds complete.
class GraphBuilder {
public:
    /// @brief Reserves at most one neighbour slot per local cell face.
    GraphBuilder(DualGraph& graph, const RawMesh& mesh)
        : graph_(graph), first_(mesh.distribution.cells.begin(mesh.distribution.rank)),
          degrees_(mesh.cells.size(), 0)
    {
        graph_.offsets.resize(mesh.cells.size() + 1, 0);
        // A conforming cell cannot have more neighbours than faces.
        for (std::size_t cell = 0; cell < mesh.cells.size(); ++cell) {
            const auto type = static_cast<std::size_t>(mesh.cells.types[cell]);
            graph_.offsets[cell + 1] = graph_.offsets[cell] +
                                       static_cast<std::size_t>(faces_per_type[type]);
        }
        graph_.adjacency.resize(graph_.offsets.back());
    }

    /// @brief Adds one directed adjacency to an owned row without reallocating.
    void add(GlobalIndex cell, GlobalIndex neighbour)
    {
        const auto row = static_cast<std::size_t>(cell - first_);
        assert(row < degrees_.size());
        assert(degrees_[row] < graph_.offsets[row + 1] - graph_.offsets[row]);
        // Received reverse edges go straight into their final row's reserved slots.
        graph_.adjacency[graph_.offsets[row] + degrees_[row]++] = neighbour;
    }

    /// @brief Sorts neighbours, removes duplicate adjacencies and compacts rows in place.
    void finish()
    {
        std::size_t written = 0;
        for (std::size_t row = 0; row < degrees_.size(); ++row) {
            const auto start = graph_.offsets[row];
            auto first = graph_.adjacency.begin() + static_cast<std::ptrdiff_t>(start);
            auto last = first + degrees_[row];
            std::sort(first, last);
            last = std::unique(first, last);
            graph_.offsets[row] = written;
            // The destination never passes the source, so forward copying is safe.
            for (auto at = first; at != last; ++at) graph_.adjacency[written++] = *at;
        }
        graph_.offsets.back() = written;
        graph_.adjacency.resize(written); // Keep the small face-count upper-bound capacity.
        if (written == 0) release(graph_.adjacency);
        release(degrees_);
    }

private:
    DualGraph& graph_;
    GlobalIndex first_;
    std::vector<std::uint8_t> degrees_;
};

/// @brief Validates stage contracts once and counts local half-faces and labels.
[[nodiscard]] std::uint64_t count_sources(const RawMesh& mesh, mpi::CommView comm)
{
    // get cell distribution
    const auto& ownership = mesh.distribution.cells;

    // corectness check
    if (mesh.cell_dimension != 3 || mesh.distribution.rank != comm.rank() ||
        ownership.parts() != comm.size() || !std::in_range<std::size_t>(ownership.local_size(comm.rank())) ||
        mesh.cells.size() != static_cast<std::size_t>(ownership.local_size(comm.rank())))
        mpi::fatal("Face construction requires a 3D mesh in its current block ownership");
    if (!std::in_range<LocalIndex>(mesh.cells.size()))
        mpi::fatal("Local cell count exceeds LocalIndex");
    if (mesh.cells.size() > (std::numeric_limits<std::size_t>::max() - mesh.boundary.size()) / 8)
        mpi::fatal("Face source references exceed size_t");

    assert(mesh.cells.offsets.size() == mesh.cells.size() + 1);
    assert(mesh.boundary.elements.offsets.size() == mesh.boundary.size() + 1);
    assert(mesh.boundary.patch_ids.size() == mesh.boundary.size());

    // get numberof  local boundary faces
    std::uint64_t count = mesh.boundary.size();

    // type checks protect the fixed face tables; connectivity validity is a caller contract.
    for (std::size_t cell = 0; cell < mesh.cells.size(); ++cell) {
        const std::size_t type = static_cast<std::size_t>(mesh.cells.types[cell]);
        if (type > static_cast<std::size_t>(CellType::HEXA))
            mpi::fatal("Unsupported volume cell type in face construction");
        assert(mesh.cells.nodes(cell).size() == static_cast<std::size_t>(nodes_per_type[type]));
        count += static_cast<std::uint64_t>(faces_per_type[type]);
    }

    // 
    for (std::size_t i = 0; i < mesh.boundary.size(); ++i) {
        const auto type = mesh.boundary.elements.types[i];
        if (type != CellType::TRI && type != CellType::QUAD)
            mpi::fatal("Face construction requires TRI or QUAD boundary elements");
        assert(mesh.boundary.elements.nodes(i).size() == static_cast<std::size_t>(nodes_per_element(type)));
    }
    return count;
}

/// @brief Runs key-based rendezvous batches and delivers faces to their cell owners.
[[nodiscard]] BuildFacesResult build(const RawMesh& mesh, mpi::CommView comm,
                                     const BuildFacesOptions& options)
{
    // get total number of local faces (boundary faces are counted twice: as boundary element and as face of boundary cell)
    const std::uint64_t local_sources = count_sources(mesh, comm);
    
    // get number of ranks
    const std::size_t peers = static_cast<std::size_t>(comm.size());
    if (options.batch_bytes < sizeof(HalfFace)) mpi::fatal("Face batch size is too small");

    // all ranks need the same round count; this is algorithm planning, not status checking.
    const auto maximum = mpi::allreduce<mpi::Reduction::max>(comm, local_sources);
    
    const auto target = std::min(options.batch_bytes / sizeof(HalfFace),
        static_cast<std::size_t>(std::numeric_limits<int>::max()) / peers);
    const auto rounds = std::max<std::uint64_t>(1, maximum / target + (maximum % target != 0));
    if (rounds > (std::numeric_limits<std::size_t>::max() - 1) / peers) 
        mpi::fatal("Face bucket count exceeds size_t");
    
    const std::size_t buckets = static_cast<std::size_t>(rounds) * peers;

    // the same key always selects the same round and rendezvous rank.
    std::vector<std::size_t> offsets(buckets + 1, 0);
    visit_sources(mesh, [&](const FaceKey& key, std::size_t) {
        ++offsets[static_cast<std::size_t>(face_hash(key) % buckets) + 1];
    });
    for (std::size_t i = 0; i < buckets; ++i) offsets[i + 1] += offsets[i];
    std::vector<std::size_t> sources(offsets.back());
    {
        auto cursor = offsets;
        visit_sources(mesh, [&](const FaceKey& key, std::size_t source) {
            sources[cursor[static_cast<std::size_t>(face_hash(key) % buckets)]++] = source;
        });
    }

    BuildFacesResult result;
    GraphBuilder graph(result.graph, mesh);
    mpi::AlltoallvWorkspace workspace;
    const mpi::RecordDatatype<HalfFace> half_type;
    const mpi::RecordDatatype<FaceRecord> face_type;
    const mpi::RecordDatatype<DualEdge> edge_type;
    std::vector<std::vector<FaceRecord>> chunks;
    chunks.reserve(static_cast<std::size_t>(rounds));
    std::vector<int> counts(peers), face_counts(peers), edge_counts(peers);
    std::size_t total_faces = 0;

    for (std::size_t round = 0; round < static_cast<std::size_t>(rounds); ++round) {
        const auto bucket = round * peers;
        for (std::size_t peer = 0; peer < peers; ++peer) {
            const auto count = offsets[bucket + peer + 1] - offsets[bucket + peer];
            if (!std::in_range<int>(count)) mpi::fatal("Face bucket exceeds MPI INT_MAX; reduce batch_bytes");
            counts[peer] = static_cast<int>(count);
        }
        std::vector<HalfFace> send(offsets[bucket + peers] - offsets[bucket]);
        for (std::size_t i = 0; i < send.size(); ++i)
            send[i] = source_record(mesh, sources[offsets[bucket] + i]);
        std::vector<HalfFace> received;
        mpi::alltoallv_packed(comm, std::span<const int>{counts}, std::span<const HalfFace>{send},
                             received, workspace, half_type);
        release(send);
        if (!std::in_range<int>(received.size()))
            mpi::fatal("Face receive batch exceeds MPI INT_MAX; reduce batch_bytes");
        std::sort(received.begin(), received.end(), [](const HalfFace& a, const HalfFace& b) {
            return a.key < b.key;
        });

        std::fill(face_counts.begin(), face_counts.end(), 0);
        std::fill(edge_counts.begin(), edge_counts.end(), 0);
        visit_matches(received, mesh.distribution.cells, [&](const FaceRecord&, int a, int b) {
            ++face_counts[static_cast<std::size_t>(a)];
            if (b != -1 && a != b) ++edge_counts[static_cast<std::size_t>(b)];
        });
        auto face_cursor = packing_offsets(face_counts);
        auto edge_cursor = packing_offsets(edge_counts);
        std::vector<FaceRecord> face_send(face_cursor.back());
        std::vector<DualEdge> edge_send(edge_cursor.back());
        visit_matches(received, mesh.distribution.cells, [&](const FaceRecord& face, int a, int b) {
            face_send[face_cursor[static_cast<std::size_t>(a)]++] = face;
            if (b != -1 && a != b)
                edge_send[edge_cursor[static_cast<std::size_t>(b)]++] = {face.cell_b, face.cell_a};
        });
        release(received);

        // Keep unique faces in chunks; avoid repeatedly growing a large output array.
        chunks.emplace_back();
        mpi::alltoallv_packed(comm, std::span<const int>{face_counts},
                             std::span<const FaceRecord>{face_send}, chunks.back(), workspace, face_type);
        release(face_send);
        total_faces += chunks.back().size();
        for (const auto& face : chunks.back()) {
            if (face.cell_b == invalid_cell_id) {
                ++result.stats.boundary_faces;
            } else {
                ++result.stats.interior_faces;
                graph.add(face.cell_a, face.cell_b);
                if (mesh.distribution.cells.owner(face.cell_b) == comm.rank())
                    graph.add(face.cell_b, face.cell_a);
            }
        }
        std::vector<DualEdge> reverse_edges;
        mpi::alltoallv_packed(comm, std::span<const int>{edge_counts},
                             std::span<const DualEdge>{edge_send}, reverse_edges, workspace, edge_type);
        // Reverse edges are consumed now rather than retained until CSR assembly.
        for (const auto& edge : reverse_edges) graph.add(edge.cell, edge.neighbour);
    }
    release(sources);
    release(offsets);
    graph.finish();

    if (chunks.size() == 1) {
        result.faces = std::move(chunks.front());
    } else {
        // Allocate once and release each source chunk immediately after copying.
        result.faces.reserve(total_faces);
        for (auto& chunk : chunks) {
            result.faces.insert(result.faces.end(), chunk.begin(), chunk.end());
            release(chunk);
        }
    }
    result.stats.faces = result.faces.size();
    result.stats.graph_arcs = result.graph.adjacency.size();
    std::array<std::uint64_t, 4> totals{result.stats.faces, result.stats.boundary_faces,
                                       result.stats.interior_faces, result.stats.graph_arcs};
    // One reduction returns all global counts; there are no diagnostic reductions.
    mpi::allreduce_in_place<mpi::Reduction::sum>(comm, std::span{totals});
    result.stats = {totals[0], totals[1], totals[2], totals[3]};
    return result;
}

} // namespace

/// @brief Builds unique faces and symmetric adjacency from an owned 3D mesh snapshot.
BuildFacesResult build_faces(const RawMesh& mesh, mpi::CommView comm, const BuildFacesOptions& options)
{
    if (comm.is_null() || !mpi::is_active()) mpi::fatal("Face construction requires active MPI");
    mpi::require_main_thread();
    try {
        // Abort immediately on local allocation failure so peers cannot remain blocked.
        return build(mesh, comm, options);
    } catch (const std::exception& error) {
        mpi::fatal(error.what());
    } catch (...) {
        mpi::fatal("Unknown exception during face construction");
    }
}
 
} // namespace krona::mesh
