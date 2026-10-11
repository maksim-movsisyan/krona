#include <krona/mesh/migrate.hpp>
#include <krona/mpi/context.hpp>

#include "krona/mpi/error.hpp"
#include "migration_detail.hpp"

#include <array>
#include <numeric>
#include <span>

namespace krona::mesh {
namespace {

using namespace migration_detail;

/// @brief Resolves the original owner of a valid ID in contiguous rank ranges.
[[nodiscard]] int original_owner(std::span<const GlobalIndex> ranges, GlobalIndex gid)
{
    assert(gid >= 0 && gid < ranges.back());
    // upper_bound handles repeated endpoints belonging to empty ranks.
    return static_cast<int>(std::upper_bound(ranges.begin(), ranges.end(), gid) -
                            ranges.begin() - 1);
}

/// @brief Validates replicated ownership and local payload dimensions once at entry.
void check_source(const MigrationSource& source, std::span<const int> targets, mpi::CommView comm)
{
    const auto peers = static_cast<std::size_t>(comm.size());
    for (const auto& ranges : {std::span<const GlobalIndex>{source.cell_distribution},
                              std::span<const GlobalIndex>{source.node_distribution}}) {
        if (ranges.size() != peers + 1 || ranges.front() != 0 ||
            !std::is_sorted(ranges.begin(), ranges.end()))
            mpi::fatal("Migration requires P+1 nondecreasing ownership endpoints starting at zero");
    }
    const auto rank = static_cast<std::size_t>(comm.rank());
    const auto cells = static_cast<std::uint64_t>(source.cell_distribution[rank + 1] -
                                               source.cell_distribution[rank]);
    const auto nodes = static_cast<std::uint64_t>(source.node_distribution[rank + 1] -
                                               source.node_distribution[rank]);
    if (cells != source.cells.size() || targets.size() != source.cells.size() ||
        source.cells.offsets.size() != source.cells.size() + 1 ||
        source.cells.offsets.front() != 0 ||
        source.cells.offsets.back() != source.cells.node_ids.size() ||
        nodes != source.coordinates.size() || source.coordinates.y.size() != nodes ||
        source.coordinates.z.size() != nodes)
        mpi::fatal("Migration payload dimensions do not match source ownership");
#ifndef NDEBUG
    // Full topology validity is the caller's contract, not another collective check.
    for (std::size_t cell = 0; cell < source.cells.size(); ++cell) {
        assert(dimension(source.cells.types[cell]) == 3);
        assert(source.cells.nodes(cell).size() ==
               static_cast<std::size_t>(nodes_per_element(source.cells.types[cell])));
    }
    for (const auto gid : source.cells.node_ids)
        assert(gid >= 0 && gid < source.node_distribution.back());
#endif
}

struct TargetLookup {
    GlobalIndex first, last;
    std::span<const int> local;
    GidIndex remote;
    std::vector<int> destinations;

    /// @brief Resolves a destination from local targets or deduplicated remote replies.
    [[nodiscard]] int at(GlobalIndex gid) const
    {
        if (gid == invalid_cell_id) return -1;
        if (gid >= first && gid < last) return local[static_cast<std::size_t>(gid - first)];
        return destinations[static_cast<std::size_t>(remote.at(gid))];
    }
};

/// @brief Queries each distinct remote face-neighbour destination once.
[[nodiscard]] TargetLookup resolve_targets(
    const MigrationSource& source, std::span<const FaceRecord> faces,
    std::span<const int> targets, mpi::CommView comm)
{
    const auto rank = static_cast<std::size_t>(comm.rank());
    const auto first = source.cell_distribution[rank], last = source.cell_distribution[rank + 1];
    std::vector<GlobalIndex> ids;
    for (const auto& face : faces)
        if (face.cell_b != invalid_cell_id && (face.cell_b < first || face.cell_b >= last))
            ids.push_back(face.cell_b);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    TargetLookup result{first, last, targets, cell_index(ids), {}};
    if (comm.size() == 1) return result;

    // Sorted GIDs are already grouped by their original contiguous owners.
    std::vector<std::size_t> counts(static_cast<std::size_t>(comm.size()), 0);
    for (const auto gid : ids)
        ++counts[static_cast<std::size_t>(original_owner(source.cell_distribution, gid))];
    mpi::AlltoallvWorkspace work;
    auto requests = exchange(comm, layout(counts), std::move(ids), work);
    std::vector<int> answers;
    answers.reserve(requests.size());
    for (const auto gid : requests) answers.push_back(result.at(gid));
    release(requests);
    result.destinations = reply(comm, std::move(answers), work);
    return result;
}

struct CellHeader { GlobalIndex gid; CellType type; };
struct GlobalCells { std::vector<GlobalIndex> ids; ElementTable table; };

/// @brief Receives exact-length connectivity after cell headers have established its layout.
[[nodiscard]] GlobalCells receive_cells(
    mpi::CommView comm, std::vector<CellHeader>&& headers,
    std::vector<GlobalIndex>&& nodes, const Layout& send_nodes,
    std::span<const int> header_counts, std::span<const int> header_offsets)
{
    GlobalCells result;
    result.ids.reserve(headers.size());
    result.table.types.reserve(headers.size());
    result.table.offsets.reserve(headers.size() + 1);
    static_cast<void>(local_count(headers.size()));
    std::vector<std::size_t> counts(header_counts.size(), 0);
    for (std::size_t peer = 0; peer < counts.size(); ++peer) {
        const auto first = static_cast<std::size_t>(header_offsets[peer]);
        const auto last = first + static_cast<std::size_t>(header_counts[peer]);
        for (auto i = first; i < last; ++i)
            counts[peer] += static_cast<std::size_t>(nodes_per_element(headers[i].type));
    }
    for (const auto& header : headers) {
        result.ids.push_back(header.gid);
        result.table.types.push_back(header.type);
        result.table.offsets.push_back(result.table.offsets.back() +
            static_cast<std::size_t>(nodes_per_element(header.type)));
    }
    release(headers);
    if (comm.size() == 1) result.table.node_ids = std::move(nodes);
    else {
        const auto recv_nodes = layout(counts);
        result.table.node_ids.resize(recv_nodes.size);
        // Header arities give receive counts; no connectivity-count collective is needed.
        mpi::alltoallv(comm, std::span{nodes}, send_nodes.counts, send_nodes.offsets,
                      std::span{result.table.node_ids}, recv_nodes.counts, recv_nodes.offsets);
        release(nodes);
    }
    return result;
}

/// @brief Redistributes owned cells using small headers and unpadded connectivity arrays.
[[nodiscard]] GlobalCells move_cells(
    MigrationSource& source, std::span<const int> targets, mpi::CommView comm)
{
    const auto first = source.cell_distribution[static_cast<std::size_t>(comm.rank())];
    if (comm.size() == 1) {
        GlobalCells result;
        result.ids.resize(source.cells.size());
        std::iota(result.ids.begin(), result.ids.end(), first);
        result.table = std::move(source.cells);
        return result;
    }
    const auto peers = static_cast<std::size_t>(comm.size());
    std::vector<std::size_t> cell_counts(peers, 0), node_counts(peers, 0);
    for (std::size_t cell = 0; cell < targets.size(); ++cell) {
        const auto peer = static_cast<std::size_t>(targets[cell]);
        ++cell_counts[peer];
        node_counts[peer] += source.cells.nodes(cell).size();
    }
    const auto cells = layout(cell_counts), nodes = layout(node_counts);
    auto cell_cursor = cells.offsets, node_cursor = nodes.offsets;
    std::vector<CellHeader> headers(cells.size);
    std::vector<GlobalIndex> connectivity(nodes.size);
    for (std::size_t cell = 0; cell < targets.size(); ++cell) {
        const auto peer = static_cast<std::size_t>(targets[cell]);
        headers[static_cast<std::size_t>(cell_cursor[peer]++)] =
            {first + static_cast<GlobalIndex>(cell), source.cells.types[cell]};
        const auto row = source.cells.nodes(cell);
        std::copy(row.begin(), row.end(), connectivity.begin() + node_cursor[peer]);
        node_cursor[peer] += static_cast<int>(row.size());
    }
    // Free original topology before allocating receive connectivity.
    source.cells = ElementTable{};
    mpi::AlltoallvWorkspace work;
    auto received = exchange(comm, cells, std::move(headers), work);
    return receive_cells(comm, std::move(received), std::move(connectivity), nodes,
                         work.recv_counts, work.recv_displacements);
}

struct FacePacket {
    GlobalIndex owner, neighbour;
    PatchId patch;
    int donor;
    std::uint8_t facet;
};
struct GhostEntry {
    GlobalIndex gid;
    int donor;
    /// @brief Orders and deduplicates ghosts by their new donor and source cell ID.
    auto operator<=>(const GhostEntry& other) const noexcept
    {
        if (const auto order = donor <=> other.donor; order != 0) return order;
        return gid <=> other.gid;
    }
    /// @brief Identifies repeated requests for the same ghost and donor.
    bool operator==(const GhostEntry&) const = default;
};
struct LocalFace {
    LocalIndex owner, neighbour;
    PatchId patch;
    std::uint8_t facet;
};

/// @brief Sends compact oriented faces and counts actual unique, boundary and cut faces.
[[nodiscard]] std::vector<FacePacket> move_faces(
    std::vector<FaceRecord>& faces, const TargetLookup& targets, mpi::CommView comm,
    std::array<std::uint64_t, 6>& totals)
{
    std::vector<std::size_t> counts(static_cast<std::size_t>(comm.size()), 0);
    totals[1] = faces.size();
    for (const auto& face : faces) {
        const int a = targets.at(face.cell_a), b = targets.at(face.cell_b);
        ++counts[static_cast<std::size_t>(a)];
        if (b >= 0 && a != b) { ++counts[static_cast<std::size_t>(b)]; ++totals[4]; }
        if (b < 0) {
            ++totals[2];
            if (face.patch != invalid_patch_id) ++totals[3];
        }
    }
    const auto plan = layout(counts);
    auto cursor = plan.offsets;
    std::vector<FacePacket> send(plan.size);
    for (const auto& face : faces) {
        const int a = targets.at(face.cell_a), b = targets.at(face.cell_b);
        send[static_cast<std::size_t>(cursor[static_cast<std::size_t>(a)]++)] = {
            face.cell_a, face.cell_b, b < 0 ? face.patch : invalid_patch_id,
            b >= 0 && a != b ? b : -1, face.local_face_a};
        if (b >= 0 && a != b) {
            // Swap owner and neighbour; use cell_b's facet to preserve orientation.
            send[static_cast<std::size_t>(cursor[static_cast<std::size_t>(b)]++)] = {
                face.cell_b, face.cell_a, invalid_patch_id, a, face.local_face_b};
        }
    }
    release(faces);
    mpi::AlltoallvWorkspace work;
    return exchange(comm, plan, std::move(send), work);
}

/// @brief Fetches one-hop ghost topology and builds halo maps from the same requests.
[[nodiscard]] GlobalCells fetch_ghosts(
    const GlobalCells& owned, const GidIndex& owned_index, std::span<const GhostEntry> ghosts,
    MeshPart& mesh, mpi::CommView comm)
{
    if (comm.size() == 1) return {};
    const auto peers = static_cast<std::size_t>(comm.size());
    std::vector<std::size_t> counts(peers, 0);
    std::vector<GlobalIndex> ids;
    ids.reserve(ghosts.size());
    for (const auto& ghost : ghosts) {
        ++counts[static_cast<std::size_t>(ghost.donor)];
        ids.push_back(ghost.gid);
    }
    mpi::AlltoallvWorkspace work;
    auto requests = exchange(comm, layout(counts), std::move(ids), work);

    // A request's position becomes its future send-halo position on the donor.
    mesh.send_owned_local.reserve(requests.size());
    for (const auto gid : requests) mesh.send_owned_local.push_back(owned_index.at(gid));
    mesh.recv_ghost_local.resize(ghosts.size());
    std::iota(mesh.recv_ghost_local.begin(), mesh.recv_ghost_local.end(), mesh.n_own);
    for (std::size_t peer = 0; peer < peers; ++peer) {
        if (counts[peer] == 0 && work.recv_counts[peer] == 0) continue;
        mesh.nb_ranks.push_back(static_cast<int>(peer));
        mesh.recv_offsets.push_back(mesh.recv_offsets.back() + counts[peer]);
        mesh.send_offsets.push_back(mesh.send_offsets.back() +
                                   static_cast<std::size_t>(work.recv_counts[peer]));
    }
    std::vector<std::size_t> node_counts(peers, 0);
    for (std::size_t peer = 0; peer < peers; ++peer) {
        const auto first = static_cast<std::size_t>(work.recv_displacements[peer]);
        const auto last = first + static_cast<std::size_t>(work.recv_counts[peer]);
        for (auto i = first; i < last; ++i)
            node_counts[peer] += owned.table.nodes(
                static_cast<std::size_t>(mesh.send_owned_local[i])).size();
    }
    const auto nodes = layout(node_counts);
    std::vector<CellHeader> headers;
    std::vector<GlobalIndex> connectivity;
    headers.reserve(requests.size());
    connectivity.reserve(nodes.size);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto cell = static_cast<std::size_t>(mesh.send_owned_local[i]);
        headers.push_back({requests[i], owned.table.types[cell]});
        const auto row = owned.table.nodes(cell);
        connectivity.insert(connectivity.end(), row.begin(), row.end());
    }
    release(requests);
    auto received = reply(comm, std::move(headers), work);
    // Responses return in the donor/GID order of the original ghost requests.
    return receive_cells(comm, std::move(received), std::move(connectivity), nodes,
                         work.send_counts, work.send_displacements);
}

/// @brief Numbers unique nodes and converts owned/ghost global connectivity to local CSR.
void assemble_cells(GlobalCells& owned, GlobalCells& ghosts,
                    std::span<const GhostEntry> donors, MeshPart& mesh)
{
    GlobalIndex lo = std::numeric_limits<GlobalIndex>::max(), hi = -1;
    for (const auto& ids : {std::span<const GlobalIndex>{owned.table.node_ids},
                           std::span<const GlobalIndex>{ghosts.table.node_ids}})
        for (const auto gid : ids) { lo = std::min(lo, gid); hi = std::max(hi, gid); }
    if (hi < lo) lo = 0;
    GidIndex nodes(lo, hi, static_cast<std::size_t>(mesh.n_cells));
    for (const auto gid : owned.table.node_ids)
        if (nodes.insert(gid, 0) && !nodes.dense()) mesh.node_gid.push_back(gid);
    const auto owned_nodes = nodes.size();
    for (const auto gid : ghosts.table.node_ids)
        if (nodes.insert(gid, 1) && !nodes.dense()) mesh.node_gid.push_back(gid);
    mesh.n_nodes_own = local_count(owned_nodes);
    mesh.n_nodes = local_count(nodes.size());
    if (nodes.dense()) {
        mesh.node_gid.resize(nodes.size());
        nodes.write_nodes(mesh.node_gid, owned_nodes);
    } else {
        const auto split = mesh.node_gid.begin() + mesh.n_nodes_own;
        std::sort(mesh.node_gid.begin(), split);
        std::sort(split, mesh.node_gid.end());
    }
    for (std::size_t i = 0; i < mesh.node_gid.size(); ++i)
        nodes.assign(mesh.node_gid[i], static_cast<LocalIndex>(i));

    mesh.cell_gid = std::move(owned.ids);
    mesh.cell_type = std::move(owned.table.types);
    mesh.cell_nodes_offsets = std::move(owned.table.offsets);
    const auto cells = static_cast<std::size_t>(mesh.n_cells);
    mesh.cell_gid.reserve(cells);
    mesh.cell_type.reserve(cells);
    mesh.cell_nodes_offsets.reserve(cells + 1);
    mesh.cell_gid.insert(mesh.cell_gid.end(), ghosts.ids.begin(), ghosts.ids.end());
    mesh.cell_type.insert(mesh.cell_type.end(), ghosts.table.types.begin(), ghosts.table.types.end());
    mesh.cell_donor.assign(cells, -1);
    for (std::size_t i = 0; i < donors.size(); ++i)
        mesh.cell_donor[static_cast<std::size_t>(mesh.n_own) + i] = donors[i].donor;
    const auto owned_entries = owned.table.node_ids.size();
    for (std::size_t i = 1; i < ghosts.table.offsets.size(); ++i)
        mesh.cell_nodes_offsets.push_back(owned_entries + ghosts.table.offsets[i]);
    mesh.cell_nodes.resize(owned_entries + ghosts.table.node_ids.size());
    for (std::size_t i = 0; i < owned_entries; ++i)
        mesh.cell_nodes[i] = nodes.at(owned.table.node_ids[i]);
    release(owned.table.node_ids);
    for (std::size_t i = 0; i < ghosts.table.node_ids.size(); ++i)
        mesh.cell_nodes[owned_entries + i] = nodes.at(ghosts.table.node_ids[i]);
    ghosts = GlobalCells{};
    // The node lookup and global connectivity die before coordinate allocation.
}

struct CoordinateReply { GeometryReal x, y, z; };

/// @brief Copies local coordinates and requests only missing triples from original owners.
void fetch_coordinates(MigrationSource& source, MeshPart& mesh, mpi::CommView comm)
{
    const auto rank = static_cast<std::size_t>(comm.rank());
    const auto first = source.node_distribution[rank], last = source.node_distribution[rank + 1];
    if (comm.size() == 1 && mesh.node_gid.size() == source.coordinates.size()) {
        // All input nodes are used in their original sorted order on a single rank.
        mesh.node_x = std::move(source.coordinates.x);
        mesh.node_y = std::move(source.coordinates.y);
        mesh.node_z = std::move(source.coordinates.z);
        return;
    }
    const auto size = mesh.node_gid.size();
    mesh.node_x.resize(size); mesh.node_y.resize(size); mesh.node_z.resize(size);
    std::vector<std::size_t> counts(static_cast<std::size_t>(comm.size()), 0);
    for (std::size_t i = 0; i < size; ++i) {
        const auto gid = mesh.node_gid[i];
        if (gid >= first && gid < last) {
            const auto local = static_cast<std::size_t>(gid - first);
            mesh.node_x[i] = source.coordinates.x[local];
            mesh.node_y[i] = source.coordinates.y[local];
            mesh.node_z[i] = source.coordinates.z[local];
        } else ++counts[static_cast<std::size_t>(original_owner(source.node_distribution, gid))];
    }
    if (comm.size() == 1) { source.coordinates = {}; return; }
    const auto plan = layout(counts);
    auto cursor = plan.offsets;
    std::vector<GlobalIndex> send(plan.size);
    std::vector<LocalIndex> slots(plan.size);
    for (std::size_t i = 0; i < size; ++i) {
        const auto gid = mesh.node_gid[i];
        if (gid >= first && gid < last) continue;
        const auto peer = static_cast<std::size_t>(original_owner(source.node_distribution, gid));
        const auto slot = static_cast<std::size_t>(cursor[peer]++);
        send[slot] = gid;
        slots[slot] = static_cast<LocalIndex>(i);
    }
    mpi::AlltoallvWorkspace work;
    auto requests = exchange(comm, plan, std::move(send), work);
    std::vector<CoordinateReply> answers;
    answers.reserve(requests.size());
    for (const auto gid : requests) {
        const auto local = static_cast<std::size_t>(gid - first);
        answers.push_back({source.coordinates.x[local], source.coordinates.y[local],
                           source.coordinates.z[local]});
    }
    release(requests);
    source.coordinates = {};
    auto received = reply(comm, std::move(answers), work);
    for (std::size_t i = 0; i < received.size(); ++i) {
        const auto slot = static_cast<std::size_t>(slots[i]);
        mesh.node_x[slot] = received[i].x;
        mesh.node_y[slot] = received[i].y;
        mesh.node_z[slot] = received[i].z;
    }
}

/// @brief Resolves compact local face references before releasing cell-GID lookups.
[[nodiscard]] std::vector<LocalFace> localize_faces(
    std::vector<FacePacket>& faces, const GidIndex& owned, const GidIndex& ghosts,
    LocalIndex owned_count)
{
    static_cast<void>(local_count(faces.size()));
    std::vector<LocalFace> result;
    result.reserve(faces.size());
    for (const auto& face : faces) {
        const auto neighbour = face.neighbour == invalid_cell_id ? invalid_local_index :
            face.donor < 0 ? owned.at(face.neighbour) :
            static_cast<LocalIndex>(owned_count + ghosts.at(face.neighbour));
        result.push_back({owned.at(face.owner), neighbour, face.patch, face.facet});
    }
    // GIDs and donor ranks are no longer needed in the face buffer.
    release(faces);
    return result;
}

/// @brief Reconstructs oriented face-node CSR without RCM/SFC or face grouping.
void assemble_faces(std::vector<LocalFace>& faces, MeshPart& mesh)
{
    mesh.n_faces = local_count(faces.size());
    mesh.face_owner.reserve(faces.size()); mesh.face_neigh.reserve(faces.size());
    mesh.face_patch.reserve(faces.size()); mesh.face_type.reserve(faces.size());
    mesh.face_local_index.reserve(faces.size()); mesh.face_nodes_offsets.reserve(faces.size() + 1);
    for (const auto& face : faces) {
        const auto type = static_cast<std::size_t>(mesh.cell_type[static_cast<std::size_t>(face.owner)]);
        assert(face.facet < faces_per_type[type]);
        const auto arity = face_nodes[type][face.facet];
        mesh.face_owner.push_back(face.owner); mesh.face_neigh.push_back(face.neighbour);
        mesh.face_patch.push_back(face.patch); mesh.face_type.push_back(arity == 3 ? CellType::TRI : CellType::QUAD);
        mesh.face_local_index.push_back(face.facet);
        mesh.face_nodes_offsets.push_back(mesh.face_nodes_offsets.back() + static_cast<std::size_t>(arity));
    }
    mesh.face_nodes.resize(mesh.face_nodes_offsets.back());
    for (std::size_t f = 0; f < faces.size(); ++f) {
        const auto owner = static_cast<std::size_t>(mesh.face_owner[f]);
        const auto type = static_cast<std::size_t>(mesh.cell_type[owner]);
        const auto cell_begin = mesh.cell_nodes_offsets[owner], face_begin = mesh.face_nodes_offsets[f];
        for (int i = 0; i < face_nodes[type][faces[f].facet]; ++i)
            mesh.face_nodes[face_begin + static_cast<std::size_t>(i)] = mesh.cell_nodes[cell_begin +
                static_cast<std::size_t>(face_table[type][faces[f].facet][i])];
    }
    release(faces);
}

/// @brief Builds patch-face CSR including labelled exterior faces only.
void assemble_patches(MeshPart& mesh)
{
    mesh.patch_face_offsets.assign(mesh.patches.size() + 1, 0);
    for (const auto patch : mesh.face_patch) {
        if (patch == invalid_patch_id) continue;
        if (patch < 0 || static_cast<std::size_t>(patch) >= mesh.patches.size())
            mpi::fatal("Migrated boundary face has an invalid patch ID");
        ++mesh.patch_face_offsets[static_cast<std::size_t>(patch) + 1];
    }
    std::partial_sum(mesh.patch_face_offsets.begin(), mesh.patch_face_offsets.end(),
                     mesh.patch_face_offsets.begin());
    mesh.patch_faces.resize(mesh.patch_face_offsets.back());
    auto cursor = mesh.patch_face_offsets;
    for (std::size_t face = 0; face < mesh.face_patch.size(); ++face) {
        const auto patch = mesh.face_patch[face];
        if (patch != invalid_patch_id)
            mesh.patch_faces[cursor[static_cast<std::size_t>(patch)]++] = static_cast<LocalIndex>(face);
    }
}

/// @brief Reduces actual topology totals and the bounds of nodes used by owned cells.
void global_statistics(MeshPart& mesh, std::array<std::uint64_t, 6> totals, mpi::CommView comm)
{
    totals[0] = static_cast<std::uint64_t>(mesh.n_own);
    totals[5] = static_cast<std::uint64_t>(mesh.n_cells - mesh.n_own);
    if (comm.size() > 1) mpi::allreduce_in_place<mpi::Reduction::sum>(comm, std::span{totals});
    if (totals[0] != mesh.n_cells_g) mpi::fatal("Migration did not preserve the global cell count");
    mesh.n_faces_g = totals[1]; mesh.n_bfaces_g = totals[2];
    mesh.n_labeled_bfaces_g = totals[3]; mesh.n_cut_faces_g = totals[4];
    mesh.n_ghost_copies_g = totals[5];
    if (mesh.n_cells_g == 0) return; // Empty meshes retain zero bounds.
    std::array<GeometryReal, 6> bounds;
    bounds.fill(std::numeric_limits<GeometryReal>::infinity());
    for (std::size_t i = 0; i < static_cast<std::size_t>(mesh.n_nodes_own); ++i) {
        const std::array<GeometryReal, 3> node{mesh.node_x[i], mesh.node_y[i], mesh.node_z[i]};
        for (std::size_t d = 0; d < 3; ++d) {
            bounds[d] = std::min(bounds[d], node[d]);
            bounds[d + 3] = std::min(bounds[d + 3], -node[d]);
        }
    }
    // Negated maxima share the same MIN reduction as the minima.
    if (comm.size() > 1) mpi::allreduce_in_place<mpi::Reduction::min>(comm, std::span{bounds});
    for (std::size_t d = 0; d < 3; ++d) { mesh.bbox_lo[d] = bounds[d]; mesh.bbox_hi[d] = -bounds[d + 3]; }
}

/// @brief Runs ownership migration, halo construction and local topology assembly.
[[nodiscard]] MeshPart migrate(MigrationSource& source, std::vector<FaceRecord>& faces,
                               std::span<const int> targets, mpi::CommView comm)
{
    check_source(source, targets, comm);
    for (const auto target : targets)
        if (target < 0 || target >= comm.size()) mpi::fatal("Invalid cell migration destination rank");
    MeshPart mesh;
    mesh.rank = comm.rank(); mesh.nprocs = comm.size();
    mesh.n_cells_g = static_cast<std::uint64_t>(source.cell_distribution.back());
    mesh.n_nodes_g = static_cast<std::uint64_t>(source.node_distribution.back());
    mesh.patches = std::move(source.patches);
    std::array<std::uint64_t, 6> totals{};
    GlobalCells owned;
    std::vector<FacePacket> received_faces;
    {
        auto destinations = resolve_targets(source, faces, targets, comm);
        owned = move_cells(source, targets, comm);
        received_faces = move_faces(faces, destinations, comm, totals);
        // Remote destination lookups expire before ghost and node storage is allocated.
    }
    mesh.n_own = local_count(owned.ids.size());
    std::vector<GhostEntry> ghost_entries;
    for (const auto& face : received_faces)
        if (face.donor >= 0) ghost_entries.push_back({face.neighbour, face.donor});
    std::sort(ghost_entries.begin(), ghost_entries.end());
    ghost_entries.erase(std::unique(ghost_entries.begin(), ghost_entries.end()), ghost_entries.end());
    mesh.n_cells = local_count(owned.ids.size() + ghost_entries.size());
    GlobalCells ghosts;
    std::vector<LocalFace> local_faces;
    {
        auto owned_lookup = cell_index(owned.ids);
        ghosts = fetch_ghosts(owned, owned_lookup, ghost_entries, mesh, comm);
        auto ghost_lookup = cell_index(ghosts.ids);
        local_faces = localize_faces(received_faces, owned_lookup, ghost_lookup, mesh.n_own);
        // Cell-GID lookups die before the larger node-GID lookup is allocated.
    }
    assemble_cells(owned, ghosts, ghost_entries, mesh);
    release(ghost_entries);
    assemble_faces(local_faces, mesh);
    fetch_coordinates(source, mesh, comm);
    release(source.cell_distribution); release(source.node_distribution);
    assemble_patches(mesh);
    global_statistics(mesh, totals, comm);
    return mesh;
}

} // namespace

/// @brief Migrates source-independent payloads collectively, consuming their storage.
MeshPart migrate_local_mesh(MigrationSource&& source, std::vector<FaceRecord>&& faces,
                            std::span<const int> targets, mpi::CommView comm)
{
    mpi::require(!comm.is_null() && mpi::is_active(),"Migration requires active MPI");
    mpi::require_main_thread();
    return mpi::guarded([&](){
        return migrate(source, faces, targets, comm);
    });
}

/// @brief Adapts a 3D block-distributed import without copying cell or coordinate payloads.
MeshPart migrate_local_mesh(RawMesh&& source, std::vector<FaceRecord>&& faces,
                            std::span<const int> targets, mpi::CommView comm)
{
    mpi::require(!comm.is_null() && mpi::is_active(),"Migration requires active MPI");
    mpi::require_main_thread();
    return mpi::guarded([&](){
        if (source.cell_dimension != 3 || source.physical_dimension != 3 ||
            source.distribution.rank != comm.rank() ||
            source.distribution.cells.parts() != comm.size() ||
            source.distribution.nodes.parts() != comm.size())
            mpi::fatal("Migration requires a 3D import matching the communicator");
        MigrationSource input;
        input.cell_distribution.resize(static_cast<std::size_t>(comm.size()) + 1);
        input.node_distribution.resize(static_cast<std::size_t>(comm.size()) + 1);
        for (int rank = 0; rank < comm.size(); ++rank) {
            input.cell_distribution[static_cast<std::size_t>(rank)] = source.distribution.cells.begin(rank);
            input.node_distribution[static_cast<std::size_t>(rank)] = source.distribution.nodes.begin(rank);
        }
        input.cell_distribution.back() = source.distribution.cells.global_size();
        input.node_distribution.back() = source.distribution.nodes.global_size();
        input.cells = std::move(source.cells);
        input.coordinates = std::move(source.coordinates);
        input.patches = std::move(source.patches);
        // Boundary labels already live in FaceRecord; original surface payload is dead.
        source.boundary = {};
        return migrate(input, faces, targets, comm);
    });
}

} // namespace krona::mesh
