#include <krona/mesh/validate.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <span>
#include <tuple>

namespace krona::mesh {
namespace {

/// @brief Reports the first invalid invariant without allocating diagnostic strings.
[[noreturn]] void invalid(const MeshPart& mesh, const char* reason,
                         const char* entity = "mesh", std::size_t index = 0)
{
    char message[320];
    std::snprintf(message, sizeof(message), "Mesh validation: rank=%d %s=%zu: %s",
                  mesh.rank, entity, index, reason);
    mpi::fatal(message);
}

/// @brief Validates CSR dimensions and monotone offsets before payload indexing.
void check_csr(const MeshPart& mesh, const std::vector<std::size_t>& offsets,
               std::size_t rows, std::size_t payload, const char* name)
{
    // Check dimensions first so even an empty/malformed offsets array is safe to inspect.
    if (offsets.size() != rows + 1 || offsets.front() != 0 || offsets.back() != payload ||
        !std::is_sorted(offsets.begin(), offsets.end())) invalid(mesh, "Invalid CSR offsets", name);
}

/// @brief Checks finite coordinate or centroid components in stored precision.
[[nodiscard]] bool finite(GeometryReal x, GeometryReal y, GeometryReal z) noexcept
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

/// @brief Checks local entity counts, array extents and required metric storage.
void check_shape(const MeshPart& mesh)
{
    if (mesh.nprocs <= 0 || mesh.rank < 0 || mesh.rank >= mesh.nprocs ||
        mesh.n_own < 0 || mesh.n_cells < mesh.n_own || mesh.n_nodes_own < 0 ||
        mesh.n_nodes < mesh.n_nodes_own || mesh.n_faces < 0 || !mesh.faces_grouped ||
        mesh.n_internal_faces < 0 || mesh.n_internal_faces > mesh.n_inner_faces ||
        mesh.n_inner_faces > mesh.n_faces) invalid(mesh, "Invalid counts or unfinished face grouping");
    const auto cells = static_cast<std::size_t>(mesh.n_cells);
    const auto nodes = static_cast<std::size_t>(mesh.n_nodes);
    const auto faces = static_cast<std::size_t>(mesh.n_faces);
    if (mesh.cell_type.size() != cells || mesh.cell_gid.size() != cells || mesh.cell_donor.size() != cells ||
        mesh.cell_volume.size() != cells || mesh.cell_centroid_x.size() != cells ||
        mesh.cell_centroid_y.size() != cells || mesh.cell_centroid_z.size() != cells)
        invalid(mesh, "Cell array dimensions do not match n_cells");
    if (mesh.node_gid.size() != nodes || mesh.node_x.size() != nodes ||
        mesh.node_y.size() != nodes || mesh.node_z.size() != nodes)
        invalid(mesh, "Node array dimensions do not match n_nodes");
    if (mesh.face_owner.size() != faces || mesh.face_neigh.size() != faces || mesh.face_type.size() != faces ||
        mesh.face_patch.size() != faces || mesh.face_local_index.size() != faces || mesh.face_area.size() != faces ||
        mesh.face_centroid_x.size() != faces || mesh.face_centroid_y.size() != faces || mesh.face_centroid_z.size() != faces ||
        mesh.face_normal_x.size() != faces || mesh.face_normal_y.size() != faces || mesh.face_normal_z.size() != faces)
        invalid(mesh, "Face array dimensions do not match n_faces");
    check_csr(mesh, mesh.cell_nodes_offsets, cells, mesh.cell_nodes.size(), "cell_nodes");
    check_csr(mesh, mesh.face_nodes_offsets, faces, mesh.face_nodes.size(), "face_nodes");
    check_csr(mesh, mesh.patch_face_offsets, mesh.patches.size(), mesh.patch_faces.size(), "patch_faces");
}

/// @brief Checks owned/ghost cell topology, global IDs, donors and positive finite metrics.
void check_cells(const MeshPart& mesh)
{
    for (std::size_t c = 0; c < mesh.cell_type.size(); ++c) {
        const auto type = static_cast<std::size_t>(mesh.cell_type[c]);
        const auto begin = mesh.cell_nodes_offsets[c], end = mesh.cell_nodes_offsets[c + 1];
        if (type >= 4 || end - begin != static_cast<std::size_t>(nodes_per_type[type]))
            invalid(mesh, "Unsupported cell type or connectivity arity", "cell", c);
        if (mesh.cell_gid[c] < 0 || static_cast<std::uint64_t>(mesh.cell_gid[c]) >= mesh.n_cells_g)
            invalid(mesh, "Cell global ID outside snapshot range", "cell", c);
        const bool owned = c < static_cast<std::size_t>(mesh.n_own);
        const int donor = mesh.cell_donor[c];
        if (owned ? donor != -1 : donor < 0 || donor >= mesh.nprocs || donor == mesh.rank)
            invalid(mesh, "Invalid owned/ghost cell donor", "cell", c);
        for (auto k = begin; k < end; ++k) {
            const auto node = mesh.cell_nodes[k];
            if (node < 0 || node >= mesh.n_nodes || (owned && node >= mesh.n_nodes_own))
                invalid(mesh, "Cell node outside its allowed local range", "cell", c);
        }
        if (!(mesh.cell_volume[c] > 0) || !std::isfinite(mesh.cell_volume[c]) ||
            !finite(mesh.cell_centroid_x[c], mesh.cell_centroid_y[c], mesh.cell_centroid_z[c]))
            invalid(mesh, "Nonpositive or nonfinite cell metrics", "cell", c);
    }
    // Coordinate validity is checked once per node, not once per incident cell/face.
    for (std::size_t n = 0; n < mesh.node_gid.size(); ++n)
        if (mesh.node_gid[n] < 0 || static_cast<std::uint64_t>(mesh.node_gid[n]) >= mesh.n_nodes_g ||
            !finite(mesh.node_x[n], mesh.node_y[n], mesh.node_z[n]))
            invalid(mesh, "Invalid node global ID or coordinates", "node", n);
}

/// @brief Checks the three face ranges, sorted keys, owner facets and stored metrics.
void check_faces(const MeshPart& mesh)
{
    constexpr double normal_tolerance = 32 * static_cast<double>(std::numeric_limits<GeometryReal>::epsilon());
    for (std::size_t f = 0; f < mesh.face_owner.size(); ++f) {
        const auto owner = mesh.face_owner[f], neighbour = mesh.face_neigh[f];
        const auto patch = mesh.face_patch[f];
        if (owner < 0 || owner >= mesh.n_own) invalid(mesh, "Face owner is not owned", "face", f);
        const bool internal = f < static_cast<std::size_t>(mesh.n_internal_faces);
        const bool inner = f < static_cast<std::size_t>(mesh.n_inner_faces);
        if (internal ? neighbour < 0 || neighbour >= mesh.n_own || neighbour == owner :
            inner ? neighbour < mesh.n_own || neighbour >= mesh.n_cells : neighbour != invalid_local_index)
            invalid(mesh, "Neighbour does not match face region", "face", f);
        if (inner ? patch != invalid_patch_id :
            patch != invalid_patch_id && (patch < 0 || static_cast<std::size_t>(patch) >= mesh.patches.size()))
            invalid(mesh, "Invalid patch assignment", "face", f);
        if (f > 0 && inner && f != static_cast<std::size_t>(mesh.n_internal_faces)) {
            if (std::tie(mesh.face_owner[f - 1], mesh.face_neigh[f - 1]) > std::tie(owner, neighbour))
                invalid(mesh, "Interior/coupled faces are not sorted", "face", f);
        } else if (f > static_cast<std::size_t>(mesh.n_inner_faces)) {
            const auto previous = mesh.face_patch[f - 1];
            const auto a = previous == invalid_patch_id ? std::numeric_limits<PatchId>::max() : previous;
            const auto b = patch == invalid_patch_id ? std::numeric_limits<PatchId>::max() : patch;
            if (std::tie(a, mesh.face_owner[f - 1]) > std::tie(b, owner))
                invalid(mesh, "Boundary faces are not sorted by patch/owner", "face", f);
        }
        const auto cell = static_cast<std::size_t>(owner);
        const auto type = static_cast<std::size_t>(mesh.cell_type[cell]);
        const auto facet = mesh.face_local_index[f];
        const auto begin = mesh.face_nodes_offsets[f], end = mesh.face_nodes_offsets[f + 1];
        if (facet >= faces_per_type[type] || end - begin != static_cast<std::size_t>(face_nodes[type][facet]) ||
            mesh.face_type[f] != (end - begin == 3 ? CellType::TRI : CellType::QUAD))
            invalid(mesh, "Face type/arity does not match owner facet", "face", f);
        // Exact oriented-loop comparison catches stale facet/connectivity after reordering.
        for (auto k = begin; k < end; ++k) {
            const auto slot = static_cast<std::size_t>(face_table[type][facet][k - begin]);
            if (mesh.face_nodes[k] != mesh.cell_nodes[mesh.cell_nodes_offsets[cell] + slot])
                invalid(mesh, "Face node loop differs from owner facet", "face", f);
        }
        const double length = std::hypot(static_cast<double>(mesh.face_normal_x[f]),
            static_cast<double>(mesh.face_normal_y[f]), static_cast<double>(mesh.face_normal_z[f]));
        if (!(mesh.face_area[f] > 0) || !std::isfinite(mesh.face_area[f]) ||
            !finite(mesh.face_centroid_x[f], mesh.face_centroid_y[f], mesh.face_centroid_z[f]) ||
            !std::isfinite(length) || std::abs(length - 1) > normal_tolerance)
            invalid(mesh, "Invalid face area, centroid or unit normal", "face", f);
        // Outward winding was checked with local offsets by geometry.cpp; rounded world centres
        // cannot safely repeat that test on strongly shifted, thin meshes.
    }
}

/// @brief Verifies that patch CSR lists every labelled boundary face exactly once in order.
void check_patches(const MeshPart& mesh)
{
    auto expected = static_cast<std::size_t>(mesh.n_inner_faces);
    for (std::size_t p = 0; p < mesh.patches.size(); ++p) {
        for (auto k = mesh.patch_face_offsets[p]; k < mesh.patch_face_offsets[p + 1]; ++k) {
            if (expected >= mesh.face_patch.size() || mesh.face_patch[expected] != static_cast<PatchId>(p) ||
                mesh.patch_faces[k] != static_cast<LocalIndex>(expected))
                invalid(mesh, "Patch CSR does not match sorted boundary faces", "patch", p);
            ++expected;
        }
        if (expected < mesh.face_patch.size() && mesh.face_patch[expected] == static_cast<PatchId>(p))
            invalid(mesh, "Labelled boundary face is missing from patch CSR", "patch", p);
    }
    if (expected < mesh.face_patch.size() && mesh.face_patch[expected] != invalid_patch_id)
        invalid(mesh, "Patch CSR leaves labelled boundary faces unlisted");
}

/// @brief Checks peer layouts, send ranges and one receive slot per ghost without communication.
void check_halo(const MeshPart& mesh)
{
    const auto peers = mesh.nb_ranks.size();
    check_csr(mesh, mesh.send_offsets, peers, mesh.send_owned_local.size(), "halo_send");
    check_csr(mesh, mesh.recv_offsets, peers, mesh.recv_ghost_local.size(), "halo_recv");
    const auto ghosts = static_cast<std::size_t>(mesh.n_cells - mesh.n_own);
    if (mesh.recv_ghost_local.size() != ghosts) invalid(mesh, "Halo receive count differs from ghost count");
    std::vector<std::uint8_t> seen(ghosts, 0);
    for (std::size_t p = 0; p < peers; ++p) {
        const int rank = mesh.nb_ranks[p];
        if (rank < 0 || rank >= mesh.nprocs || rank == mesh.rank || (p > 0 && rank <= mesh.nb_ranks[p - 1]))
            invalid(mesh, "Invalid or unsorted halo peer ranks", "peer", p);
        for (auto k = mesh.send_offsets[p]; k < mesh.send_offsets[p + 1]; ++k)
            if (mesh.send_owned_local[k] < 0 || mesh.send_owned_local[k] >= mesh.n_own)
                invalid(mesh, "Halo send references a non-owned cell", "peer", p);
        for (auto k = mesh.recv_offsets[p]; k < mesh.recv_offsets[p + 1]; ++k) {
            const auto cell = mesh.recv_ghost_local[k];
            if (cell < mesh.n_own || cell >= mesh.n_cells) invalid(mesh, "Halo receive references a non-ghost cell", "peer", p);
            const auto index = static_cast<std::size_t>(cell);
            const auto ghost = static_cast<std::size_t>(cell - mesh.n_own);
            if (seen[ghost] != 0 || mesh.cell_donor[index] != rank)
                invalid(mesh, "Duplicate ghost receive or wrong donor group", "peer", p);
            seen[ghost] = 1;
        }
    }
}

} // namespace

/// @brief Runs local structural checks with immediate WORLD-abort on invalid data.
void validate_meshpart(const MeshPart& mesh)
{
    if (mpi::is_active()) mpi::require_main_thread();
    try {
        check_shape(mesh); check_cells(mesh); check_faces(mesh); check_patches(mesh); check_halo(mesh);
    } catch (const std::exception& error) { mpi::fatal(error.what()); }
      catch (...) { mpi::fatal("Unknown exception during local mesh validation"); }
}

/// @brief Reduces exact counts, flux-area sums and packed load extrema, then checks conservation.
MeshStatistics collect_mesh_statistics(const MeshPart& mesh, mpi::CommView comm)
{
    if (comm.is_null() || !mpi::is_active()) mpi::fatal("Mesh statistics require active MPI");
    mpi::require_main_thread();
    if (mesh.rank != comm.rank() || mesh.nprocs != comm.size()) mpi::fatal("Mesh statistics communicator mismatch");
    try {
        const auto patches = mesh.patches.size();
        std::vector<std::uint64_t> counts(6 + patches);
        counts[0] = static_cast<std::uint64_t>(mesh.n_own);
        counts[1] = static_cast<std::uint64_t>(mesh.n_cells - mesh.n_own);
        counts[2] = static_cast<std::uint64_t>(mesh.n_internal_faces);
        counts[3] = static_cast<std::uint64_t>(mesh.n_inner_faces - mesh.n_internal_faces);
        counts[4] = static_cast<std::uint64_t>(mesh.n_faces - mesh.n_inner_faces);
        counts[5] = static_cast<std::uint64_t>(mesh.patch_faces.size());
        std::vector<double> measures(1 + patches, 0);
        for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_own); ++c)
            measures[0] += static_cast<double>(mesh.cell_volume[c]);
        for (std::size_t p = 0; p < patches; ++p) {
            counts[p + 6] = mesh.patch_face_offsets[p + 1] - mesh.patch_face_offsets[p];
            for (auto k = mesh.patch_face_offsets[p]; k < mesh.patch_face_offsets[p + 1]; ++k)
                measures[p + 1] += static_cast<double>(mesh.face_area[static_cast<std::size_t>(mesh.patch_faces[k])]);
        }
        constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
        // Complemented maxima share an exact uint64 MIN reduction with minima.
        std::array<std::uint64_t, 4> extrema{counts[0], maximum - counts[0], counts[1], maximum - counts[1]};
        if (comm.size() > 1) {
            mpi::allreduce_in_place<mpi::Reduction::sum>(comm, std::span{counts});
            mpi::allreduce_in_place<mpi::Reduction::sum>(comm, std::span{measures});
            mpi::allreduce_in_place<mpi::Reduction::min>(comm, std::span{extrema});
        }
        const auto cuts = counts[3] / 2;
        if (counts[3] % 2 != 0 || counts[0] != mesh.n_cells_g || counts[1] != mesh.n_ghost_copies_g ||
            counts[2] + cuts + counts[4] != mesh.n_faces_g || counts[4] != mesh.n_bfaces_g ||
            cuts != mesh.n_cut_faces_g || counts[5] != mesh.n_labeled_bfaces_g)
            mpi::fatal("Global mesh cell/face/halo conservation failed");
        for (auto value : measures) if (!std::isfinite(value)) mpi::fatal("Global mesh metric sum is not finite");
        MeshStatistics result;
        result.owned_cells = counts[0]; result.ghost_copies = counts[1]; result.interior_faces = counts[2];
        result.cut_faces = cuts; result.boundary_faces = counts[4]; result.labelled_faces = counts[5];
        result.min_owned = extrema[0]; result.max_owned = maximum - extrema[1];
        result.min_ghosts = extrema[2]; result.max_ghosts = maximum - extrema[3];
        result.total_volume = measures[0];
        result.patch_face_counts.assign(counts.begin() + 6, counts.end());
        result.patch_flux_areas.assign(measures.begin() + 1, measures.end());
        return result;
    } catch (const std::exception& error) { mpi::fatal(error.what()); }
      catch (...) { mpi::fatal("Unknown exception during mesh statistics"); }
}

} // namespace krona::mesh
