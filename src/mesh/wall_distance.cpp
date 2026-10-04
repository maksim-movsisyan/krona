#include <krona/mesh/aux_geometry.hpp>
#include <krona/mpi/collectives.hpp>

#include "aux_detail.hpp"

#include <algorithm>
#include <limits>

namespace krona::mesh {
namespace {

constexpr auto mpi_chunk = static_cast<std::size_t>(std::numeric_limits<int>::max());

/// @brief Counts MPI 3.1 messages needed for a potentially 64-bit halo group.
std::size_t chunk_count(std::size_t count)
{
    // Counts are in GeometryReal elements, not bytes.
    return count / mpi_chunk + static_cast<std::size_t>(count % mpi_chunk != 0);
}

/// @brief Copies owned distances to ghosts using existing matching halo order and reusable buffers.
void exchange_distances(const MeshPart& m, mpi::CommView comm, std::span<GeometryReal> distance,
                        std::vector<GeometryReal>& send, std::vector<GeometryReal>& receive,
                        std::vector<MPI_Request>& requests)
{
    const auto datatype = mpi::datatype<GeometryReal>();
    std::size_t request = 0;
    // Post receives first; multiple chunks from the same peer retain MPI message order.
    for (std::size_t p = 0; p < m.nb_ranks.size(); ++p) {
        for (auto i = m.recv_offsets[p]; i < m.recv_offsets[p + 1];) {
            const auto count = std::min(mpi_chunk, m.recv_offsets[p + 1] - i);
            mpi::check(MPI_Irecv(receive.data() + i, static_cast<int>(count), datatype,
                                 m.nb_ranks[p], 0, comm.native(), &requests[request++]), "MPI_Irecv(wall distance)");
            i += count;
        }
    }
    for (std::size_t i = 0; i < send.size(); ++i)
        send[i] = distance[static_cast<std::size_t>(m.send_owned_local[i])];
    for (std::size_t p = 0; p < m.nb_ranks.size(); ++p) {
        for (auto i = m.send_offsets[p]; i < m.send_offsets[p + 1];) {
            const auto count = std::min(mpi_chunk, m.send_offsets[p + 1] - i);
            mpi::check(MPI_Isend(send.data() + i, static_cast<int>(count), datatype,
                                 m.nb_ranks[p], 0, comm.native(), &requests[request++]), "MPI_Isend(wall distance)");
            i += count;
        }
    }
    // Waitall also has an int count; waiting does not allocate new communication storage.
    for (std::size_t i = 0; i < request;) {
        const auto count = std::min(mpi_chunk, request - i);
        mpi::check(MPI_Waitall(static_cast<int>(count), requests.data() + i, MPI_STATUSES_IGNORE),
                   "MPI_Waitall(wall distance)");
        i += count;
    }
    for (std::size_t i = 0; i < receive.size(); ++i)
        distance[static_cast<std::size_t>(m.recv_ghost_local[i])] = receive[i];
}

/// @brief Performs one owned-cell relaxation pass and measures changes in stored precision.
double relax(std::span<GeometryReal> distance, std::span<const std::size_t> offsets,
             std::span<const LocalIndex> neighbours, std::span<const GeometryReal> edge_distance,
             bool reverse)
{
    double change = 0;
    const auto owned = offsets.size() - 1;
    for (std::size_t i = 0; i < owned; ++i) {
        const auto c = reverse ? owned - 1 - i : i;
        const double old = distance[c];
        double best = old;
        for (auto j = offsets[c]; j < offsets[c + 1]; ++j)
            best = std::min(best, static_cast<double>(distance[static_cast<std::size_t>(neighbours[j])]) + edge_distance[j]);
        if (!(best < old)) continue;
        const auto updated = aux_detail::positive(best);
        // FP32 rounding can erase a double improvement; do not report a fictitious update.
        if (!(updated < distance[c])) continue;
        distance[c] = updated;
        const double relative = std::isinf(old) ? 1 : (old - static_cast<double>(updated)) / old;
        change = std::max(change, relative);
    }
    return change;
}

} // namespace

/// @brief Recomputes graph-path wall distance with collective convergence and a final halo refresh.
WallDistanceStats MeshAuxGeometry::add_wall_distance(const MeshPart& m,
                                                     MeshAuxConnectivity& connectivity,
                                                     mpi::CommView comm,
                                                     std::span<const std::uint8_t> wall_patch,
                                                     WallDistanceOptions options)
{
    if (!mpi::is_active() || comm.is_null()) mpi::fatal("Wall distance requires an active MPI communicator");
    mpi::require_main_thread();
    if (comm.rank() != m.rank || comm.size() != m.nprocs || wall_patch.size() != m.patches.size())
        mpi::fatal("Wall distance communicator or patch mask does not match MeshPart");
    if (options.max_sweeps <= 0 || !std::isfinite(options.relative_tolerance) ||
        options.relative_tolerance < 0 || options.relative_tolerance >= 1 ||
        (options.stencil != WallDistStencil::FaceNeighbors && options.stencil != WallDistStencil::NodeNeighbors))
        mpi::fatal("Invalid wall distance options");
    aux_detail::require_geometry(m);
    active_mask = static_cast<AuxGeomType>(static_cast<std::uint32_t>(active_mask) &
                                          ~static_cast<std::uint32_t>(AuxGeomType::WallDistance));
    cell_wall_dist.assign(static_cast<std::size_t>(m.n_cells), std::numeric_limits<GeometryReal>::infinity());

    // Physical wall faces seed the field; unlabelled faces are not implicitly treated as walls.
    for (auto f = static_cast<std::size_t>(m.n_inner_faces); f < static_cast<std::size_t>(m.n_faces); ++f) {
        const auto patch = m.face_patch[f];
        if (patch < 0 || wall_patch[static_cast<std::size_t>(patch)] == 0) continue;
        const auto owner = static_cast<std::size_t>(m.face_owner[f]);
        const auto seed = aux_detail::positive(aux_detail::length(
            aux_detail::subtract(aux_detail::face_center(m, f), aux_detail::cell_center(m, owner))));
        cell_wall_dist[owner] = std::min(cell_wall_dist[owner], seed);
    }

    const bool node_stencil = options.stencil == WallDistStencil::NodeNeighbors;
    connectivity.add_connectivity(m, node_stencil ? AuxConnType::CellCellsByNode : AuxConnType::CellCellsByFace);
    const auto offsets = std::span<const std::size_t>{node_stencil ? connectivity.cell_cells_node_offsets
                                                                 : connectivity.cell_cells_face_offsets};
    const auto neighbours = std::span<const LocalIndex>{node_stencil ? connectivity.cell_cells_node
                                                                  : connectivity.cell_cells_face};
    std::vector<GeometryReal> edge_distance(neighbours.size());
    // Centre distances are invariant during relaxation, so compute them once.
    for (std::size_t c = 0; c < static_cast<std::size_t>(m.n_own); ++c) {
        const auto origin = aux_detail::cell_center(m, c);
        for (auto j = offsets[c]; j < offsets[c + 1]; ++j)
            edge_distance[j] = aux_detail::positive(aux_detail::length(aux_detail::subtract(
                aux_detail::cell_center(m, static_cast<std::size_t>(neighbours[j])), origin)));
    }

    // Isolate halo tags from the solver and other users of the borrowed communicator.
    auto halo_comm = mpi::Comm::duplicate(comm);
    std::vector<GeometryReal> send(m.send_owned_local.size()), receive(m.recv_ghost_local.size());
    std::size_t request_count = 0;
    for (std::size_t p = 0; p < m.nb_ranks.size(); ++p) {
        request_count += chunk_count(m.send_offsets[p + 1] - m.send_offsets[p]);
        request_count += chunk_count(m.recv_offsets[p + 1] - m.recv_offsets[p]);
    }
    std::vector<MPI_Request> requests(request_count);
    WallDistanceStats stats;
    for (int sweep = 0; sweep < options.max_sweeps; ++sweep) {
        exchange_distances(m, halo_comm.view(), cell_wall_dist, send, receive, requests);
        double change = relax(cell_wall_dist, offsets, neighbours, edge_distance, false);
        change = std::max(change, relax(cell_wall_dist, offsets, neighbours, edge_distance, true));
        // Every rank participates, including empty ranks and ranks without halo peers.
        stats.relative_change = comm.size() == 1 ? change : mpi::allreduce<mpi::Reduction::max>(comm, change);
        stats.sweeps = sweep + 1;
        if (stats.relative_change <= options.relative_tolerance) {
            stats.converged = true;
            break;
        }
    }
    // Owned values changed after the last exchange, even when the sweep limit was reached.
    exchange_distances(m, halo_comm.view(), cell_wall_dist, send, receive, requests);
    std::uint64_t unreachable = 0;
    for (std::size_t c = 0; c < static_cast<std::size_t>(m.n_own); ++c)
        if (std::isinf(cell_wall_dist[c])) ++unreachable;
    stats.unreachable_owned = comm.size() == 1 ? unreachable : mpi::allreduce<mpi::Reduction::sum>(comm, unreachable);
    active_mask = active_mask | AuxGeomType::WallDistance;
    return stats;
}

} // namespace krona::mesh
