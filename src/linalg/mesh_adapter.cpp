#include <krona/linalg/mesh_adapter.hpp>

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace krona::linalg {
namespace {

class PrivateComm {
public:
    explicit PrivateComm(MPI_Comm source) { mpi::check(MPI_Comm_dup(source, &comm_), "MPI_Comm_dup(linalg numbering)"); }
    ~PrivateComm() { mpi::check(MPI_Comm_free(&comm_), "MPI_Comm_free(linalg numbering)"); }
    PrivateComm(const PrivateComm&) = delete;
    PrivateComm& operator=(const PrivateComm&) = delete;
    [[nodiscard]] MPI_Comm get() const noexcept { return comm_; }
private:
    MPI_Comm comm_ = MPI_COMM_NULL;
};

void validate(const mesh::MeshPart& m, int rank, int ranks)
{
    mpi::require(m.rank == rank && m.nprocs == ranks && m.n_own >= 0 && m.n_cells >= m.n_own,
                    "Linalg: mesh/communicator or cell counts mismatch");
    detail::local_count(static_cast<std::size_t>(m.n_cells));
    const auto peers = m.nb_ranks.size();
    detail::local_count(peers);
    mpi::require(m.send_offsets.size() == peers+1 && m.recv_offsets.size() == peers+1 &&
        m.send_offsets.front() == 0 && m.recv_offsets.front() == 0 &&
        m.send_offsets.back() == m.send_owned_local.size() && m.recv_offsets.back() == m.recv_ghost_local.size(),
        "Linalg: invalid halo offsets/extents");
    std::unordered_set<int> seen_peers;
    for (std::size_t peer = 0; peer < peers; ++peer) {
        mpi::require(m.nb_ranks[peer] >= 0 && m.nb_ranks[peer] < ranks && m.nb_ranks[peer] != rank &&
            seen_peers.insert(m.nb_ranks[peer]).second, "Linalg: invalid/duplicate halo peer");
        mpi::require(m.send_offsets[peer] <= m.send_offsets[peer+1] && m.recv_offsets[peer] <= m.recv_offsets[peer+1],
                        "Linalg: nonmonotone halo offsets");
        detail::local_count(m.send_offsets[peer+1]-m.send_offsets[peer]);
        detail::local_count(m.recv_offsets[peer+1]-m.recv_offsets[peer]);
    }
    for (const auto cell : m.send_owned_local)
        mpi::require(cell >= 0 && cell < m.n_own, "Linalg: halo send index is not owned");
    std::vector<bool> seen(static_cast<std::size_t>(m.n_cells-m.n_own), false);
    for (const auto cell : m.recv_ghost_local) {
        mpi::require(cell >= m.n_own && cell < m.n_cells, "Linalg: halo receive index is not a ghost");
        const auto slot = static_cast<std::size_t>(cell-m.n_own);
        mpi::require(!seen[slot], "Linalg: duplicate ghost receive index");
        seen[slot] = true;
    }
    mpi::require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
                    "Linalg: halo plan does not cover every ghost cell");
}
} // namespace

CellNumbering build_cell_numbering(const mesh::MeshPart& m, MPI_Comm comm)
{
    return mpi::guarded([&] {
        mpi::require(comm != MPI_COMM_NULL, "Linalg: null numbering communicator");
        int inter = 0;
        mpi::check(MPI_Comm_test_inter(comm, &inter), "MPI_Comm_test_inter(linalg numbering)");
        mpi::require(inter == 0, "Linalg: intercommunicators are unsupported");
        int rank = 0, ranks = 0;
        mpi::check(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank(linalg numbering)");
        mpi::check(MPI_Comm_size(comm, &ranks), "MPI_Comm_size(linalg numbering)");
        validate(m, rank, ranks);
        CellNumbering result;
        result.n_owned = static_cast<std::size_t>(m.n_own);
        const auto owned = static_cast<GlobalOrdinal>(m.n_own);
        mpi::check(MPI_Exscan(&owned, &result.owned_begin, 1, MPI_LONG_LONG_INT, MPI_SUM, comm), "MPI_Exscan(linalg numbering)");
        if (rank == 0) result.owned_begin = 0;
        mpi::check(MPI_Allreduce(&owned, &result.global_cells, 1, MPI_LONG_LONG_INT, MPI_SUM, comm), "MPI_Allreduce(linalg numbering)");
        result.cell_to_global.assign(static_cast<std::size_t>(m.n_cells), GlobalOrdinal{-1});
        for (std::size_t cell = 0; cell < result.n_owned; ++cell)
            result.cell_to_global[cell] = result.owned_begin+static_cast<GlobalOrdinal>(cell);

        PrivateComm private_comm(comm);
        std::vector<GlobalOrdinal> send(m.send_owned_local.size()), recv(m.recv_ghost_local.size());
        for (std::size_t i = 0; i < send.size(); ++i) send[i] = result.cell_to_global[static_cast<std::size_t>(m.send_owned_local[i])];
        mpi::require(m.nb_ranks.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()/2),
                        "Linalg: too many numbering requests");
        std::vector<MPI_Request> requests;
        requests.reserve(2*m.nb_ranks.size());
        for (std::size_t peer = 0; peer < m.nb_ranks.size(); ++peer) {
            const auto begin = m.recv_offsets[peer];
            const auto count = detail::local_count(m.recv_offsets[peer+1]-begin);
            if (count == 0) continue;
            MPI_Request request = MPI_REQUEST_NULL;
            mpi::check(MPI_Irecv(recv.data()+begin, count, MPI_LONG_LONG_INT, m.nb_ranks[peer], 0, private_comm.get(), &request),
                       "MPI_Irecv(linalg numbering)");
            requests.push_back(request);
        }
        for (std::size_t peer = 0; peer < m.nb_ranks.size(); ++peer) {
            const auto begin = m.send_offsets[peer];
            const auto count = detail::local_count(m.send_offsets[peer+1]-begin);
            if (count == 0) continue;
            MPI_Request request = MPI_REQUEST_NULL;
            mpi::check(MPI_Isend(send.data()+begin, count, MPI_LONG_LONG_INT, m.nb_ranks[peer], 0, private_comm.get(), &request),
                       "MPI_Isend(linalg numbering)");
            requests.push_back(request);
        }
        if (!requests.empty())
            mpi::check(MPI_Waitall(detail::local_count(requests.size()), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(linalg numbering)");
        for (std::size_t i = 0; i < recv.size(); ++i) {
            mpi::require(recv[i] >= 0 && recv[i] < result.global_cells, "Linalg: invalid received algebraic GID");
            result.cell_to_global[static_cast<std::size_t>(m.recv_ghost_local[i])] = recv[i];
        }
        return result;
    });
}

std::vector<GlobalOrdinal> CellNumbering::expanded_gids(int components) const
{
    mpi::require(components > 0 && global_cells >= 0 && global_cells <= std::numeric_limits<GlobalOrdinal>::max()/components,
                    "Linalg: invalid component count or global DOF overflow");
    mpi::require(cell_to_global.size() <= std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(components),
                    "Linalg: expanded local DOF size overflow");
    detail::local_count(cell_to_global.size()*static_cast<std::size_t>(components));
    std::vector<GlobalOrdinal> result(cell_to_global.size()*static_cast<std::size_t>(components));
    for (std::size_t cell = 0; cell < cell_to_global.size(); ++cell) for (int c = 0; c < components; ++c)
        result[cell*static_cast<std::size_t>(components)+static_cast<std::size_t>(c)] = cell_to_global[cell]*components+c;
    return result;
}

CsrInput make_face_csr(const mesh::MeshPart& m, const CellNumbering& numbering)
{
    return mpi::guarded([&] {
        mpi::require(m.n_own >= 0 && m.n_cells >= m.n_own && m.n_faces >= 0 &&
            numbering.n_owned == static_cast<std::size_t>(m.n_own) &&
            numbering.cell_to_global.size() == static_cast<std::size_t>(m.n_cells) &&
            m.face_owner.size() == static_cast<std::size_t>(m.n_faces) && m.face_neigh.size() == m.face_owner.size(),
            "Linalg: mesh graph/numbering mismatch");
        const auto rows = numbering.n_owned;
        CsrInput result;
        std::vector<std::size_t> counts(rows, 1);
        for (std::size_t face = 0; face < m.face_owner.size(); ++face) {
            const auto owner = m.face_owner[face], neighbour = m.face_neigh[face];
            mpi::require(owner >= 0 && owner < m.n_own && neighbour >= -1 && neighbour < m.n_cells && neighbour != owner,
                            "Linalg: invalid face endpoints");
            if (neighbour < 0) continue;
            ++counts[static_cast<std::size_t>(owner)];
            if (neighbour < m.n_own) ++counts[static_cast<std::size_t>(neighbour)];
        }
        result.row_offsets.resize(rows+1, 0);
        for (std::size_t row = 0; row < rows; ++row) {
            mpi::require(counts[row] <= std::numeric_limits<std::size_t>::max()-result.row_offsets[row], "Linalg: CSR size overflow");
            result.row_offsets[row+1] = result.row_offsets[row]+counts[row];
        }
        result.column_gids.resize(result.row_offsets.back());
        counts.assign(result.row_offsets.begin(), result.row_offsets.end()-1);
        for (std::size_t row = 0; row < rows; ++row) result.column_gids[counts[row]++] = numbering.cell_to_global[row];
        for (std::size_t face = 0; face < m.face_owner.size(); ++face) {
            const auto owner = m.face_owner[face], neighbour = m.face_neigh[face];
            if (neighbour < 0) continue;
            result.column_gids[counts[static_cast<std::size_t>(owner)]++] = numbering.cell_to_global[static_cast<std::size_t>(neighbour)];
            if (neighbour < m.n_own)
                result.column_gids[counts[static_cast<std::size_t>(neighbour)]++] = numbering.cell_to_global[static_cast<std::size_t>(owner)];
        }
        std::size_t old_begin = 0, write = 0;
        for (std::size_t row = 0; row < rows; ++row) {
            const auto old_end = result.row_offsets[row+1];
            auto begin = result.column_gids.begin()+static_cast<std::ptrdiff_t>(old_begin);
            auto end = result.column_gids.begin()+static_cast<std::ptrdiff_t>(old_end);
            std::sort(begin, end); end = std::unique(begin, end);
            result.row_offsets[row] = write;
            for (auto entry = begin; entry != end; ++entry) result.column_gids[write++] = *entry;
            old_begin = old_end;
        }
        result.row_offsets.back() = write;
        result.column_gids.resize(write);
        return result;
    });
}

} // namespace krona::linalg
