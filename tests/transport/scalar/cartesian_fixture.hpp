#pragma once

#include <krona/mesh/localmesh.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <vector>

namespace krona::test::diffusion {

/// @brief Orthogonal unit cube, partitioned into x slabs; deliberately reversed ghost storage.
inline mesh::MeshPart cartesian(int nx, int ny, int nz, MPI_Comm comm)
{
    mesh::MeshPart m;
    mpi::check(MPI_Comm_rank(comm, &m.rank), "MPI_Comm_rank(test mesh)");
    mpi::check(MPI_Comm_size(comm, &m.nprocs), "MPI_Comm_size(test mesh)");
    const std::array<int, 3> dimensions{nx, ny, nz};
    const std::array<double, 3> spacing{1.0/nx, 1.0/ny, 1.0/nz};
    std::vector<int> cuts(static_cast<std::size_t>(m.nprocs)+1, 0);
    for (int r = 0; r < m.nprocs; ++r)
        cuts[static_cast<std::size_t>(r)+1] = cuts[static_cast<std::size_t>(r)] + nx/m.nprocs + (r < nx%m.nprocs ? 1 : 0);
    const auto owner_rank = [&](int gid) {
        const int x = gid/(ny*nz);
        return static_cast<int>(std::upper_bound(cuts.begin(), cuts.end(), x)-cuts.begin())-1;
    };
    const auto coordinates = [&](int gid) { return std::array<int, 3>{gid/(ny*nz), (gid/nz)%ny, gid%nz}; };
    const auto global_id = [&](std::array<int, 3> p) { return (p[0]*ny+p[1])*nz+p[2]; };
    const int first = cuts[static_cast<std::size_t>(m.rank)]*ny*nz;
    const int last = cuts[static_cast<std::size_t>(m.rank)+1]*ny*nz;
    const auto owned = [&](int gid) { return gid >= first && gid < last; };

    struct Face {
        int owner, neighbour, patch;
        std::array<double, 3> center{}, normal{};
        double area = 0;
    };
    std::vector<Face> faces;
    std::set<int> ghost_ids;
    std::map<int, std::set<int>> send, recv;
    for (int gid = 0; gid < nx*ny*nz; ++gid) {
        const auto p = coordinates(gid);
        std::array<double, 3> center{};
        for (int d = 0; d < 3; ++d) center[static_cast<std::size_t>(d)] =
            (static_cast<double>(p[static_cast<std::size_t>(d)])+0.5)*spacing[static_cast<std::size_t>(d)];
        for (int d = 0; d < 3; ++d) {
            const auto u = static_cast<std::size_t>(d);
            const double area = spacing[(u+1)%3]*spacing[(u+2)%3];
            if (p[u]+1 < dimensions[u]) {
                auto next = p; ++next[u];
                const int other = global_id(next);
                if (owned(gid) || owned(other)) {
                    Face f{owned(gid) ? gid : other, owned(gid) ? other : gid, -1};
                    f.center = center; f.center[u] += 0.5*spacing[u];
                    f.normal[u] = owned(gid) ? 1.0 : -1.0; f.area = area;
                    faces.push_back(f);
                    if (!owned(f.neighbour)) {
                        ghost_ids.insert(f.neighbour);
                        const int peer = owner_rank(f.neighbour);
                        send[peer].insert(f.owner); recv[peer].insert(f.neighbour);
                    }
                }
            }
            if (!owned(gid)) continue;
            if (p[u] == 0) {
                Face f{gid, -1, 2*d};
                f.center = center; f.center[u] = 0; f.normal[u] = -1; f.area = area;
                faces.push_back(f);
            }
            if (p[u]+1 == dimensions[u]) {
                Face f{gid, -1, 2*d+1};
                f.center = center; f.center[u] = 1; f.normal[u] = 1; f.area = area;
                faces.push_back(f);
            }
        }
    }
    m.n_own = static_cast<LocalIndex>(last-first);
    std::vector<int> local_ids;
    for (int gid = first; gid < last; ++gid) local_ids.push_back(gid);
    for (auto it = ghost_ids.rbegin(); it != ghost_ids.rend(); ++it) local_ids.push_back(*it);
    m.n_cells = static_cast<LocalIndex>(local_ids.size());
    std::map<int, LocalIndex> local;
    for (std::size_t i = 0; i < local_ids.size(); ++i) {
        const int gid = local_ids[i]; local[gid] = static_cast<LocalIndex>(i);
        const auto p = coordinates(gid);
        m.cell_gid.push_back(static_cast<GlobalIndex>(1000009+17*gid)); // Not algebraic GIDs.
        m.cell_centroid_x.push_back((static_cast<double>(p[0])+0.5)*spacing[0]);
        m.cell_centroid_y.push_back((static_cast<double>(p[1])+0.5)*spacing[1]);
        m.cell_centroid_z.push_back((static_cast<double>(p[2])+0.5)*spacing[2]);
        m.cell_volume.push_back(spacing[0]*spacing[1]*spacing[2]);
    }
    const auto group = [&](const Face& f) { return f.patch >= 0 ? 2+f.patch : (owned(f.neighbour) ? 0 : 1); };
    std::stable_sort(faces.begin(), faces.end(), [&](const Face& a, const Face& b) { return group(a) < group(b); });
    for (const auto& f : faces) {
        m.face_owner.push_back(local.at(f.owner));
        m.face_neigh.push_back(f.neighbour < 0 ? LocalIndex{-1} : local.at(f.neighbour));
        m.face_patch.push_back(f.patch);
        m.face_centroid_x.push_back(f.center[0]); m.face_centroid_y.push_back(f.center[1]); m.face_centroid_z.push_back(f.center[2]);
        m.face_normal_x.push_back(f.normal[0]); m.face_normal_y.push_back(f.normal[1]); m.face_normal_z.push_back(f.normal[2]);
        m.face_area.push_back(f.area);
        if (group(f) == 0) ++m.n_internal_faces;
        if (f.patch < 0) ++m.n_inner_faces;
    }
    m.n_faces = static_cast<LocalIndex>(faces.size()); m.faces_grouped = true;
    m.patches = {{"xmin", "test", "BCUserDefined"}, {"xmax", "test", "BCUserDefined"},
        {"ymin", "test", "BCUserDefined"}, {"ymax", "test", "BCUserDefined"},
        {"zmin", "test", "BCUserDefined"}, {"zmax", "test", "BCUserDefined"}};
    m.n_cells_g = static_cast<std::uint64_t>(nx*ny*nz);
    m.n_bfaces_g = static_cast<std::uint64_t>(2*(ny*nz+nx*nz+nx*ny));
    m.n_labeled_bfaces_g = m.n_bfaces_g;
    for (const auto& [peer, outgoing] : send) {
        m.nb_ranks.push_back(peer);
        for (const int gid : outgoing) m.send_owned_local.push_back(local.at(gid));
        for (const int gid : recv.at(peer)) m.recv_ghost_local.push_back(local.at(gid));
        m.send_offsets.push_back(m.send_owned_local.size()); m.recv_offsets.push_back(m.recv_ghost_local.size());
    }
    return m;
}

/// @brief Splits one interior/cut face into two equal-area faces with the same endpoints.
inline void split_face(mesh::MeshPart& m)
{
    if (m.n_inner_faces == 0) return;
    m.face_area[0] *= 0.5;
    const auto duplicate = [](auto& v) { const auto value = v[0]; v.insert(v.begin(), value); };
    duplicate(m.face_owner); duplicate(m.face_neigh); duplicate(m.face_patch);
    duplicate(m.face_centroid_x); duplicate(m.face_centroid_y); duplicate(m.face_centroid_z);
    duplicate(m.face_normal_x); duplicate(m.face_normal_y); duplicate(m.face_normal_z); duplicate(m.face_area);
    if (m.n_internal_faces > 0) ++m.n_internal_faces;
    ++m.n_inner_faces; ++m.n_faces;
}

} // namespace krona::test::diffusion
