#pragma once

#include <krona/mesh/localmesh.hpp>
#include <krona/mesh/element.hpp>

#include <cmath>

namespace krona::test {

/// @brief Creates a synthetic rank-0 snapshot: two owned tetrahedra and one ghost.
/// Rank metadata describes two partitions; tests do not perform communication.
inline mesh::MeshPart make_mesh()
{
    using mesh::CellType;
    mesh::MeshPart m;
    m.rank = 0;
    m.nprocs = 2; 
    m.n_cells_g = 3;
    m.n_nodes_g = 6;
    m.n_faces_g = 10;
    m.n_bfaces_g = m.n_labeled_bfaces_g = 8;
    m.n_cut_faces_g = 1;
    m.n_ghost_copies_g = 2;
    m.bbox_lo = {0, 0, -1};
    m.bbox_hi = {1, 1, 1};

    m.n_own = 2;
    m.n_cells = 3;
    m.cell_type = {CellType::TET, CellType::TET, CellType::TET};
    m.cell_gid = {0, 1, 2};
    m.cell_donor = {-1, -1, 1};
    m.cell_nodes_offsets = {0, 4, 8, 12};
    m.cell_nodes = {0, 1, 2, 3, 0, 2, 1, 4, 1, 2, 3, 5};

    m.n_nodes_own = 5;
    m.n_nodes = 6;
    m.node_gid = {0, 1, 2, 3, 4, 5};
    m.node_x = {0, 1, 0, 0, 0, 1};
    m.node_y = {0, 0, 1, 0, 0, 1};
    m.node_z = {0, 0, 0, 1, -1, 1};
    m.cell_centroid_x = {GeometryReal{0.25}, GeometryReal{0.25}, GeometryReal{0.5}};
    m.cell_centroid_y = m.cell_centroid_x;
    m.cell_centroid_z = {GeometryReal{0.25}, GeometryReal{-0.25}, GeometryReal{0.5}};
    m.cell_volume = {GeometryReal{1} / 6, GeometryReal{1} / 6, GeometryReal{1} / 3};

    m.n_faces = 7;
    m.n_internal_faces = 1;
    m.n_inner_faces = 2;
    m.faces_grouped = true;
    m.face_owner = {0, 0, 0, 0, 1, 1, 1};
    m.face_neigh = {1, 2, -1, -1, -1, -1, -1};
    m.face_type.assign(7, CellType::TRI);
    m.face_local_index = {0, 2, 1, 3, 1, 2, 3};
    m.face_nodes_offsets = {0, 3, 6, 9, 12, 15, 18, 21};
    m.face_nodes = {0, 2, 1, 1, 2, 3, 0, 1, 3, 2, 0, 3,
                    0, 2, 4, 2, 1, 4, 1, 0, 4};
    m.face_patch = {-1, -1, 0, 0, 0, 0, 0};
    m.patches.push_back({"wall", "", ""});
    m.patch_face_offsets = {0, 5};
    m.patch_faces = {2, 3, 4, 5, 6};

    // Metrics follow the oriented triangular loops; no production builder is called.
    for (std::size_t f = 0; f < 7; ++f) {
        const auto a = static_cast<std::size_t>(m.face_nodes[3 * f]);
        const auto b = static_cast<std::size_t>(m.face_nodes[3 * f + 1]);
        const auto c = static_cast<std::size_t>(m.face_nodes[3 * f + 2]);
        const auto ux = m.node_x[b] - m.node_x[a], uy = m.node_y[b] - m.node_y[a], uz = m.node_z[b] - m.node_z[a];
        const auto vx = m.node_x[c] - m.node_x[a], vy = m.node_y[c] - m.node_y[a], vz = m.node_z[c] - m.node_z[a];
        const auto nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const auto length = std::sqrt(nx * nx + ny * ny + nz * nz);
        m.face_centroid_x.push_back((m.node_x[a] + m.node_x[b] + m.node_x[c]) / 3);
        m.face_centroid_y.push_back((m.node_y[a] + m.node_y[b] + m.node_y[c]) / 3);
        m.face_centroid_z.push_back((m.node_z[a] + m.node_z[b] + m.node_z[c]) / 3);
        m.face_normal_x.push_back(nx / length);
        m.face_normal_y.push_back(ny / length);
        m.face_normal_z.push_back(nz / length);
        m.face_area.push_back(length / 2);
    }

    m.nb_ranks = {1};
    m.send_offsets = {0, 1};
    m.recv_offsets = {0, 1};
    m.send_owned_local = {0};
    m.recv_ghost_local = {2};
    return m;
}

} // namespace krona::test
