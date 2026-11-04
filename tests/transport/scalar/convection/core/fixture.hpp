#pragma once

// Included inside each convection test translation unit's anonymous namespace.
struct Fixture {
    mesh::host::MeshPart host;
    mesh::host::MeshAuxConnectivity connectivity;
    mesh::host::MeshAuxGeometry geometry;
    int first = 0, total = 0;
};

Fixture bar(mpi::CommView comm, bool duplicate = false, bool empty_last = false, bool all_empty = false, int boundary_copies = 1)
{
    Fixture f;
    auto& m = f.host;
    const int active = all_empty ? 0 : (empty_last && comm.size() > 1 ? comm.size()-1 : comm.size());
    const int count = comm.rank() < active ? 3 : 0;
    f.first = comm.rank()*3; f.total = active*3;
    m.rank = comm.rank(); m.nprocs = comm.size(); m.n_own = count;
    m.n_cells_g = static_cast<std::uint64_t>(f.total);
    m.faces_grouped = true;
    m.patches.resize(2); m.patches[0].name = "left"; m.patches[1].name = "right";
    m.n_bfaces_g = f.total == 0 ? 0 : static_cast<std::uint64_t>(2*boundary_copies); m.n_labeled_bfaces_g = m.n_bfaces_g;
    for (int i = 0; i < count; ++i) m.cell_centroid_x.push_back(static_cast<GeometryReal>(f.first+i)+GeometryReal{0.5});
    for (int peer : {comm.rank()-1, comm.rank()+1}) {
        if (count == 0 || peer < 0 || peer >= active) continue;
        m.nb_ranks.push_back(peer);
        m.send_owned_local.push_back(peer < comm.rank() ? 0 : count-1);
        m.recv_ghost_local.push_back(static_cast<LocalIndex>(m.cell_centroid_x.size()));
        m.cell_centroid_x.push_back(static_cast<GeometryReal>(peer < comm.rank() ? f.first-1 : f.first+count)+GeometryReal{0.5});
        m.send_offsets.push_back(m.send_owned_local.size()); m.recv_offsets.push_back(m.recv_ghost_local.size());
    }
    m.n_cells = static_cast<LocalIndex>(m.cell_centroid_x.size());
    m.cell_centroid_y.assign(m.cell_centroid_x.size(), 0); m.cell_centroid_z.assign(m.cell_centroid_x.size(), 0);
    m.cell_volume.assign(m.cell_centroid_x.size(), 1);
    const auto face = [&](LocalIndex owner, LocalIndex neighbour, GeometryReal x, GeometryReal normal, GeometryReal area, mesh::PatchId patch) {
        m.face_owner.push_back(owner); m.face_neigh.push_back(neighbour); m.face_patch.push_back(patch);
        m.face_centroid_x.push_back(x); m.face_centroid_y.push_back(0); m.face_centroid_z.push_back(0);
        m.face_normal_x.push_back(normal); m.face_normal_y.push_back(0); m.face_normal_z.push_back(0); m.face_area.push_back(area);
        const GeometryReal distance = neighbour < 0 ? GeometryReal{0.5} : GeometryReal{1};
        f.geometry.face_normal_dist.push_back(distance);
        f.geometry.face_area_over_normal_dist.push_back(area/distance);
        f.geometry.face_interp_weight.push_back(neighbour < 0 ? 0 : GeometryReal{0.5});
    };
    for (LocalIndex cell = 0; cell+1 < count; ++cell)
        for (int copy = 0; copy < (duplicate ? 2 : 1); ++copy)
            face(cell, cell+1, static_cast<GeometryReal>(f.first+cell+1), 1, duplicate ? GeometryReal{0.5} : GeometryReal{1}, -1);
    m.n_internal_faces = static_cast<LocalIndex>(m.face_owner.size());
    for (std::size_t p = 0; p < m.nb_ranks.size(); ++p) {
        const bool left = m.nb_ranks[p] < comm.rank();
        face(left ? 0 : count-1, m.recv_ghost_local[p], static_cast<GeometryReal>(left ? f.first : f.first+count), left ? -1 : 1, 1, -1);
    }
    m.n_inner_faces = static_cast<LocalIndex>(m.face_owner.size());
    for (int copy = 0; copy < boundary_copies; ++copy) {
        if (count != 0 && comm.rank() == 0) face(0, -1, 0, -1, GeometryReal{1}/static_cast<GeometryReal>(boundary_copies), 0);
    }
    for (int copy = 0; copy < boundary_copies; ++copy) {
        if (count != 0 && comm.rank() == active-1) face(count-1, -1, static_cast<GeometryReal>(f.total), 1, GeometryReal{1}/static_cast<GeometryReal>(boundary_copies), 1);
    }
    m.n_faces = static_cast<LocalIndex>(m.face_owner.size());
    auto& c = f.connectivity;
    c.active_mask = mesh::AuxConnType::CellFaces; c.n_faces = m.n_faces; c.n_cells_own = m.n_own;
    c.cell_faces_offsets = {0};
    for (LocalIndex cell = 0; cell < count; ++cell) {
        for (LocalIndex entry = 0; entry < m.n_faces; ++entry)
            if (m.face_owner[static_cast<std::size_t>(entry)] == cell || m.face_neigh[static_cast<std::size_t>(entry)] == cell)
                c.cell_faces.push_back(entry);
        c.cell_faces_offsets.push_back(c.cell_faces.size());
    }
    f.geometry.active_mask = mesh::AuxGeomType::FaceNormalDistance | mesh::AuxGeomType::FaceAreaOverNormalDistance |
                             mesh::AuxGeomType::FaceInterpolationWeights;
    f.geometry.n_faces = m.n_faces;
    f.geometry.n_cells = m.n_cells;
    return f;
}



