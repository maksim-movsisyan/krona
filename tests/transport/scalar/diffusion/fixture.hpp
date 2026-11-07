#pragma once

// Shared analytical fixtures; included inside the tests' anonymous namespace.
struct Fixture {
    mesh::host::MeshPart host;
    mesh::host::MeshAuxConnectivity connectivity;
    mesh::host::MeshAuxGeometry geometry;
    int first = 0, total = 0;
};

Fixture bar(mpi::CommView comm, bool duplicate = false, bool empty_last = false, bool all_empty = false)
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
    m.n_bfaces_g = f.total == 0 ? 0 : 2; m.n_labeled_bfaces_g = m.n_bfaces_g;
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
    if (count != 0 && comm.rank() == 0) face(0, -1, 0, -1, 1, 0);
    if (count != 0 && comm.rank() == active-1) face(count-1, -1, static_cast<GeometryReal>(f.total), 1, 1, 1);
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

// Stretched/sheared parallelepipeds with independently computed face metrics.
Fixture skew_grid(mpi::CommView comm, bool skew = true)
{
    Fixture f; 
    auto& m = f.host;
    constexpr int local_x = 2, ny = 2, nz = 2;
    const int first = local_x*comm.rank(), nx = local_x*comm.size();
    const double sy = skew ? 0.27 : 0.0, sz = skew ? 0.13 : 0.0, yz = skew ? 0.19 : 0.0;
    const double curvature = skew ? 0.09 : 0.0;
    const auto edge = [](int x) { return double(x)+0.08*double(x*x); };
    const auto shift = [&](int x) { return curvature*double(x*x); };
    const auto point = [&](double x, double y, double z) { return std::array{x+sy*y+sz*z,y+yz*z,z}; };
    const auto local = [](int x, int y, int z) { return LocalIndex((x*ny+y)*nz+z); };
    m.rank=comm.rank();m.nprocs=comm.size();m.n_own=local_x*ny*nz;m.faces_grouped=true;
    m.n_cells_g=static_cast<std::uint64_t>(nx*ny*nz);
    m.patches.resize(6);
    for(int p=0;p<6;++p)m.patches[static_cast<std::size_t>(p)].name="side"+std::to_string(p);
    struct Cell {int x,y,z;};
    std::vector<Cell> cells;
    for(int x=first;x<first+local_x;++x)for(int y=0;y<ny;++y)for(int z=0;z<nz;++z)cells.push_back({x,y,z});
    for(int peer : {comm.rank()-1,comm.rank()+1}) {
        if(peer<0||peer>=comm.size())continue;
        m.nb_ranks.push_back(peer);
        for(int y=0;y<ny;++y)for(int z=0;z<nz;++z) {
            m.send_owned_local.push_back(local(peer<comm.rank()?0:local_x-1,y,z));
            m.recv_ghost_local.push_back(static_cast<LocalIndex>(cells.size()));
            cells.push_back({peer<comm.rank()?first-1:first+local_x,y,z});
        }
        m.send_offsets.push_back(m.send_owned_local.size());m.recv_offsets.push_back(m.recv_ghost_local.size());
    }
    m.n_cells=static_cast<LocalIndex>(cells.size());
    for(const auto c:cells) {
        const auto x=point(0.5*(edge(c.x)+edge(c.x+1)),c.y+0.5+0.5*(shift(c.x)+shift(c.x+1)),c.z+0.5);
        m.cell_centroid_x.push_back(static_cast<GeometryReal>(x[0]));
        m.cell_centroid_y.push_back(static_cast<GeometryReal>(x[1]));
        m.cell_centroid_z.push_back(static_cast<GeometryReal>(x[2]));
        m.cell_volume.push_back(static_cast<GeometryReal>(edge(c.x+1)-edge(c.x)));
    }
    struct Face {LocalIndex owner,neighbour;std::array<double,3> x,s;int patch;};
    std::array<std::vector<Face>,3> groups;
    for(LocalIndex i=0;i<m.n_own;++i) {
        const auto cell=cells[static_cast<std::size_t>(i)];
        for(int axis=0;axis<3;++axis)for(int side : {-1,1}) {
            Cell other=cell;
            (axis==0?other.x:axis==1?other.y:other.z)+=side;
            const bool exterior=other.x<0||other.x>=nx||other.y<0||other.y>=ny||other.z<0||other.z>=nz;
            LocalIndex j=-1;
            if(!exterior) {
                for(std::size_t k=0;k<cells.size();++k)
                    if(cells[k].x==other.x&&cells[k].y==other.y&&cells[k].z==other.z){j=static_cast<LocalIndex>(k);break;}
                check(j>=0,"skew fixture is missing a real ghost");
                if(j<m.n_own&&i>j)continue;
            }
            const double dx=edge(cell.x+1)-edge(cell.x);
            const double dy=shift(cell.x+1)-shift(cell.x);
            const double face_shift=axis==0?shift(cell.x+(side>0)):0.5*(shift(cell.x)+shift(cell.x+1));
            auto x=point(axis==0?edge(cell.x+(side>0)):0.5*(edge(cell.x)+edge(cell.x+1)),
                         cell.y+(axis==1?(side>0?1.0:0.0):0.5)+face_shift,
                         cell.z+(axis==2?(side>0?1.0:0.0):0.5));
            const std::array<double,3> area=axis==0?std::array{1.0,-sy,sy*yz-sz}:
                axis==1?std::array{-dy,dx+sy*dy,sz*dy-yz*(dx+sy*dy)}:std::array{0.0,0.0,dx};
            auto s=area;for(auto& value:s)value*=side;
            groups[exterior?2:j<m.n_own?0:1].push_back({i,j,x,s,exterior?2*axis+(side>0):-1});
        }
    }
    std::sort(groups[2].begin(),groups[2].end(),[](const auto& a,const auto& b){return a.patch<b.patch;});
    m.n_internal_faces=static_cast<LocalIndex>(groups[0].size());
    m.n_inner_faces=m.n_internal_faces+static_cast<LocalIndex>(groups[1].size());
    for(const auto& group:groups)for(const auto& face:group) {
        const double area=std::sqrt(face.s[0]*face.s[0]+face.s[1]*face.s[1]+face.s[2]*face.s[2]);
        const std::array n{face.s[0]/area,face.s[1]/area,face.s[2]/area};
        const auto i=static_cast<std::size_t>(face.owner),j=static_cast<std::size_t>(face.neighbour<0?0:face.neighbour);
        const std::array p{double(m.cell_centroid_x[i]),double(m.cell_centroid_y[i]),double(m.cell_centroid_z[i])};
        const std::array q=face.neighbour<0?face.x:std::array{double(m.cell_centroid_x[j]),double(m.cell_centroid_y[j]),double(m.cell_centroid_z[j])};
        const std::array d{q[0]-p[0],q[1]-p[1],q[2]-p[2]};
        const double h=d[0]*n[0]+d[1]*n[1]+d[2]*n[2];
        const double left=(face.x[0]-p[0])*n[0]+(face.x[1]-p[1])*n[1]+(face.x[2]-p[2])*n[2];
        m.face_owner.push_back(face.owner);m.face_neigh.push_back(face.neighbour);m.face_patch.push_back(face.patch);
        m.face_centroid_x.push_back(static_cast<GeometryReal>(face.x[0]));m.face_centroid_y.push_back(static_cast<GeometryReal>(face.x[1]));m.face_centroid_z.push_back(static_cast<GeometryReal>(face.x[2]));
        m.face_normal_x.push_back(static_cast<GeometryReal>(n[0]));m.face_normal_y.push_back(static_cast<GeometryReal>(n[1]));m.face_normal_z.push_back(static_cast<GeometryReal>(n[2]));
        m.face_area.push_back(static_cast<GeometryReal>(area));
        f.geometry.face_normal_dist.push_back(static_cast<GeometryReal>(h));
        f.geometry.face_area_over_normal_dist.push_back(static_cast<GeometryReal>(area/h));
        f.geometry.face_interp_weight.push_back(static_cast<GeometryReal>(face.neighbour<0?0:(h-left)/h));
        f.geometry.face_non_ortho_x.push_back(static_cast<GeometryReal>(area*(n[0]-d[0]/h)));
        f.geometry.face_non_ortho_y.push_back(static_cast<GeometryReal>(area*(n[1]-d[1]/h)));
        f.geometry.face_non_ortho_z.push_back(static_cast<GeometryReal>(area*(n[2]-d[2]/h)));
    }
    m.n_faces=static_cast<LocalIndex>(m.face_owner.size());
    m.n_bfaces_g=static_cast<std::uint64_t>(2*(ny*nz+nx*nz+nx*ny));m.n_labeled_bfaces_g=m.n_bfaces_g;
    f.connectivity.active_mask=mesh::AuxConnType::CellFaces;f.connectivity.n_faces=m.n_faces;f.connectivity.n_cells_own=m.n_own;
    f.connectivity.cell_faces_offsets={0};
    for(LocalIndex i=0;i<m.n_own;++i) {
        for(LocalIndex face=0;face<m.n_faces;++face)
            if(m.face_owner[static_cast<std::size_t>(face)]==i||m.face_neigh[static_cast<std::size_t>(face)]==i)
                f.connectivity.cell_faces.push_back(face);
        f.connectivity.cell_faces_offsets.push_back(f.connectivity.cell_faces.size());
    }
    f.geometry.n_faces=m.n_faces;
    f.geometry.active_mask=mesh::AuxGeomType::FaceNormalDistance|mesh::AuxGeomType::FaceAreaOverNormalDistance|
        mesh::AuxGeomType::FaceInterpolationWeights|mesh::AuxGeomType::FaceNonOrthogonalityVector;
    return f;
}

