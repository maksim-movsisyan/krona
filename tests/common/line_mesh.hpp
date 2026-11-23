#pragma once

#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>
#include <algorithm>
#include <cmath>

namespace krona::test {

/// @brief Builds a line of hexahedral cells with optional partitions, skew and unlabelled walls.
inline krona::mesh::host::MeshPart line_mesh(krona::mpi::CommView comm,int start,int owned,int total,bool unlabelled=false,double ly=1.,double lz=1.,bool skew=false)
{
    krona::mesh::host::MeshPart m;m.rank=comm.rank();m.nprocs=comm.size();m.n_own=owned;m.faces_grouped=true;
    for(int i=0;i<owned;++i)m.cell_gid.push_back(start+i);
    if(start>0)m.cell_gid.push_back(start-1);
    if(start+owned<total)m.cell_gid.push_back(start+owned);
    m.n_cells=static_cast<krona::LocalIndex>(m.cell_gid.size());
    for(auto id:m.cell_gid){m.cell_centroid_x.push_back(static_cast<krona::GeometryReal>(id)+krona::GeometryReal{0.5});
        m.cell_centroid_y.push_back(0);m.cell_centroid_z.push_back(0);m.cell_volume.push_back(static_cast<krona::GeometryReal>(ly*lz));}
    auto local=[&](int gid){return static_cast<krona::LocalIndex>(std::find(m.cell_gid.begin(),m.cell_gid.end(),gid)-m.cell_gid.begin());};
    auto face=[&](int owner,krona::LocalIndex neighbour,int patch,double x,double y,double z,double nx,double ny,double nz){
        m.face_owner.push_back(owner);m.face_neigh.push_back(neighbour);m.face_patch.push_back(unlabelled&&patch==2?-1:patch);
        m.face_centroid_x.push_back(static_cast<krona::GeometryReal>(x));m.face_centroid_y.push_back(static_cast<krona::GeometryReal>(y));m.face_centroid_z.push_back(static_cast<krona::GeometryReal>(z));
        m.face_normal_x.push_back(static_cast<krona::GeometryReal>(nx));m.face_normal_y.push_back(static_cast<krona::GeometryReal>(ny));m.face_normal_z.push_back(static_cast<krona::GeometryReal>(nz));m.face_area.push_back(static_cast<krona::GeometryReal>(nx!=0.?ly*lz:ny!=0.?lz:ly));
    };
    for(int i=0;i+1<owned;++i)face(i,i+1,-1,start+i+1.,0,0,1,0,0);

    // Keep owned-owned faces, partition interfaces and physical boundaries in separate ranges.
    m.n_internal_faces=static_cast<krona::LocalIndex>(m.face_owner.size());
    if(start>0)face(0,local(start-1),-1,start,0,0,-1,0,0);
    if(start+owned<total)face(owned-1,local(start+owned),-1,start+owned,0,0,1,0,0);
    m.n_inner_faces=static_cast<krona::LocalIndex>(m.face_owner.size());
    if(start==0&&owned>0)face(0,-1,0,0,0,0,-1,0,0);
    if(start+owned==total&&owned>0)face(owned-1,-1,1,total,0,0,1,0,0);
    for(int i=0;i<owned;++i){const double x=start+i+0.5;
        face(i,-1,2,x,-0.5*ly,0,0,-1,0);face(i,-1,2,x,0.5*ly,0,0,1,0);
        face(i,-1,2,x,0,-0.5*lz,0,0,-1);face(i,-1,2,x,0,0.5*lz,0,0,1);}

    // Transform face normals and areas consistently with the affine coordinate skew.
    if(skew){
        for(std::size_t i=0;i<m.cell_centroid_x.size();++i){m.cell_centroid_x[i]+=static_cast<krona::GeometryReal>(0.2)*m.cell_centroid_y[i];m.cell_centroid_y[i]+=static_cast<krona::GeometryReal>(0.3)*m.cell_centroid_z[i];}
        for(std::size_t f=0;f<m.face_owner.size();++f){
            m.face_centroid_x[f]+=static_cast<krona::GeometryReal>(0.2)*m.face_centroid_y[f];
            m.face_centroid_y[f]+=static_cast<krona::GeometryReal>(0.3)*m.face_centroid_z[f];
            const double nx=m.face_normal_x[f],ny=m.face_normal_y[f]-0.2*nx,nz=m.face_normal_z[f]-0.3*m.face_normal_y[f]+0.06*nx;
            const double length=std::sqrt(nx*nx+ny*ny+nz*nz);m.face_normal_x[f]=static_cast<krona::GeometryReal>(nx/length);
            m.face_normal_y[f]=static_cast<krona::GeometryReal>(ny/length);m.face_normal_z[f]=static_cast<krona::GeometryReal>(nz/length);
            m.face_area[f]*=static_cast<krona::GeometryReal>(length);
        }
    }
    m.n_faces=static_cast<krona::LocalIndex>(m.face_owner.size());
    m.n_cells_g=static_cast<std::uint64_t>(total);m.n_faces_g=static_cast<std::uint64_t>(total==0?0:5*total+1);
    m.n_bfaces_g=m.n_labeled_bfaces_g=static_cast<std::uint64_t>(total==0?0:4*total+2);
    if(unlabelled)m.n_labeled_bfaces_g=2;
    m.patches.resize(3);m.patches[0].name="left";m.patches[1].name="right";m.patches[2].name="sides";
    return m;
}

/// @brief Builds owned-cell face lists independently of the production connectivity builder.
inline krona::mesh::host::MeshAuxConnectivity line_incidence(const krona::mesh::host::MeshPart& m)
{
    krona::mesh::host::MeshAuxConnectivity c;c.active_mask=krona::mesh::AuxConnType::CellFaces;
    c.n_faces=m.n_faces;c.n_cells_own=m.n_own;c.n_nodes_own=m.n_nodes_own;c.cell_faces_offsets.push_back(0);
    for(krona::LocalIndex cell=0;cell<m.n_own;++cell){
        for(krona::LocalIndex f=0;f<m.n_faces;++f)
            if(m.face_owner[static_cast<std::size_t>(f)]==cell||m.face_neigh[static_cast<std::size_t>(f)]==cell)c.cell_faces.push_back(f);
        c.cell_faces_offsets.push_back(c.cell_faces.size());
    }return c;
}

} // namespace krona::test
