#pragma once

// Included by test_diffusion.cpp inside its anonymous namespace; uses the shared test fixture types.
// A stretched grid has column-dependent shear and skew parallelepipeds. Metrics are independent
// analytical geometry, so an affine solution has zero integrated divergence in every owned cell.
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

KOKKOS_INLINE_FUNCTION Real affine_value(Real x,Real y,Real z)
{return Real{2}+Real(0.7)*x-Real(1.1)*y+Real(0.6)*z;}

struct ExactGradient {
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t) const
    {return Kokkos::Array<Real,3>{Real(0.7),Real(-1.1),Real(0.6)};}
};

struct AffineValue {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c,Real increment=0)const
    {return bc::FixedValue<Real>{affine_value(c.x,c.y,c.z)}(c,increment);}
};
struct AffineRobin {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c,Real increment=0)const
    {
        const Real normal=Real(0.7)*c.nx-Real(1.1)*c.ny+Real(0.6)*c.nz;
        const Real value=affine_value(c.x,c.y,c.z)+normal;
        const Real denominator=Real{1}+c.distance;
        return {Real{1}/denominator,(c.distance*value+increment)/denominator,
            -Real{1}/denominator,(value-increment)/denominator};
    }
};
struct ChangingSlope {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c,Real increment=0)const
    {return {0,0,-(Real{1}+increment*increment)/c.distance,0};}
};
using SkewBoundary=bc::Manager<Exec,Real,AffineValue,AffineRobin,ChangingSlope>;

void configure_affine(SkewBoundary& boundary,const Fixture& f,bool mixed=false)
{
    for(int p=0;p<6;++p)boundary.set(p,AffineValue{});
    if(!mixed)return;
    for(int p : {0,1})for(LocalIndex face=f.host.n_inner_faces;face<f.host.n_faces;++face) {
        const auto k=static_cast<std::size_t>(face);
        if(f.host.face_patch[k]!=p)continue;
        const Real gradient=Real(0.7)*f.host.face_normal_x[k]-Real(1.1)*f.host.face_normal_y[k]+Real(0.6)*f.host.face_normal_z[k];
        if(p==0)boundary.set(p,bc::NormalGradient<Real>{gradient});
        else boundary.set(p,bc::FixedFlux<Real>{-Real{2}*gradient});
        break;
    }
    boundary.set(2,AffineRobin{});
}

template<class Storage=double,class Layout=Kokkos::LayoutLeft>
void corrected_case(const Exec& exec,bool mixed=false,bool skew=true)
{
    const auto comm=mpi::CommView::world();const auto f=skew_grid(comm,skew);
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    SkewBoundary boundary(f.host,a.mesh.view(),exec);configure_affine(boundary,f,mixed);
    fields::FieldStorage<Storage,2,Exec,Layout> field("affine",boundary.required_field_size(),fields::FieldLocation::Custom,exec);
    fields::FieldStorage<Storage,3,Exec,Layout> gradients("gradients",static_cast<std::size_t>(f.host.n_cells),fields::FieldLocation::Custom,exec);
    field.fill(exec,Storage{-33});gradients.fill(exec,std::numeric_limits<Storage>::quiet_NaN());
    const auto phi=field.view().component(1);const auto m=a.mesh.view();
    using Policy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<LocalIndex>>;
    Kokkos::parallel_for("test/affine_owned",Policy(exec,0,f.host.n_own),KOKKOS_LAMBDA(LocalIndex i){
        phi(i)=static_cast<Storage>(affine_value(m.geometry.cell_centroid_x(i),m.geometry.cell_centroid_y(i),m.geometry.cell_centroid_z(i)));
    });
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    halo::Exchange<Storage,Exec> field_halo(m,a.mesh.halo_plan(),comm,31,exec);
    field_halo.add_component(phi);field_halo.prepare();field_halo.exchange();
    boundary.apply_component_async(phi,law,td::NonOrthogonalCorrection{ExactGradient{}});
    numerics::gradient::GradientOperator<Exec> gradient(m,a.connectivity.view(),a.geometry.view(),{},exec);
    gradient.apply_component(phi,gradients.view().component(0),gradients.view().component(1),gradients.view().component(2));
    halo::Exchange<Storage,Exec> gradient_halo(m,a.mesh.halo_plan(),comm,32,exec);
    gradient_halo.add_field(gradients.view());gradient_halo.prepare();gradient_halo.exchange();
    const auto g=gradients.create_host_buffer();Kokkos::deep_copy(exec,g,gradients.view().kokkos_view());exec.fence("test gradients ready");
    const double tolerance=sizeof(Storage) == sizeof(float)?3e-5:2e-11;
    for(LocalIndex i=0;i<f.host.n_cells;++i) {
        test::near(static_cast<double>(g(i,0)),0.7,tolerance,"affine gradient x including ghosts");
        test::near(static_cast<double>(g(i,1)),-1.1,tolerance,"affine gradient y including ghosts");
        test::near(static_cast<double>(g(i,2)),0.6,tolerance,"affine gradient z including ghosts");
    }
    const auto correction=td::NonOrthogonalCorrection{td::CellGradients{
        gradients.view().component(0),gradients.view().component(1),gradients.view().component(2)}};
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);
    Kokkos::View<Real*,Space> residual("corrected residual",boundary.required_field_size()),uncorrected("uncorrected",f.host.n_own);
    Kokkos::deep_copy(exec,residual,Real{7});Kokkos::deep_copy(exec,uncorrected,Real{0});
    status.begin_async();diffusion.add_residual_async(phi,residual,boundary,status,law,correction);
    diffusion.add_residual_async(phi,uncorrected,boundary,status,law);
    (void)status.check(comm);
    const auto hr=copy_view(residual,exec),hu=copy_view(uncorrected,exec);
    double local_error=0;
    for(std::size_t i=0;i<hr.extent(0);++i)test::near(hr(i),7.0,tolerance,"corrected affine residual or untouched tail");
    for(LocalIndex i=0;i<f.host.n_own;++i)local_error=std::max(local_error,std::abs(double(hu(i))));
    double global_error=0;mpi::check(MPI_Allreduce(&local_error,&global_error,1,MPI_DOUBLE,MPI_MAX,comm.native()),"uncorrected error");
    if(skew)check(global_error>1e-3,"skew test did not exercise a nonzero correction");
    else check(global_error<tolerance,"zero correction changed orthogonal diffusion");

    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern()),base(a.assembly.pattern());
    linalg::Vector<double,Exec> rhs(matrix.distribution()),base_rhs(base.distribution()),x(matrix.distribution()),y(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec),baseline=ts::make_system_view(base,base_rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);
    Kokkos::deep_copy(exec,baseline.values,0.0);Kokkos::deep_copy(exec,baseline.rhs,0.0);
    status.begin_async();diffusion.add_system_async(target,boundary,status,law,correction);
    diffusion.add_system_async(baseline,boundary,status,law);(void)status.check(comm);
    matrix.commit();rhs.commit();base.commit();base_rhs.commit();
    const auto av=copy_view(matrix.read_values(exec),exec),bv=copy_view(base.read_values(exec),exec);matrix.commit();base.commit();
    for(std::size_t k=0;k<av.extent(0);++k)test::near(av(k),bv(k),0.0,"explicit correction modified matrix coefficients");
    x.copy_from_component(phi,exec);x.commit();matrix.apply(x,y);
    const auto ax=copy_view(y.read(exec),exec),b=copy_view(rhs.read(exec),exec);y.commit();rhs.commit();
    for(LocalIndex i=0;i<f.host.n_own;++i)test::near(b(i,0)-ax(i,0),hr(i)-7.0,tolerance,"assembled and matrix-free corrected residual disagree");
    const auto revision=base.revision();
    Kokkos::View<Real*,Space> extra("correction only",f.host.n_own);Kokkos::deep_copy(exec,extra,Real{0});
    status.begin_async();diffusion.add_correction_async(extra,boundary,status,law,correction);(void)status.check(comm);
    const auto delta=copy_view(extra,exec);const auto bb=copy_view(base_rhs.read(exec),exec);base_rhs.commit();
    check(base.revision()==revision,"correction-only call touched matrix revision");
    for(LocalIndex i=0;i<f.host.n_own;++i)test::near(bb(i,0)+delta(i),b(i,0),tolerance,"correction-only RHS disagrees with complete assembly");
    const auto balance=diffusion.balance(phi,boundary,law,ts::UniformSource<Real>{0},correction);
    test::near(balance.defect,0.0,tolerance*comm.size(),"corrected affine boundary balance");
    // The same increment provider must also work with the scalar boundary manager.
    boundary.apply_component_async(phi,law,correction);exec.fence("corrected boundary values ready");
}

void corrected_linear(const Exec& exec){corrected_case(exec);}
void corrected_mixed(const Exec& exec){corrected_case(exec,true);}
void corrected_float(const Exec& exec){corrected_case<float,Kokkos::LayoutRight>(exec);}
void corrected_orthogonal(const Exec& exec){corrected_case(exec,false,false);}

void corrected_weights(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=skew_grid(comm);
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    bc::Manager<Exec,Real> boundary(f.host,a.mesh.view(),exec);
    for(int p=0;p<6;++p)boundary.set(p,bc::FixedFlux<Real>{0});
    fields::FieldStorage<Real,3,Exec,Kokkos::LayoutRight> gradients("varying gradients",f.host.n_cells,fields::FieldLocation::Custom,exec);
    auto g=gradients.create_host_buffer();
    for(LocalIndex i=0;i<f.host.n_cells;++i) {
        const auto k=static_cast<std::size_t>(i);const Real x=f.host.cell_centroid_x[k],y=f.host.cell_centroid_y[k],z=f.host.cell_centroid_z[k];
        g(i,0)=x*x+y;g(i,1)=Real{2}*x+Real{3}*y;g(i,2)=-z;
    }
    Kokkos::deep_copy(exec,gradients.view().kokkos_view(),g);
    const auto correction=td::NonOrthogonalCorrection{td::CellGradients{
        gradients.view().component(0),gradients.view().component(1),gradients.view().component(2)}};
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);
    Kokkos::View<Real*,Space> output("weighted correction",f.host.n_own);Kokkos::deep_copy(exec,output,Real{0});
    status.begin_async();diffusion.add_correction_async(output,boundary,status,law,correction);(void)status.check(comm);
    std::vector<double> expected(static_cast<std::size_t>(f.host.n_own),0.0);bool non_midpoint=false;
    for(LocalIndex face=0;face<f.host.n_inner_faces;++face) {
        const auto k=static_cast<std::size_t>(face);const auto p=f.host.face_owner[k],n=f.host.face_neigh[k];
        const double w=f.geometry.face_interp_weight[k];non_midpoint=non_midpoint||std::abs(w-0.5)>1e-5;
        const double gx=w*g(p,0)+(1-w)*g(n,0),gy=w*g(p,1)+(1-w)*g(n,1),gz=w*g(p,2)+(1-w)*g(n,2);
        const double value=2*(f.geometry.face_non_ortho_x[k]*gx+f.geometry.face_non_ortho_y[k]*gy+f.geometry.face_non_ortho_z[k]*gz);
        expected[static_cast<std::size_t>(p)]+=value;
        if(n<f.host.n_own)expected[static_cast<std::size_t>(n)]-=value;
    }
    check(non_midpoint,"gradient interpolation test lacks non-midpoint faces");
    const auto h=copy_view(output,exec);double local=0,global=0;
    for(LocalIndex i=0;i<f.host.n_own;++i){test::near(h(i),expected[static_cast<std::size_t>(i)],1e-11,"wrong gradient weights or owner/neighbour correction sign");local+=h(i);}
    mpi::check(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,comm.native()),"correction conservation");
    test::near(global,0.0,1e-10*comm.size(),"internal correction is not conservative across ranks");

    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::BsrMatrix<double,Exec> blocks(a.assembly.pattern(),3);
    linalg::Vector<double,Exec> rhs(matrix.distribution()),block_rhs(blocks.distribution());
    Kokkos::deep_copy(exec,blocks.overwrite_blocks(exec),4.0);Kokkos::deep_copy(exec,block_rhs.overwrite(exec),7.0);
    const auto scalar=ts::make_system_view(matrix,rhs,a.assembly,exec);const auto block=ts::make_system_view(blocks,block_rhs,a.assembly,exec,1);
    Kokkos::deep_copy(exec,scalar.values,4.0);Kokkos::deep_copy(exec,scalar.rhs,7.0);
    status.begin_async();diffusion.add_system_async(scalar,boundary,status,law,correction);
    diffusion.add_system_async(block,boundary,status,law,correction);(void)status.check(comm);
    matrix.commit();rhs.commit();blocks.commit();block_rhs.commit();
    const auto sv=copy_view(matrix.read_values(exec),exec);const auto bv=copy_view(blocks.read_blocks(exec),exec);
    const auto sb=copy_view(rhs.read(exec),exec),bb=copy_view(block_rhs.read(exec),exec);
    matrix.commit();rhs.commit();blocks.commit();block_rhs.commit();
    for(std::size_t k=0;k<sv.extent(0);++k)for(int r=0;r<3;++r)for(int c=0;c<3;++c)
        test::near(bv(k,r,c),r==1&&c==1?sv(k):4.0,1e-12,"corrected BSR coefficient slice");
    for(std::size_t i=0;i<sb.extent(0);++i)for(int c=0;c<3;++c)
        test::near(bb(3*i+static_cast<std::size_t>(c),0),c==1?sb(i,0):7.0,1e-11,"corrected BSR RHS slice");
}

void corrected_empty(const Exec& exec)
{
    const auto comm=mpi::CommView::world();
    for(bool globally_empty : {false,true}) {
        auto f=bar(comm,false,true,globally_empty);
        const auto faces=static_cast<std::size_t>(f.host.n_faces);
        f.geometry.active_mask=f.geometry.active_mask|mesh::AuxGeomType::FaceNonOrthogonalityVector;
        f.geometry.face_non_ortho_x.assign(faces,0);f.geometry.face_non_ortho_y.assign(faces,0);f.geometry.face_non_ortho_z.assign(faces,0);
        Attached a(f,comm,exec);bc::Manager<Exec,Real> boundary(f.host,a.mesh.view(),exec);
        boundary.set(0,bc::FixedValue<Real>{0});boundary.set(1,bc::FixedValue<Real>{0});
        fields::FieldStorage<Real,1,Exec> field("empty field",boundary.required_field_size(),fields::FieldLocation::Custom,exec);
        fields::FieldStorage<Real,3,Exec> gradients("empty gradients",f.host.n_cells,fields::FieldLocation::Custom,exec);
        field.fill(exec,Real{0});gradients.fill(exec,Real{0});
        const auto correction=td::NonOrthogonalCorrection{td::CellGradients{
            gradients.view().component(0),gradients.view().component(1),gradients.view().component(2)}};
        const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{1}};
        Kokkos::View<Real*,Space> output("empty residual with tail",static_cast<std::size_t>(f.host.n_own)+1);
        Kokkos::deep_copy(exec,output,Real{5});
        td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);status.begin_async();
        diffusion.add_residual_async(field.view().component(0),output,boundary,status,law,correction);
        diffusion.add_correction_async(output,boundary,status,law,correction);(void)status.check(comm);
        const auto h=copy_view(output,exec);for(std::size_t i=0;i<h.extent(0);++i)test::near(h(i),Real{5},Real{0},"empty/zero correction modified a target");
        ts::SteadyDiffusion<> solver(a.plan,a.assembly,field.view().component(0),settings());
        check(solver.solve_owned(boundary,law,ts::UniformSource<Real>{0},correction).converged,"corrected empty solve failed");
    }
}

// External orchestration: field publication -> owned gradients -> gradient publication -> RHS correction -> solve.
// The immutable matrix and the same linear solver are retained across all outer iterations.
void corrected_iterations(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=skew_grid(comm);
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    SkewBoundary boundary(f.host,a.mesh.view(),exec);configure_affine(boundary,f);
    fields::FieldStorage<Real,1,Exec> field("iterate",boundary.required_field_size(),fields::FieldLocation::Custom,exec);
    fields::FieldStorage<Real,3,Exec> gradients("iterate gradients",f.host.n_cells,fields::FieldLocation::Custom,exec);
    field.fill(exec,Real{0});gradients.fill(exec,Real{0});const auto phi=field.view().component(0);
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    const auto correction=td::NonOrthogonalCorrection{td::CellGradients{
        gradients.view().component(0),gradients.view().component(1),gradients.view().component(2)}};
    halo::Exchange<Real,Exec> field_halo(a.mesh.view(),a.mesh.halo_plan(),comm,35,exec),gradient_halo(a.mesh.view(),a.mesh.halo_plan(),comm,36,exec);
    field_halo.add_field(field.view());field_halo.prepare();gradient_halo.add_field(gradients.view());gradient_halo.prepare();
    numerics::gradient::GradientOperator<Exec> gradient(a.mesh.view(),a.connectivity.view(),a.geometry.view(),{},exec);
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution()),x(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);
    status.begin_async();diffusion.add_system_async(target,boundary,status,law);(void)status.check(comm);matrix.commit();rhs.commit();
    const auto revision=matrix.revision();
    Kokkos::View<double*,Space> base_rhs("frozen RHS",f.host.n_own);
    Kokkos::deep_copy(exec,base_rhs,Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0));rhs.commit();
    linalg::Solver<decltype(matrix)> linear(matrix,settings().linear);
    double error=1;int iterations=0;
    for(;iterations<60&&error>2e-9;++iterations) {
        field_halo.exchange();boundary.apply_async(field.view(),law,correction);
        gradient.apply(field.view(),gradients.view());gradient_halo.exchange();
        const auto b=Kokkos::subview(rhs.overwrite(exec),Kokkos::ALL(),0);
        Kokkos::deep_copy(exec,b,base_rhs);status.begin_async();
        diffusion.add_correction_async(b,boundary,status,law,correction);(void)status.check(comm);rhs.commit();
        x.copy_from_component(phi,exec);x.commit();check(linear.solve(rhs,x).converged,"nonorthogonal outer linear solve failed");
        x.copy_to_component(phi,exec);x.commit();
        const auto h=field.create_host_buffer();Kokkos::deep_copy(exec,h,field.view().kokkos_view());exec.fence("outer error ready");
        double local=0;
        for(LocalIndex i=0;i<f.host.n_own;++i)local=std::max(local,std::abs(double(h(i))-double(affine_value(
            f.host.cell_centroid_x[static_cast<std::size_t>(i)],f.host.cell_centroid_y[static_cast<std::size_t>(i)],f.host.cell_centroid_z[static_cast<std::size_t>(i)]))));
        mpi::check(MPI_Allreduce(&local,&error,1,MPI_DOUBLE,MPI_MAX,comm.native()),"outer error");
        check(matrix.revision()==revision,"outer correction iterations rebuilt the matrix");
    }
    check(iterations>1&&error<=2e-9,"nonorthogonal outer iterations did not converge to affine solution");
    // The convenience solver also accepts one frozen-gradient corrected solve.
    field_halo.exchange();boundary.apply_async(field.view(),law,correction);
    gradient.apply(field.view(),gradients.view());gradient_halo.exchange();
    ts::SteadyDiffusion<> steady(a.plan,a.assembly,phi,settings());
    check(steady.solve_owned(boundary,law,ts::UniformSource<Real>{0},correction).converged,"corrected convenience solve failed");
    test::near(steady.balance(boundary,law,ts::UniformSource<Real>{0},correction).defect,0.0,1e-7,"corrected convenience balance");
}

template<int Which>
void rejected_correction(const Exec& exec)
{
    const auto comm=mpi::CommView::world();auto f=skew_grid(comm);
    if constexpr(Which==0){f.geometry.active_mask=mesh::AuxGeomType::FaceNormalDistance|mesh::AuxGeomType::FaceAreaOverNormalDistance|mesh::AuxGeomType::FaceInterpolationWeights;}
    if constexpr(Which==1){f.geometry.active_mask=mesh::AuxGeomType::FaceNormalDistance|mesh::AuxGeomType::FaceAreaOverNormalDistance|mesh::AuxGeomType::FaceNonOrthogonalityVector;f.geometry.face_interp_weight.clear();}
    if constexpr(Which==3)f.geometry.face_non_ortho_x[0]+=GeometryReal{1};
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    SkewBoundary boundary(f.host,a.mesh.view(),exec);configure_affine(boundary,f);
    if constexpr(Which==5)for(int p=0;p<6;++p)boundary.set(p,ChangingSlope{});
    const auto count=static_cast<std::size_t>(f.host.n_cells)-(Which==4?1:0);
    Kokkos::View<Real*,Space> gx("gx",count),gy("gy",count),gz("gz",count),output("output",f.host.n_own);
    Kokkos::deep_copy(exec,gx,Real{1});Kokkos::deep_copy(exec,gy,Real{2});Kokkos::deep_copy(exec,gz,Real{3});Kokkos::deep_copy(exec,output,Real{0});
    if constexpr(Which==2)Kokkos::deep_copy(exec,gx,std::numeric_limits<Real>::quiet_NaN());
    const auto correction=td::NonOrthogonalCorrection{td::CellGradients{gx,gy,gz}};
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);status.begin_async();
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{1}};
    diffusion.add_correction_async(output,boundary,status,law,correction);(void)status.check(comm);
    check(false,"invalid correction unexpectedly accepted");
}
