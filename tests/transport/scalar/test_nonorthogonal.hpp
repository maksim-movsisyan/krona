#pragma once

// Included by test_diffusion.cpp inside its anonymous namespace; uses the shared test fixture types.
// A stretched grid has column-dependent shear and skew parallelepipeds. Metrics are independent
// analytical geometry, so an affine solution has zero integrated divergence in every owned cell.

KOKKOS_INLINE_FUNCTION Real affine_value(Real x,Real y,Real z)
{return Real{2}+Real(0.7)*x-Real(1.1)*y+Real(0.6)*z;}

/// @brief Supplies analytic affine gradient components for independent correction references.
struct ExactGradient {
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t) const
    {return Kokkos::Array<Real,3>{Real(0.7),Real(-1.1),Real(0.6)};}
};

/// @brief Prescribes the analytic skew-grid affine field at boundary face positions.
struct AffineValue {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c,Real increment=0)const
    {return bc::FixedValue<Real>{affine_value(c.x,c.y,c.z)}(c,increment);}
};

/// @brief Supplies an affine-consistent Robin relation including the explicit geometric correction.
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

/// @brief Deliberately makes an implicit boundary slope depend on the explicit correction.
struct ChangingSlope {
    KOKKOS_INLINE_FUNCTION bc::Relation<Real> operator()(const boundary::FaceContext<Real>& c,Real increment=0)const
    {return {0,0,-(Real{1}+increment*increment)/c.distance,0};}
};
using SkewBoundary=bc::Manager<Exec,Real,AffineValue,AffineRobin,ChangingSlope>;

/// @brief Assigns fixed or mixed boundary relations matching the exact skew-grid affine field.
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

/// @brief Checks nonorthogonal correction exactness, residual agreement and unchanged implicit coefficients.
template<class Storage=double,class Layout=Kokkos::LayoutLeft>
void corrected_case(const Exec& exec,bool mixed=false,bool skew=true)
{
    const auto comm=mpi::CommView::world();const auto f=skew_grid(comm,skew);
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    krona::boundary::BoundaryLayout<Exec, Real> boundary_layout(f.host, a.mesh.view(), exec);
    SkewBoundary boundary(boundary_layout);configure_affine(boundary,f,mixed);
    fields::FieldStorage<Storage,2,Exec,Layout> field("affine",boundary.required_field_size(),fields::FieldLocation::Custom,exec);
    fields::FieldStorage<Storage,3,Exec,Layout> gradients("gradients",static_cast<std::size_t>(f.host.n_cells),fields::FieldLocation::Custom,exec);
    field.fill_async(exec,Storage{-33});gradients.fill_async(exec,std::numeric_limits<Storage>::quiet_NaN());
    const auto phi=field.view().component(1);const auto m=a.mesh.view();
    using Policy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<LocalIndex>>;
    Kokkos::parallel_for("test/affine_owned",Policy(exec,0,f.host.n_own),KOKKOS_LAMBDA(LocalIndex i){
        phi(i)=static_cast<Storage>(affine_value(m.geometry.cell_centroid_x(i),m.geometry.cell_centroid_y(i),m.geometry.cell_centroid_z(i)));
    });
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    halo::Exchange<Storage,Exec> field_halo(m,a.mesh.halo_plan(),comm,31,exec);
    field_halo.add_component(phi);field_halo.prepare();field_halo.exchange();
    boundary.finalize();
    krona::boundary::scalar::Status<Exec> boundary_status_0(exec);
    boundary_status_0.begin_async();
    boundary.apply_component_async(phi, boundary_status_0,law,numerics::gradient::GradientCorrection{ExactGradient{}});
    boundary_status_0.check();
    numerics::gradient::GradientOperator<Exec> gradient(m,a.connectivity.view(),a.geometry.view(),{},exec);
    gradient.apply_component_async(phi,gradients.view().component(0),gradients.view().component(1),gradients.view().component(2));
    halo::Exchange<Storage,Exec> gradient_halo(m,a.mesh.halo_plan(),comm,32,exec);
    gradient_halo.add_field(gradients.view());gradient_halo.prepare();gradient_halo.exchange();
    const auto g=gradients.create_host_buffer();Kokkos::deep_copy(exec,g,gradients.view().kokkos_view());exec.fence("test gradients ready");
    const double tolerance=sizeof(Storage) == sizeof(float)?3e-5:2e-11;
    for(LocalIndex i=0;i<f.host.n_cells;++i) {
        test::near(static_cast<double>(g(i,0)),0.7,tolerance,"affine gradient x including ghosts");
        test::near(static_cast<double>(g(i,1)),-1.1,tolerance,"affine gradient y including ghosts");
        test::near(static_cast<double>(g(i,2)),0.6,tolerance,"affine gradient z including ghosts");
    }
    const auto correction=numerics::gradient::GradientCorrection{numerics::gradient::cell_gradients(
        gradients.view().component(0),gradients.view().component(1),gradients.view().component(2))};
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

    // Require an observable uncorrected error so a zero correction cannot pass the skew test.
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
    x.copy_from_component_async(phi,exec);x.commit();matrix.apply(x,y);
    const auto ax=copy_view(y.read(exec),exec),b=copy_view(rhs.read(exec),exec);y.commit();rhs.commit();
    for(LocalIndex i=0;i<f.host.n_own;++i)test::near(b(i,0)-ax(i,0),hr(i)-7.0,tolerance,"assembled and matrix-free corrected residual disagree");

    // A correction-only call may update the RHS while leaving the frozen matrix revision unchanged.
    const auto revision=base.revision();
    Kokkos::View<Real*,Space> extra("correction only",f.host.n_own);Kokkos::deep_copy(exec,extra,Real{0});
    status.begin_async();diffusion.add_correction_async(extra,boundary,status,law,correction);(void)status.check(comm);
    const auto delta=copy_view(extra,exec);const auto bb=copy_view(base_rhs.read(exec),exec);base_rhs.commit();
    check(base.revision()==revision,"correction-only call touched matrix revision");
    for(LocalIndex i=0;i<f.host.n_own;++i)test::near(bb(i,0)+delta(i),b(i,0),tolerance,"correction-only RHS disagrees with complete assembly");
    const auto balance=diffusion.balance(phi,boundary,law,ts::UniformSource<Real>{0},correction);
    test::near(balance.defect,0.0,tolerance*comm.size(),"corrected affine boundary balance");

    // The same increment provider must also work with the scalar boundary manager.
    krona::boundary::scalar::Status<Exec> boundary_status_1(exec);
    boundary_status_1.begin_async();
    boundary.apply_component_async(phi, boundary_status_1,law,correction);exec.fence("corrected boundary values ready");
    boundary_status_1.check();
}

/// @brief Checks affine nonorthogonal correction with fixed-value boundaries.
void corrected_linear(const Exec& exec){corrected_case(exec);}

/// @brief Checks nonorthogonal correction with mixed gradient, flux and Robin boundaries.
void corrected_mixed(const Exec& exec){corrected_case(exec,true);}

/// @brief Checks corrected diffusion with float/AoS state and gradient storage.
void corrected_float(const Exec& exec){corrected_case<float,Kokkos::LayoutRight>(exec);}

/// @brief Checks that the same correction path reduces to zero on orthogonal geometry.
void corrected_orthogonal(const Exec& exec){corrected_case(exec,false,false);}

/// @brief Checks asymmetric gradient interpolation, conservative correction signs and BSR slices.
void corrected_weights(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=skew_grid(comm);
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    krona::boundary::BoundaryLayout<Exec, Real> boundary_layout(f.host, a.mesh.view(), exec);
    bc::Manager<Exec,Real> boundary(boundary_layout);
    for(int p=0;p<6;++p)boundary.set(p,bc::FixedFlux<Real>{0});
    fields::FieldStorage<Real,3,Exec,Kokkos::LayoutRight> gradients("varying gradients",f.host.n_cells,fields::FieldLocation::Custom,exec);
    auto g=gradients.create_host_buffer();
    for(LocalIndex i=0;i<f.host.n_cells;++i) {
        const auto k=static_cast<std::size_t>(i);const Real x=f.host.cell_centroid_x[k],y=f.host.cell_centroid_y[k],z=f.host.cell_centroid_z[k];
        g(i,0)=x*x+y;g(i,1)=Real{2}*x+Real{3}*y;g(i,2)=-z;
    }
    Kokkos::deep_copy(exec,gradients.view().kokkos_view(),g);
    const auto correction=numerics::gradient::GradientCorrection{numerics::gradient::cell_gradients(
        gradients.view().component(0),gradients.view().component(1),gradients.view().component(2))};
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);
    Kokkos::View<Real*,Space> output("weighted correction",f.host.n_own);Kokkos::deep_copy(exec,output,Real{0});
    boundary.finalize();
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

/// @brief Checks empty and zero correction updates preserve their targets.
void corrected_empty(const Exec& exec)
{
    const auto comm=mpi::CommView::world();
    for(bool globally_empty : {false,true}) {
        auto f=bar(comm,false,true,globally_empty);
        const auto faces=static_cast<std::size_t>(f.host.n_faces);
        f.geometry.active_mask=f.geometry.active_mask|mesh::AuxGeomType::FaceNonOrthogonalityVector;
        f.geometry.face_non_ortho_x.assign(faces,0);f.geometry.face_non_ortho_y.assign(faces,0);f.geometry.face_non_ortho_z.assign(faces,0);
        Attached a(f,comm,exec);krona::boundary::BoundaryLayout<Exec, Real> boundary_layout(f.host, a.mesh.view(), exec);
        bc::Manager<Exec,Real> boundary(boundary_layout);
        boundary.set(0,bc::FixedValue<Real>{0});boundary.set(1,bc::FixedValue<Real>{0});
        fields::FieldStorage<Real,1,Exec> field("empty field",boundary.required_field_size(),fields::FieldLocation::Custom,exec);
        fields::FieldStorage<Real,3,Exec> gradients("empty gradients",f.host.n_cells,fields::FieldLocation::Custom,exec);
        field.fill_async(exec,Real{0});gradients.fill_async(exec,Real{0});
        const auto correction=numerics::gradient::GradientCorrection{numerics::gradient::cell_gradients(
            gradients.view().component(0),gradients.view().component(1),gradients.view().component(2))};
        const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{1}};
        Kokkos::View<Real*,Space> output("empty residual with tail",static_cast<std::size_t>(f.host.n_own)+1);
        Kokkos::deep_copy(exec,output,Real{5});
        td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);status.begin_async();
        boundary.finalize();
        diffusion.add_residual_async(field.view().component(0),output,boundary,status,law,correction);
        diffusion.add_correction_async(output,boundary,status,law,correction);(void)status.check(comm);
        const auto h=copy_view(output,exec);for(std::size_t i=0;i<h.extent(0);++i)test::near(h(i),Real{5},Real{0},"empty/zero correction modified a target");
        check(solve_field(a,field.view().component(0),boundary,settings(),law,ts::UniformSource<Real>{0},correction).converged,"corrected empty solve failed");
    }
}

// External orchestration: field publication -> owned gradients -> gradient publication -> RHS correction -> solve.
// The immutable matrix and the same linear solver are retained across all outer iterations.
/// @brief Checks externally orchestrated gradient/correction iterations with matrix and solver reuse.
void corrected_iterations(const Exec& exec)
{
    const auto comm=mpi::CommView::world();const auto f=skew_grid(comm);
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    krona::boundary::BoundaryLayout<Exec, Real> boundary_layout(f.host, a.mesh.view(), exec);
    SkewBoundary boundary(boundary_layout);configure_affine(boundary,f);
    fields::FieldStorage<Real,1,Exec> field("iterate",boundary.required_field_size(),fields::FieldLocation::Custom,exec);
    fields::FieldStorage<Real,3,Exec> gradients("iterate gradients",f.host.n_cells,fields::FieldLocation::Custom,exec);
    field.fill_async(exec,Real{0});gradients.fill_async(exec,Real{0});const auto phi=field.view().component(0);
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
    const auto correction=numerics::gradient::GradientCorrection{numerics::gradient::cell_gradients(
        gradients.view().component(0),gradients.view().component(1),gradients.view().component(2))};
    halo::Exchange<Real,Exec> field_halo(a.mesh.view(),a.mesh.halo_plan(),comm,35,exec),gradient_halo(a.mesh.view(),a.mesh.halo_plan(),comm,36,exec);
    field_halo.add_field(field.view());field_halo.prepare();gradient_halo.add_field(gradients.view());gradient_halo.prepare();
    numerics::gradient::GradientOperator<Exec> gradient(a.mesh.view(),a.connectivity.view(),a.geometry.view(),{},exec);
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);
    linalg::CsrMatrix<double,Exec> matrix(a.assembly.pattern());linalg::Vector<double,Exec> rhs(matrix.distribution()),x(matrix.distribution());
    const auto target=ts::make_system_view(matrix,rhs,a.assembly,exec);
    Kokkos::deep_copy(exec,target.values,0.0);Kokkos::deep_copy(exec,target.rhs,0.0);
    boundary.finalize();
    status.begin_async();diffusion.add_system_async(target,boundary,status,law);(void)status.check(comm);matrix.commit();rhs.commit();
    const auto revision=matrix.revision();
    Kokkos::View<double*,Space> base_rhs("frozen RHS",f.host.n_own);
    Kokkos::deep_copy(exec,base_rhs,Kokkos::subview(rhs.read(exec),Kokkos::ALL(),0));rhs.commit();
    linalg::Solver<decltype(matrix)> linear(matrix,settings());
    double error=1;int iterations=0;
    for(;iterations<60&&error>2e-9;++iterations) {
        krona::boundary::scalar::Status<Exec> boundary_status_2(exec);
        boundary_status_2.begin_async();
        field_halo.exchange();boundary.apply_async(field.view(), boundary_status_2,law,correction);
        boundary_status_2.check();
        gradient.apply_async(field.view(),gradients.view());gradient_halo.exchange();
        const auto b=Kokkos::subview(rhs.overwrite(exec),Kokkos::ALL(),0);
        Kokkos::deep_copy(exec,b,base_rhs);status.begin_async();
        diffusion.add_correction_async(b,boundary,status,law,correction);(void)status.check(comm);rhs.commit();
        x.copy_from_component_async(phi,exec);x.commit();check(linear.solve(rhs,x).converged,"nonorthogonal outer linear solve failed");
        x.copy_to_component_async(phi,exec);x.commit();
        const auto h=field.create_host_buffer();Kokkos::deep_copy(exec,h,field.view().kokkos_view());exec.fence("outer error ready");
        double local=0;
        for(LocalIndex i=0;i<f.host.n_own;++i)local=std::max(local,std::abs(double(h(i))-double(affine_value(
            f.host.cell_centroid_x[static_cast<std::size_t>(i)],f.host.cell_centroid_y[static_cast<std::size_t>(i)],f.host.cell_centroid_z[static_cast<std::size_t>(i)]))));
        mpi::check(MPI_Allreduce(&local,&error,1,MPI_DOUBLE,MPI_MAX,comm.native()),"outer error");
        check(matrix.revision()==revision,"outer correction iterations rebuilt the matrix");
    }
    check(iterations>1&&error<=2e-9,"nonorthogonal outer iterations did not converge to affine solution");

    // The convenience solver also accepts one frozen-gradient corrected solve.
    krona::boundary::scalar::Status<Exec> boundary_status_3(exec);
    boundary_status_3.begin_async();
    field_halo.exchange();boundary.apply_async(field.view(), boundary_status_3,law,correction);
    boundary_status_3.check();
    gradient.apply_async(field.view(),gradients.view());gradient_halo.exchange();
    check(solve_field(a,phi,boundary,settings(),law,ts::UniformSource<Real>{0},correction).converged,"corrected rebuilt solve failed");
    test::near(diffusion.balance(phi,boundary,law,ts::UniformSource<Real>{0},correction).defect,0.0,1e-7,"corrected rebuilt balance");
}

/// @brief Triggers missing/inconsistent metrics, invalid gradients or correction-dependent relations.
template<int Which>
void rejected_correction(const Exec& exec)
{
    const auto comm=mpi::CommView::world();auto f=skew_grid(comm);
    if constexpr(Which==0){f.geometry.active_mask=mesh::AuxGeomType::FaceNormalDistance|mesh::AuxGeomType::FaceAreaOverNormalDistance|mesh::AuxGeomType::FaceInterpolationWeights;}
    if constexpr(Which==1){f.geometry.active_mask=mesh::AuxGeomType::FaceNormalDistance|mesh::AuxGeomType::FaceAreaOverNormalDistance|mesh::AuxGeomType::FaceNonOrthogonalityVector;f.geometry.face_interp_weight.clear();}
    if constexpr(Which==3)f.geometry.face_non_ortho_x[0]+=GeometryReal{1};
    Attached a(f,comm,exec,td::DiffusionOptions{false,1e-6});
    krona::boundary::BoundaryLayout<Exec, Real> boundary_layout(f.host, a.mesh.view(), exec);
    SkewBoundary boundary(boundary_layout);configure_affine(boundary,f);
    if constexpr(Which==5)for(int p=0;p<6;++p)boundary.set(p,ChangingSlope{});
    const auto count=static_cast<std::size_t>(f.host.n_cells)-(Which==4?1:0);
    Kokkos::View<Real*,Space> gx("gx",count),gy("gy",count),gz("gz",count),output("output",f.host.n_own);
    Kokkos::deep_copy(exec,gx,Real{1});Kokkos::deep_copy(exec,gy,Real{2});Kokkos::deep_copy(exec,gz,Real{3});Kokkos::deep_copy(exec,output,Real{0});
    if constexpr(Which==2)Kokkos::deep_copy(exec,gx,std::numeric_limits<Real>::quiet_NaN());
    const auto correction=numerics::gradient::GradientCorrection{numerics::gradient::cell_gradients(gx,gy,gz)};
    td::DiffusionOperator<> diffusion(a.plan);ts::AssemblyStatus<Exec> status(exec);status.begin_async();
    const auto law=transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{1}};
    boundary.finalize();
    diffusion.add_correction_async(output,boundary,status,law,correction);(void)status.check(comm);
    check(false,"invalid correction unexpectedly accepted");
}
