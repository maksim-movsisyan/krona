#pragma once
#include <krona/transport/euler/detail/kernels.hpp>
#include <krona/numerics/riemann/hllc.hpp>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace krona::transport::euler {

/// @brief Local additive r -= sum(A*F_inviscid), no volume division. Borrowed input/output; no halo/time integration.
/// Row gather recomputes owned-owned face fluxes. Boundary topology is shared through EulerPlan; no face-flux buffer.
template<class StateLaw,class Solver=numerics::riemann::Hllc<>,class Exec=execution::ExecutionSpace,
    class Real=ComputeReal,class Accumulator=AccumulationReal>
class EulerOperator {
    static_assert(std::is_floating_point_v<Real>&&std::is_floating_point_v<Accumulator>);
    using Space=typename Exec::memory_space;
    using Policy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<std::size_t>>;
public:
    using execution_space=Exec;
    using memory_space=Space;
    using plan_type=EulerPlan<Exec,Real>;
    EulerOperator(plan_type plan,StateLaw law,Solver solver={}):plan_(std::move(plan)),law_(std::move(law)),solver_(std::move(solver))
    { if constexpr(requires{law_.validate();})law_.validate(); }

    /// @brief Selected inner faces only; caller guarantees readiness of the complete selected reconstruction stencil.
    template<class Reconstruction,class Output>
    void add_inner_residual_async(Reconstruction reconstruction,Output output,AssemblyStatus<Exec>& status,
        FaceRange faces={},CellRange cells={})const
    {
        validate(reconstruction,output,false);cells=cell_range(cells);
        const auto mesh=plan_.mesh_view();
        if(faces.end==-1)faces.end=mesh.sizes.n_inner_faces;
        mpi::require(faces.begin>=0&&faces.end>=faces.begin&&faces.end<=mesh.sizes.n_inner_faces,"Euler: invalid inner face range");
        const auto& exec=execution_space_instance();const auto flags=status.device_flags(exec);
        if(faces.begin==faces.end||cells.begin==cells.end)return;
        const auto physics=detail::Physics<Real,Reconstruction,StateLaw,Solver,decltype(flags)>{reconstruction,law_,solver_,flags};
        Kokkos::parallel_for("euler/inner_rows",Policy(exec,cells.begin,cells.end),
            detail::InnerRows<Space,Real,Accumulator,Output,decltype(physics)>{detail::row_data(mesh,plan_.connectivity_view()),output,physics,faces});
    }

    /// @brief Typed patch dispatch, one thread per (patch,owned cell). Different patch launches are ordered on exec.
    /// Physical face slots must already be current; exterior states are temporary local values.
    template<class Reconstruction,class Output,class Boundary,class Closure=PreparedExterior>
    void add_boundary_residual_async(Reconstruction reconstruction,Output output,const Boundary& boundary,
        AssemblyStatus<Exec>& status,Closure closure={},Real time=0,CellRange cells={})const
    {
        using Model=typename Boundary::model_type;
        static_assert(Model::compressible&&Model::thermal,"Euler requires a compressible thermal boundary model");
        static_assert(!std::is_arithmetic_v<Closure>,"Pass time after an explicit boundary closure policy");
        validate(reconstruction,output,true);validate_boundary(boundary);
        mpi::require(std::isfinite(time),"Euler: time must be finite");cells=cell_range(cells);
        const auto& exec=execution_space_instance();const auto flags=status.device_flags(exec);
        const auto physics=detail::Physics<Real,Reconstruction,StateLaw,Solver,decltype(flags)>{reconstruction,law_,solver_,flags};
        boundary.for_each_patch([&](auto range,const auto& condition){
            using C=std::remove_cvref_t<decltype(condition)>;
            const auto rows=plan_.patch_rows(range);
            if(rows.begin==rows.end||cells.begin==cells.end)return;
            Kokkos::parallel_for("euler/boundary_cell_rows",Policy(exec,rows.begin,rows.end),
                detail::BoundaryCellRows<Space,Real,Accumulator,Output,decltype(physics),typename Boundary::model_type,C,Closure>{
                    boundary.geometry_view(),plan_.boundary_rows(),output,physics,boundary.model(),condition,closure,cells,time});
        });
    }

    template<class Reconstruction,class Output,class Boundary,class Closure=PreparedExterior>
    void add_residual_async(Reconstruction reconstruction,Output output,const Boundary& boundary,
        AssemblyStatus<Exec>& status,Closure closure={},Real time=0,CellRange cells={})const
    {
        // Validate the complete request before submitting either contribution.
        validate(reconstruction,output,true);validate_boundary(boundary);
        mpi::require(std::isfinite(time),"Euler: time must be finite");
        cells=cell_range(cells);
        add_inner_residual_async(reconstruction,output,status,{},cells);
        add_boundary_residual_async(reconstruction,output,boundary,status,closure,time,cells);
    }
    [[nodiscard]] const plan_type& plan()const noexcept{return plan_;}
    [[nodiscard]] const Exec& execution_space_instance()const noexcept{return plan_.execution_space_instance();}
private:
    plan_type plan_;
    StateLaw law_;
    Solver solver_;
    CellRange cell_range(CellRange range)const
    {
        const auto count=static_cast<std::size_t>(plan_.mesh_view().sizes.n_own);
        if(range.end==std::numeric_limits<std::size_t>::max())range.end=count;
        mpi::require(range.begin<=range.end&&range.end<=count,"Euler: invalid owned cell range");return range;
    }
    template<class Reconstruction,class Output>
    void validate(const Reconstruction& reconstruction,const Output& output,bool with_boundary)const
    {
        const auto s=plan_.mesh_view().sizes;
        const auto cells=static_cast<std::size_t>(s.n_cells),owned=static_cast<std::size_t>(s.n_own);
        reconstruction.template validate<Exec>(cells,cells+static_cast<std::size_t>(s.n_faces-s.n_inner_faces),with_boundary);
        detail::validate_pack<Exec,true>(output,owned);
        if(owned==0)return;
        detail::reject_exact_alias(reconstruction.data.values,output);
        if constexpr(requires{reconstruction.data.gradients.x;}){
            detail::reject_exact_alias(reconstruction.data.gradients.x,output);
            detail::reject_exact_alias(reconstruction.data.gradients.y,output);
            detail::reject_exact_alias(reconstruction.data.gradients.z,output);
        }
        detail::reject_exact_alias(reconstruction.data.factors,output);
    }
    template<class Boundary>
    void validate_boundary(const Boundary& boundary)const
    {
        const auto b=boundary.geometry_view();const auto m=plan_.mesh_view();const auto& g=m.geometry;
        mpi::require(b.n_cells==m.sizes.n_cells&&b.n_inner_faces==m.sizes.n_inner_faces&&
            b.face_owner.data()==m.connectivity.face_owner.data()&&b.cell_x.data()==g.cell_centroid_x.data()&&
            b.cell_y.data()==g.cell_centroid_y.data()&&b.cell_z.data()==g.cell_centroid_z.data()&&
            b.face_x.data()==g.face_centroid_x.data()&&b.face_y.data()==g.face_centroid_y.data()&&b.face_z.data()==g.face_centroid_z.data()&&
            b.normal_x.data()==g.face_normal_x.data()&&b.normal_y.data()==g.face_normal_y.data()&&b.normal_z.data()==g.face_normal_z.data()&&
            b.face_area.data()==g.face_area.data(),"Euler: boundary manager must use the same compute mesh");
        boundary.for_each_patch([&](auto range,const auto&){(void)plan_.patch_rows(range);});
    }
};
template<class E,class R,class L,class S>EulerOperator(EulerPlan<E,R>,L,S)->EulerOperator<L,S,E,R>;
template<class E,class R,class L>EulerOperator(EulerPlan<E,R>,L)->EulerOperator<L,numerics::riemann::Hllc<>,E,R>;
} // namespace krona::transport::euler
