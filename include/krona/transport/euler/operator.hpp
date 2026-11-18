#pragma once
#include <krona/transport/euler/detail/jacobian_kernels.hpp>
#include <krona/numerics/riemann/hllc.hpp>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace krona::transport::euler {

/// @brief Local additive r -= sum(A*F_inviscid), no volume division. Borrowed input/output; no halo/time integration.
/// Inner faces gather by row. Physical boundary faces scatter by patch with atomic adds.
template<class StateLaw,class Solver=numerics::riemann::Hllc<>,class Exec=execution::ExecutionSpace,
    class Real=ComputeReal,class Accumulator=AccumulationReal>
class EulerOperator {
    static_assert(std::is_floating_point_v<Real>&&std::is_floating_point_v<Accumulator>);
    using Space=typename Exec::memory_space;
    using Policy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<std::size_t>>;
    using FacePolicy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<LocalIndex>>;
public:
    using execution_space=Exec;
    using memory_space=Space;
    using plan_type=EulerPlan<Exec,Real>;
    EulerOperator(plan_type plan,StateLaw law,Solver solver={}):plan_(std::move(plan)),law_(std::move(law)),solver_(std::move(solver))
    { if constexpr(requires{law_.validate();})law_.validate(); }

    /// @brief Selected inner faces only; caller guarantees readiness of the complete selected reconstruction stencil.
    template<class Reconstruction,class Output>
    void add_inner_residual_async(Reconstruction reconstruction,Output output,AssemblyStatus<Exec>& status,
        FaceRange faces={},CellRange cells={}) const {
        validate(reconstruction,output,false);cells=cell_range(cells);
        const auto mesh=plan_.mesh_view();
        faces = inner_range(faces);
        const auto& exec=execution_space_instance();const auto flags=status.device_flags(exec);
        if(faces.begin==faces.end||cells.begin==cells.end)return;
        const auto physics=detail::Physics<Real,Reconstruction,StateLaw,Solver,decltype(flags)>{reconstruction,law_,solver_,flags};
        Kokkos::parallel_for("euler/inner_rows",Policy(exec,cells.begin,cells.end),
            detail::InnerRows<Space,Real,Accumulator,Output,decltype(physics)>{detail::row_data(mesh,plan_.connectivity_view()),output,physics,faces});
    }

    /// @brief Typed patch dispatch, one thread per face. Patch launches are ordered on exec.
    /// Physical face slots must already be current; exterior states are temporary local values.
    template<class Reconstruction,class Output,class Boundary,class Closure=PreparedExterior>
    void add_boundary_residual_async(Reconstruction reconstruction,Output output,const Boundary& boundary,
        AssemblyStatus<Exec>& status,Closure closure={},Real time=0,CellRange cells={})const {
        using Model=typename Boundary::model_type;
        static_assert(Model::compressible&&Model::thermal,"Euler requires a compressible thermal boundary model");
        static_assert(!std::is_arithmetic_v<Closure>,"Pass time after an explicit boundary closure policy");
        validate(reconstruction,output,true);validate_boundary(boundary);
        mpi::require(std::isfinite(time),"Euler: time must be finite");cells=cell_range(cells);
        const auto& exec=execution_space_instance();const auto flags=status.device_flags(exec);
        const auto physics=detail::Physics<Real,Reconstruction,StateLaw,Solver,decltype(flags)>{reconstruction,law_,solver_,flags};
        if (cells.begin == cells.end) return;
        boundary.for_each_patch([&](auto range, const auto& condition) {
            if (range.begin == range.end) return;
            using C = std::remove_cvref_t<decltype(condition)>;
            using Faces = detail::BoundaryResidualFaces<Space, Real, Accumulator, Output, decltype(physics),
                Model, C, Closure>;
            Kokkos::parallel_for("euler/boundary_residual", FacePolicy(exec, range.begin, range.end),
                Faces{boundary.geometry_view(), output, physics, boundary.model(), condition, closure, cells, time});
        });
    }

    template<class Reconstruction,class Output,class Boundary,class Closure=PreparedExterior>
    void add_residual_async(Reconstruction reconstruction,Output output,const Boundary& boundary,
        AssemblyStatus<Exec>& status,Closure closure={},Real time=0,CellRange cells={})const {
        // Validate the complete request before submitting either contribution.
        validate(reconstruction,output,true);validate_boundary(boundary);
        mpi::require(std::isfinite(time),"Euler: time must be finite");
        cells=cell_range(cells);
        add_inner_residual_async(reconstruction,output,status,{},cells);
        add_boundary_residual_async(reconstruction,output,boundary,status,closure,time,cells);
    }

    /// @brief Adds the compact inner-face Jacobian of outward flux to owned BSR rows.
    /// Primitives are cell-centred. Face and cell ranges match the split residual interface.
    template<class Target, class Primitives, class Linearization>
    void add_inner_jacobian_async(Target target, Primitives values, AssemblyStatus<Exec>& status,
        Linearization linearization, FaceRange faces = {}, CellRange cells = {}) const {
        validate_jacobian_target(target.blocks, target.entries);
        detail::validate_pack<Exec, false>(values, static_cast<std::size_t>(plan_.mesh_view().sizes.n_cells));
        submit_inner_jacobian(target, values, status, linearization, inner_range(faces), cell_range(cells));
    }

    /// @brief Adds physical boundary blocks to owned diagonals using the selected closure.
    /// Use the same reconstruction, closure and time as for the boundary residual.
    template<class Target, class Reconstruction, class Boundary, class Closure,
             class Linearization = FiniteDifferenceBoundary>
    void add_boundary_jacobian_async(Target target, Reconstruction reconstruction, const Boundary& boundary,
        AssemblyStatus<Exec>& status, Closure closure, Real time = 0, CellRange cells = {},
        Linearization linearization = {}) const {
        validate_jacobian_target(target.blocks, target.entries);
        validate_jacobian_boundary(reconstruction, boundary, time);
        submit_boundary_jacobian(target, reconstruction, boundary, status, closure, time, cell_range(cells), linearization);
    }

    /// @brief Adds both inner and physical boundary blocks. Leaves existing coefficients in place.
    template<class Target, class Reconstruction, class Boundary, class InnerLinearization, class Closure,
             class BoundaryLinearization = FiniteDifferenceBoundary>
    void add_jacobian_async(Target target, Reconstruction reconstruction, const Boundary& boundary,
        AssemblyStatus<Exec>& status, InnerLinearization inner, Closure closure, Real time = 0,
        CellRange cells = {}, BoundaryLinearization outer = {}) const {
        validate_jacobian_target(target.blocks, target.entries);
        validate_jacobian_boundary(reconstruction, boundary, time);
        cells = cell_range(cells);
        submit_inner_jacobian(target, reconstruction.values(), status, inner, inner_range({}), cells);
        submit_boundary_jacobian(target, reconstruction, boundary, status, closure, time, cells, outer);
    }

    [[nodiscard]] const plan_type& plan()const noexcept{return plan_;}
    [[nodiscard]] const Exec& execution_space_instance()const noexcept{return plan_.execution_space_instance();}
private:
    plan_type plan_;
    StateLaw law_;
    Solver solver_;
    FaceRange inner_range(FaceRange range) const
    {
        const auto count = plan_.mesh_view().sizes.n_inner_faces;
        if (range.end == -1) range.end = count;
        mpi::require(range.begin >= 0 && range.end >= range.begin && range.end <= count, "Euler: invalid inner face range");
        return range;
    }

    template<class Blocks, class Entries>
    void validate_jacobian_target(const Blocks& blocks, const Entries& entries) const
    {
        using Value = typename Blocks::value_type;
        static_assert(Blocks::rank == 3);
        static_assert(std::is_floating_point_v<Value> && !std::is_const_v<Value>);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Blocks::memory_space>::accessible
                      && Kokkos::SpaceAccessibility<Exec, typename Entries::memory_space>::accessible);
        const auto sizes = plan_.mesh_view().sizes;
        mpi::require(blocks.extent(1) == 5 && blocks.extent(2) == 5
                     && entries.diagonal.extent(0) == static_cast<std::size_t>(sizes.n_own)
                     && entries.owner_entries.extent(0) == static_cast<std::size_t>(sizes.n_inner_faces)
                     && entries.neighbour_entries.extent(0) == static_cast<std::size_t>(sizes.n_internal_faces),
                     "Euler: Jacobian target dimensions do not match the mesh");
    }

    template<class Reconstruction, class Boundary>
    void validate_jacobian_boundary(const Reconstruction& reconstruction, const Boundary& boundary, Real time) const
    {
        using Model = typename Boundary::model_type;
        static_assert(Model::compressible && Model::thermal, "Euler requires a compressible thermal boundary model");
        const auto sizes = plan_.mesh_view().sizes;
        const auto cells = static_cast<std::size_t>(sizes.n_cells);
        reconstruction.template validate<Exec>(cells, cells + static_cast<std::size_t>(sizes.n_faces - sizes.n_inner_faces), true);
        validate_boundary(boundary);
        mpi::require(std::isfinite(time), "Euler: time must be finite");
    }

    template<class Target, class Primitives, class Linearization>
    void submit_inner_jacobian(Target target, Primitives values, AssemblyStatus<Exec>& status,
        Linearization linearization, FaceRange faces, CellRange cells) const
    {
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        if (faces.begin == faces.end || cells.begin == cells.end) return;
        using Rows = detail::InnerJacobianRows<Space, Real, Accumulator, Target, Primitives, StateLaw, Linearization,
                                               decltype(flags)>;
        Kokkos::parallel_for("euler/inner_jacobian", Policy(exec, cells.begin, cells.end),
            Rows{detail::row_data(plan_.mesh_view(), plan_.connectivity_view()), target, values, law_, linearization, flags, faces});
    }

    template<class Target, class Reconstruction, class Boundary, class Closure, class Linearization>
    void submit_boundary_jacobian(Target target, Reconstruction reconstruction, const Boundary& boundary,
        AssemblyStatus<Exec>& status, Closure closure, Real time, CellRange cells, Linearization linearization) const
    {
        const auto& exec = execution_space_instance();
        const auto flags = status.device_flags(exec);
        if (cells.begin == cells.end) return;
        const auto physics = detail::Physics<Real, Reconstruction, StateLaw, Solver, decltype(flags)>{
            reconstruction, law_, solver_, flags};
        boundary.for_each_patch([&](auto range, const auto& condition) {
            if (range.begin == range.end) return;
            using Condition = std::remove_cvref_t<decltype(condition)>;
            using Faces = detail::BoundaryJacobianFaces<Space, Real, Accumulator, Target, decltype(physics),
                typename Boundary::model_type, Condition, Closure, Linearization>;
            Kokkos::parallel_for("euler/boundary_jacobian", FacePolicy(exec, range.begin, range.end),
                Faces{boundary.geometry_view(), target, physics, boundary.model(),
                      condition, closure, linearization, cells, time});
        });
    }

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
        detail::reject_exact_alias(reconstruction.values(),output);
        reconstruction.profiles.visit_inputs([&](const char*, const auto& input, bool = false) {
            if constexpr (requires { input.x; input.y; input.z; }) {
                detail::reject_exact_alias(input.x, output);
                detail::reject_exact_alias(input.y, output);
                detail::reject_exact_alias(input.z, output);
            } else {
                detail::reject_exact_alias(input, output);
            }
        }, with_boundary);
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
        const auto& catalog = plan_.catalog();
        const auto patches = catalog.patches();
        boundary.for_each_patch([&](auto range, const auto&) {
            const auto id = range.patch_id;
            mpi::require(id == -1 || (id >= 0 && static_cast<std::size_t>(id) < patches.size()),
                         "Euler: unknown boundary patch");
            const auto expected = id == -1 ? catalog.unlabelled() : patches[static_cast<std::size_t>(id)].range;
            mpi::require(expected.begin == range.begin && expected.end == range.end,
                         "Euler: boundary catalog differs from plan");
        });
    }
};
template<class E,class R,class L,class S>EulerOperator(EulerPlan<E,R>,L,S)->EulerOperator<L,S,E,R>;
template<class E,class R,class L>EulerOperator(EulerPlan<E,R>,L)->EulerOperator<L,numerics::riemann::Hllc<>,E,R>;
} // namespace krona::transport::euler