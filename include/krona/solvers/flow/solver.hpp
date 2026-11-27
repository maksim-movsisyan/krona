#pragma once

#include <krona/boundary/physical/status.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/linalg/bsr_matrix.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/numerics/reconstruction/stencil_data.hpp>
#include <krona/solvers/flow/detail/flux.hpp>
#include <krona/solvers/flow/detail/iteration_timer.hpp>
#include <krona/solvers/flow/detail/prepare_gradients.hpp>
#include <krona/solvers/flow/detail/pseudo_time.hpp>
#include <krona/solvers/flow/detail/reconstruction.hpp>
#include <krona/solvers/flow/detail/residual_norms.hpp>
#include <krona/solvers/flow/detail/state.hpp>
#include <krona/solvers/flow/options.hpp>
#include <krona/transport/block_assembly.hpp>
#include <krona/transport/detail/component_pack.hpp>
#include <krona/transport/euler/operator.hpp>
#include <krona/transport/viscous/operator.hpp>

#include <optional>
#include <utility>

namespace krona::solvers::flow {

namespace eu = transport::euler;
namespace vi = transport::viscous;
namespace gr = numerics::gradient;
namespace rec = numerics::reconstruction;
namespace ph = boundary::physical;

/// @brief Solves the five steady compressible equations on a partitioned static mesh.
/// @tparam Exec Execution space shared with the boundary manager.
/// @tparam Real Arithmetic type for the state, face physics and local pseudo-time step.
/// @tparam Properties Device-copyable model of shear viscosity and thermal conductivity.
/// @tparam EOS Device-copyable calorically perfect equation of state. Supplies checked
/// acoustic conversions in (p,T) and (rho,e), rho-e derivatives and constant cv().
/// @tparam Storage Value type of the primitive and reconstruction fields.
/// @details Owns the compact block system, linear solver and reconstruction workspace.
/// Keep the communicator, mesh and fields alive. Call collective methods in the same order
/// on every rank. Boundary profiles are evaluated at physical time zero.
template<class Exec, class Real, class Properties, class EOS = thermodynamics::IdealGas<Real>,
         class Storage = StorageReal>
class Solver {
    using Space = typename Exec::memory_space;
    using Mesh = mesh::device::MeshView<Space>;
    using Connectivity = mesh::device::AuxConnectivityView<Space>;
    using Geometry = mesh::device::AuxGeometryView<Space>;
    using Matrix = linalg::BsrMatrix<SolverReal, Exec>;
    using Euler = eu::EulerOperator<eu::Thermodynamics<EOS>, detail::NumericalFlux, Exec, Real>;
    using Viscous = vi::ViscousOperator<Exec, Real, Properties, EOS>;

    // A pack's type depends on the component descriptors, regardless of the input field layout.
    template<class T, int N>
    using Pack = decltype(fields::pack(std::declval<fields::FieldView<T, N, Space>>()));
    using FactorStorage = fields::FieldStorage<Storage, 5, Exec>;

public:
    using PrimitivePack = Pack<Storage, 5>;
    using ConservativePack = Pack<Real, 5>;
    using GradientPack = Pack<Storage, 15>;

    /// @brief Binds the state fields and builds the reusable block system and halo exchanges.
    /// @param host Host partition used to build the sparsity pattern and patch catalog.
    /// @param mesh Device view of the same partition.
    /// @param connectivity Cell-to-face incidence in device memory.
    /// @param geometry Auxiliary metrics required by the chosen gradient scheme.
    /// @param halo_plan Peer ranges of the mesh's cell halo.
    /// @param conserved Writable owned-cell components rho,rho*u,rho*v,rho*w,rho*E.
    /// @param primitives Writable components p,u,v,w,T, including ghosts and, when using
    /// gradients, the physical boundary slots.
    /// @param gradients Writable owned and ghost derivatives ordered as 3*component + axis.
    /// Required when options.needs_gradients() is true. Field allocations must be disjoint.
    /// @param eos Equation of state consistent with the physical boundary model.
    /// @param properties Molecular transport model. Evaluated when viscous transport is enabled.
    /// @param options Numerical settings shared by all ranks.
    /// @param comm Communicator of the partition.
    /// @param exec Execution instance also used by the boundary manager.
    Solver(const mesh::host::MeshPart& host, Mesh mesh, Connectivity connectivity, Geometry geometry,
           mesh::host::HaloPlanView halo_plan, ConservativePack conserved, PrimitivePack primitives,
           std::optional<GradientPack> gradients, EOS eos, Properties properties, Options options,
           mpi::CommView comm, Exec exec = {});

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    /// @brief Initializes the owned primitive and conservative values from a device-callable sampler.
    /// @param initial Callable returning Primitive<Real, true> from the three cell-centre coordinates.
    /// Before iterating, use this method or initialize both borrowed fields consistently.
    template<class Initial>
    void initialize(Initial initial) const {
        detail::initialize_state<Real, Storage>(exec_, mesh_, conserved_,
                                                detail::primitive_views(field_), eos_, initial);
    }

    /// @brief Performs one pseudo-time update, including one linear solve.
    /// @param boundary Finalized physical boundary manager using the same mesh, EOS and execution instance.
    /// @return Residual norms and diagnostics of the input iterate, linear iteration count and timings.
    /// On return, the owned conservative and primitive fields contain the updated state.
    /// Prepared gradients belong to the input state used during assembly.
    template<class Boundary>
    [[nodiscard]] IterationResult iterate(const Boundary& boundary);

    /// @brief Iterates until the spatial residual meets tolerance or the update limit is reached.
    /// @param boundary Finalized physical boundary manager as in iterate().
    /// @param observe Host callback invoked on every rank after evaluating an iterate, with its
    /// zero-based index and report. The last report has no linear solve.
    /// @return The convergence status, update count and initial and final residual norms.
    /// Prepared gradients correspond to the state of the final residual evaluation.
    template<class Boundary, class Observe>
    [[nodiscard]] SolveResult solve_steady(const Boundary& boundary, const Observe& observe);

    /// @brief Solves to the configured stopping criterion without an iteration callback.
    template<class Boundary>
    [[nodiscard]] SolveResult solve_steady(const Boundary& boundary) {
        return solve_steady(boundary, [](int, const IterationResult&) {});
    }

private:
    Exec exec_;
    Options options_;
    Mesh mesh_;
    Connectivity connectivity_;
    ConservativePack conserved_;
    PrimitivePack field_;
    EOS eos_;
    Properties properties_;
    Euler euler_;
    std::optional<Viscous> viscous_; ///< Borrows the plan held by euler_.
    rec::StencilData<Space, Real> stencil_;
    linalg::FaceAssemblyPlan<Exec> assembly_;
    Matrix matrix_;
    linalg::Vector<SolverReal, Exec> rhs_, increment_;
    linalg::Solver<Matrix> linear_; ///< Borrows matrix_.
    fields::FieldStorage<AccumulationReal, 5, Exec> residual_;
    eu::AssemblyStatus<Exec> status_;
    std::optional<vi::AssemblyStatus<Exec>> viscous_status_;
    std::optional<ph::Status<Exec>> boundary_status_;
    std::optional<gr::GradientOperator<Exec>> gradient_;
    std::optional<GradientPack> gradients_;
    std::optional<FactorStorage> factors_;
    halo::Exchange<Storage, Exec> field_halo_;
    std::optional<halo::Exchange<Storage, Exec>> gradient_halo_;

    static Options checked(Options options);

    template<class Boundary>
    void assemble_residual(const Boundary& boundary, IterationResult& result, detail::IterationTimer& timer);

    template<class Boundary>
    void finish_iteration(const Boundary& boundary, IterationResult& result, detail::IterationTimer& timer, bool update);

    template<class Boundary>
    void assemble_jacobian(const Boundary& boundary);

    template<class Output>
    void add_inner_residual(Output output);

    template<class Visit>
    void with_boundary_reconstruction(const Visit& visit) const;

    void solve_linear_update(IterationResult& result, detail::IterationTimer& timer);
};

} // namespace krona::solvers::flow

#include <krona/solvers/flow/detail/setup.ipp>
#include <krona/solvers/flow/detail/assembly.ipp>
#include <krona/solvers/flow/detail/iterate.ipp>
