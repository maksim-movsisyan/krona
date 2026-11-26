#pragma once

#include <krona/boundary/physical/status.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/field_transfer.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/numerics/reconstruction/stencil_data.hpp>
#include <krona/solvers/flow/detail/flux.hpp>
#include <krona/solvers/flow/detail/prepare_gradients.hpp>
#include <krona/solvers/flow/detail/pseudo_time.hpp>
#include <krona/solvers/flow/detail/reconstruction.hpp>
#include <krona/solvers/flow/detail/state.hpp>
#include <krona/solvers/flow/options.hpp>
#include <krona/transport/block_assembly.hpp>
#include <krona/transport/euler/operator.hpp>
#include <krona/transport/viscous/operator.hpp>

#include <optional>

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
/// @tparam Layout Layout of the caller's primitive field.
/// @details Owns the compact block system, linear solver and reconstruction workspace.
/// Keep the communicator, mesh and fields alive. Call collective methods in the same order
/// on every rank. Boundary profiles are evaluated at physical time zero.
template<class Exec, class Real, class Properties, class EOS = thermodynamics::IdealGas<Real>,
         class Storage = StorageReal, class Layout = Kokkos::LayoutLeft>
class Solver {
    using Space = typename Exec::memory_space;
    using Mesh = mesh::device::MeshView<Space>;
    using Connectivity = mesh::device::AuxConnectivityView<Space>;
    using Matrix = linalg::BsrMatrix<SolverReal, Exec>;
    using Euler = eu::EulerOperator<eu::Thermodynamics<EOS>, detail::NumericalFlux, Exec, Real>;
    using Viscous = vi::ViscousOperator<Exec, Real, Properties, EOS>;
    using GradientStorage = fields::FieldStorage<Storage, 15, Exec>;
    using FactorStorage = fields::FieldStorage<Storage, 5, Exec>;

public:
    using Field = fields::FieldView<Storage, 5, Space, Layout>;
    using ConservativeView = Kokkos::View<krona::flow::Conservative<Real>*, Space>;

    /// @brief Binds the state fields and builds the reusable block system and halo exchanges.
    /// @param host Host partition used to build the sparsity pattern and patch catalog.
    /// @param mesh Device view of the same partition.
    /// @param connectivity Cell-to-face incidence in device memory.
    /// @param halo_plan Peer ranges of the mesh's cell halo.
    /// @param conserved Writable conservative state for owned cells only.
    /// @param primitives Writable field ordered as p,u,v,w,T. Include MPI ghosts and, if
    /// options.needs_gradients() is true, physical boundary slots.
    /// @param eos Equation of state consistent with the physical boundary model.
    /// @param properties Molecular transport model. Evaluated when viscous transport is enabled.
    /// @param options Numerical settings shared by all ranks.
    /// @param comm Communicator of the partition.
    /// @param exec Execution instance also used by the boundary manager.
    Solver(const mesh::host::MeshPart& host, Mesh mesh, Connectivity connectivity, mesh::host::HaloPlanView halo_plan,
           ConservativeView conserved, Field primitives, EOS eos, Properties properties, Options options,
           mpi::CommView comm, Exec exec = {});

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    /// @brief Initializes both owned state representations from a device-callable sampler.
    /// @param initial Callable returning Primitive<Real, true> from the three cell-centre coordinates.
    /// Before iterating, use this method or initialize both borrowed fields consistently.
    template<class Initial>
    void initialize(Initial initial) const {
        detail::initialize_state<Real, Storage>(exec_, mesh_, conserved_,
                                                detail::primitive_views(field_), eos_, initial);
    }

    /// @brief Performs one pseudo-time update, including one linear solve.
    /// @param boundary Finalized physical boundary manager using the same mesh, EOS and execution instance.
    /// @return Residual and diagnostics of the input iterate and the linear iteration count.
    /// On return, the owned conservative and primitive fields contain the updated state.
    template<class Boundary>
    [[nodiscard]] IterationResult iterate(const Boundary& boundary);

    /// @brief Iterates until the spatial residual meets tolerance or the update limit is reached.
    /// @param boundary Finalized physical boundary manager as in iterate().
    /// @param observe Host callback invoked on every rank after evaluating an iterate, with its
    /// zero-based index and report. The last report has no linear solve.
    /// @return The convergence status, update count and initial and final residual norms.
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
    ConservativeView conserved_;
    Field field_;
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
    std::optional<GradientStorage> gradients_;
    std::optional<FactorStorage> factors_;
    halo::Exchange<Storage, Exec> field_halo_;
    std::optional<halo::Exchange<Storage, Exec>> gradient_halo_;

    static Options checked(Options options);

    template<class Boundary>
    [[nodiscard]] double assemble_residual(const Boundary& boundary);

    template<class Boundary>
    [[nodiscard]] IterationResult finish_iteration(const Boundary& boundary, double norm, bool update);

    template<class Boundary>
    void assemble_jacobian(const Boundary& boundary);

    template<class Output>
    void add_inner_residual(Output output);

    template<class Visit>
    void with_boundary_reconstruction(const Visit& visit) const;

    [[nodiscard]] int solve_linear_update();
};

} // namespace krona::solvers::flow

#include <krona/solvers/flow/detail/setup.ipp>
#include <krona/solvers/flow/detail/assembly.ipp>
#include <krona/solvers/flow/detail/iterate.ipp>
