#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/linalg/csr_matrix.hpp>
#include <krona/mesh/device/aux_view.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/numerics/reconstruction/stencil_data.hpp>
#include <krona/numerics/time/derivative.hpp>
#include <krona/solvers/scalar/detail/reconstruction.hpp>
#include <krona/solvers/scalar/options.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/transport/scalar/source/operator.hpp>
#include <krona/transport/scalar/transient/accepted_history.hpp>
#include <krona/transport/scalar/transient/detail/validation.hpp>
#include <krona/transport/scalar/transient/operator.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace krona::solvers::scalar {

namespace ts = transport::scalar;
namespace cv = ts::convection;
namespace df = ts::diffusion;
namespace tt = ts::transient;
namespace gr = numerics::gradient;
namespace rec = numerics::reconstruction;
namespace nt = numerics::time;
namespace bc = boundary::scalar;

namespace detail {

template<class Real, class Mass, class Values>
struct SampleMass {
    Mass mass;
    Values values;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        values(i) = tt::detail::mass_value<Real>(mass, i);
    }
};

} // namespace detail

/// @brief Solves one implicit finite-volume scalar equation and writes the borrowed field in place.
/// @tparam Exec Execution space shared by the mesh, boundary manager and solve.
/// @tparam Real Arithmetic type used by the transport and time operators.
/// @tparam Storage Value type of the scalar and its reconstruction fields.
/// @tparam Layout Layout of the borrowed scalar FieldView.
/// @details Owns the matrix, linear solver, halo buffers, reconstruction workspace and
/// accepted time levels. Keep the mesh storage, communicator and scalar allocation alive
/// for this object's lifetime. Call setup and solve methods in the same order on all ranks.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class Storage = StorageReal,
         class Layout = Kokkos::LayoutLeft>
class Solver {
    using Space = typename Exec::memory_space;
    using Mesh = mesh::device::MeshView<Space>;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    using Matrix = linalg::CsrMatrix<SolverReal, Exec>;

public:
    using Field = fields::FieldView<Storage, 1, Space, Layout>;

    /// @brief Builds the reusable system and binds the writable scalar field.
    /// @param host Partition used to build the matrix graph in local cell order.
    /// @param mesh Device mesh for the same partition, with cell/face geometry and halo maps.
    /// @param connectivity Device cell-to-face incidence.
    /// @param geometry Auxiliary metrics required by the selected gradient or diffusion options.
    /// @param halo_plan Host peer ranges associated with the mesh's halo maps.
    /// @param field Scalar values indexed by local cell ID. Include slots for physical boundary
    /// faces after the MPI ghosts when Options::needs_gradients() is true.
    /// @param options Numerical controls and halo tags. Peers must agree on both tags.
    /// @param comm Communicator of the partition.
    /// @param exec Execution instance that will also be used by the boundary manager.
    Solver(const mesh::host::MeshPart& host, Mesh mesh, mesh::device::AuxConnectivityView<Space> connectivity,
           mesh::device::AuxGeometryView<Space> geometry, mesh::host::HaloPlanView halo_plan,
           Field field, Options options, mpi::CommView comm, Exec exec = {})
        : exec_(exec), options_(checked(std::move(options))), field_(field),
          owned_(static_cast<std::size_t>(mesh.sizes.n_own)), sources_(mesh, exec), temporal_(owned_, exec),
          assembly_(host, comm, exec), matrix_(assembly_.pattern()),
          rhs_(matrix_.distribution()), solution_(matrix_.distribution()), linear_(matrix_, options_.linear),
          status_(exec), boundary_status_(exec), field_halo_(mesh, halo_plan, comm, options_.field_tag, exec) {
        const auto cells = static_cast<std::size_t>(mesh.sizes.n_cells);
        if (options_.convection)
            convection_.emplace(cv::ConvectionPlan<Exec, Real>{mesh, connectivity, comm, exec});
        if (options_.diffusion)
            diffusion_.emplace(df::DiffusionPlan<Exec, Real>{mesh, connectivity, geometry, comm, exec,
                                                            options_.diffusion_geometry});

        field_halo_.add_field(field_);
        field_halo_.prepare();

        if (options_.needs_gradients()) {
            gradient_.emplace(mesh, connectivity, geometry, options_.gradient, exec);
            mpi::require(field_.size() >= gradient_->required_input_size(), "Scalar solver: missing boundary slots");
            gradients_.emplace("scalar_solver/gradients", cells, fields::FieldLocation::Cell, exec);
            gradients_->fill_async(exec, Storage{0});
            base_rhs_ = decltype(base_rhs_)("scalar_solver/base_rhs", owned_);
            gradient_halo_.emplace(mesh, halo_plan, comm, options_.gradient_tag, exec);
            gradient_halo_->add_field(gradients_->view());
            if (options_.reconstructs()) {
                stencil_.emplace(mesh, connectivity);
                factors_.emplace("scalar_solver/limiter", cells, fields::FieldLocation::Cell, exec);
                factors_->fill_async(exec, Storage{1});
                if (options_.reconstruction.cell_limiter != ts::CellLimiter::None)
                    gradient_halo_->add_field(factors_->view());
            }
            gradient_halo_->prepare();
        }

        if (options_.time_scheme != ts::TimeScheme::Steady) {
            history_.emplace(owned_, exec);
            mass_values_ = decltype(mass_values_)("scalar_solver/mass", owned_);
        }
    }

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    /// @brief Returns the time at which the next step will evaluate providers and boundary profiles.
    /// A steady solve uses time zero. Transient steps start at zero and use a fixed interval.
    [[nodiscard]] Real next_time() const {
        if (options_.time_scheme == ts::TimeScheme::Steady) return Real{0};
        mpi::require(accepted_steps_ < std::numeric_limits<std::size_t>::max(),
                     "Scalar solver: step counter exhausted");
        const Real next = static_cast<Real>(accepted_steps_ + 1) * static_cast<Real>(options_.time_step);
        mpi::require(std::isfinite(next) && next > time_, "Scalar solver: time cannot advance");
        return next;
    }

    /// @brief Records the initial scalar and integrated cell masses for a transient run.
    /// @tparam Mass Device-callable cell mass provider.
    /// @param mass Device-callable provider of initial rho*V at each owned cell.
    /// The caller initializes the owned scalar values before calling this method.
    template<class Mass>
    void initialize_history(const Mass& mass) {
        mpi::require(history_ && history_->size() == 0, "Scalar solver: expected uninitialized transient history");
        sample_mass(mass);
        history_->accept_step_async(field_.component(0), mass_values_);
        exec_.fence("scalar solver: initial history ready");
    }

    /// @brief Collectively solves the steady equation and fills the owned field.
    /// Returns after all owned values are ready on the plan's execution instance.
    /// @param boundary Finalized scalar boundary manager for this mesh and execution instance.
    /// @param flux Provider of the signed mass flow rate already integrated over each face.
    /// @param law Isotropic diffusion law with a face coefficient provider.
    /// @param source Device-callable provider of source density per owned cell.
    /// @return Evaluation time and the linear iteration count for each outer update.
    template<class Boundary, class Flux, class Law, class Source = ts::UniformSource<Real>>
    [[nodiscard]] StepResult<Real> solve_steady(const Boundary& boundary, const Flux& flux,
                                                const Law& law, const Source& source = {}) {
        mpi::require(!history_, "Scalar solver: solve_steady requires the steady time scheme");
        assemble(boundary, flux, law, source, Real{0}, [](const auto&) {});
        auto result = solve_updates(boundary, flux, law, Real{0});
        exec_.fence("scalar solver: steady solution ready");
        return result;
    }

    /// @brief Collectively advances one fixed step and stores the accepted owned state.
    /// @param boundary Finalized scalar boundary manager for this mesh and execution instance.
    /// @param flux Face mass flow rate provider for next_time().
    /// @param law Isotropic diffusion law for next_time().
    /// @param mass Provider of integrated cell mass at next_time(). Accepted masses are kept with their scalars.
    /// @param source Device-callable source density for next_time().
    /// @return Accepted time and the linear iteration count for each outer update.
    template<class Boundary, class Flux, class Law, class Mass, class Source = ts::UniformSource<Real>>
    [[nodiscard]] StepResult<Real> advance(const Boundary& boundary, const Flux& flux, const Law& law,
                                           const Mass& mass, const Source& source = {}) {
        mpi::require(history_ && history_->size() != 0, "Scalar solver: initialize transient history before advancing");
        const Real next = next_time();
        const Real dt = static_cast<Real>(options_.time_step);
        sample_mass(mass);
        const tt::CellMass current_mass{mass_values_};
        assemble(boundary, flux, law, source, next, [&](const auto& target) {
            if (options_.time_scheme == ts::TimeScheme::Bdf2 && history_->size() == 2)
                temporal_.add_system_async(target, status_, nt::Bdf2{}.coefficients(dt),
                                           current_mass, history_->template history<2>());
            else
                temporal_.add_system_async(target, status_, nt::BackwardEuler{}.coefficients(dt),
                                           current_mass, history_->template history<1>());
        });
        auto result = solve_updates(boundary, flux, law, next);

        // A failed assembly or outer update must not publish a new accepted layer.
        history_->accept_step_async(field_.component(0), mass_values_);
        exec_.fence("scalar solver: accepted step ready");
        ++accepted_steps_;
        time_ = next;
        return result;
    }

private:
    Exec exec_;
    Options options_;
    Field field_;
    std::size_t owned_;
    ts::SourceOperator<Exec, Real> sources_;
    tt::TransientOperator<Exec, Real> temporal_;
    linalg::FaceAssemblyPlan<Exec> assembly_;
    Matrix matrix_;
    linalg::Vector<SolverReal, Exec> rhs_, solution_;
    linalg::Solver<Matrix> linear_; ///< Borrows matrix_, which is destroyed after the solver.
    ts::AssemblyStatus<Exec> status_;
    bc::Status<Exec> boundary_status_;
    halo::Exchange<Storage, Exec> field_halo_;
    std::optional<cv::ConvectionOperator<Exec, Real>> convection_;
    std::optional<df::DiffusionOperator<Exec, Real>> diffusion_;
    std::optional<gr::GradientOperator<Exec>> gradient_;
    std::optional<rec::StencilData<Space, Real>> stencil_;
    std::optional<fields::FieldStorage<Storage, 3, Exec>> gradients_;
    std::optional<fields::FieldStorage<Storage, 1, Exec>> factors_;
    std::optional<halo::Exchange<Storage, Exec>> gradient_halo_;
    Kokkos::View<SolverReal*, Space> base_rhs_;
    std::optional<tt::AcceptedHistory<Storage, Exec, Real>> history_;
    Kokkos::View<Real*, Space> mass_values_;
    std::size_t accepted_steps_ = 0;
    Real time_ = 0;

    /// @brief Validates public options independently of the application config parser.
    static Options checked(Options options) {
        const Real dt = static_cast<Real>(options.time_step);
        const Real relaxation = static_cast<Real>(options.outer.relaxation);
        mpi::require(options.time_scheme == ts::TimeScheme::Steady
                         ? options.convection || options.diffusion
                         : (options.time_scheme == ts::TimeScheme::BackwardEuler ||
                            options.time_scheme == ts::TimeScheme::Bdf2) && std::isfinite(dt) && dt > Real{0},
                     "Scalar solver: invalid time scheme or step size");
        mpi::require(options.outer.iterations > 0 && std::isfinite(relaxation) &&
                         relaxation > Real{0} && relaxation <= Real{1},
                     "Scalar solver: invalid outer iteration controls");
        mpi::require(!options.nonorthogonal_correction || options.diffusion,
                     "Scalar solver: nonorthogonal correction requires diffusion");
        if (options.reconstructs()) {
            const auto& reconstruction = options.reconstruction;
            mpi::require(reconstruction.method == ts::ReconstructionMethod::Linear ||
                         reconstruction.method == ts::ReconstructionMethod::MusclKappa ||
                         reconstruction.method == ts::ReconstructionMethod::Tvd,
                         "Scalar solver: invalid reconstruction method");
            if (reconstruction.method == ts::ReconstructionMethod::MusclKappa)
                mpi::require(std::abs(reconstruction.kappa) <= 1, "Scalar solver: invalid MUSCL kappa");
            mpi::require(reconstruction.cell_limiter == ts::CellLimiter::None ||
                         reconstruction.cell_limiter == ts::CellLimiter::BarthJespersen ||
                         reconstruction.cell_limiter == ts::CellLimiter::Venkatakrishnan,
                         "Scalar solver: invalid cell limiter");
            if (reconstruction.cell_limiter == ts::CellLimiter::Venkatakrishnan) {
                const auto epsilon = static_cast<Real>(reconstruction.regularization);
                mpi::require(std::isfinite(epsilon) && epsilon >= Real{0},
                             "Scalar solver: invalid limiter regularization");
            }
        }
        return options;
    }

    template<class Mass>
    void sample_mass(Mass mass) {
        tt::detail::validate_mass<Exec>(mass, owned_);
        mpi::guarded([&] {
            Kokkos::parallel_for("scalar_solver/sample_mass", Policy(exec_, 0, owned_),
                detail::SampleMass<Real, Mass, decltype(mass_values_)>{mass, mass_values_});
        });
    }

    template<class Boundary, class Flux, class Law, class Source, class TimeTerm>
    void assemble(const Boundary& boundary, const Flux& flux, const Law& law, const Source& source,
                  Real time, const TimeTerm& add_time);

    template<class Boundary, class Flux, class Law>
    StepResult<Real> solve_updates(const Boundary& boundary, const Flux& flux, const Law& law, Real time);

    [[nodiscard]] int solve_linear_update();
};

} // namespace krona::solvers::scalar

#include <krona/solvers/scalar/detail/solve_step.ipp>