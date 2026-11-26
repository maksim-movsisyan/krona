namespace krona::solvers::flow {

namespace ct = transport::compressible;

template<class Exec, class Real, class Properties, class EOS, class Storage, class Layout>
Solver<Exec, Real, Properties, EOS, Storage, Layout>::Solver(
    const mesh::host::MeshPart& host, Mesh mesh, Connectivity connectivity, mesh::host::HaloPlanView halo_plan,
    ConservativeView conserved, Field primitives, EOS eos, Properties properties, Options options,
    mpi::CommView comm, Exec exec)
    : exec_(exec), options_(checked(std::move(options))), mesh_(mesh), connectivity_(connectivity),
      conserved_(conserved), field_(primitives), eos_(eos), properties_(properties),
      euler_(eu::EulerPlan<Exec, Real>{host, mesh, connectivity, comm, exec},
             eu::Thermodynamics<EOS>{eos}, detail::NumericalFlux{options_.flux}),
      stencil_(mesh, connectivity), assembly_(host, comm, exec), matrix_(assembly_.pattern(), 5),
      rhs_(matrix_.distribution()), increment_(matrix_.distribution()), linear_(matrix_, options_.linear),
      residual_("flow_solver/residual", static_cast<std::size_t>(mesh.sizes.n_own), fields::FieldLocation::OwnCell, exec),
      status_(exec), field_halo_(mesh, halo_plan, comm, options_.field_tag, exec) {
    const std::size_t owned = static_cast<std::size_t>(mesh.sizes.n_own);
    const std::size_t cells = static_cast<std::size_t>(mesh.sizes.n_cells);
    mpi::require(conserved_.extent(0) == owned && (owned == 0 || conserved_.data() != nullptr),
                 "Steady flow: conservative state must contain exactly the owned cells");
    mpi::require(field_.size() >= cells, "Steady flow: primitive field lacks real cells");
    field_halo_.add_field(field_);
    field_halo_.prepare();

    if (options_.viscous_enabled) {
        const auto cv = static_cast<Real>(eos_.cv());
        mpi::require(std::isfinite(cv) && cv > Real{0}, "Steady flow: heat capacity must be finite and positive");
        viscous_.emplace(euler_.plan(), properties_, eos_, static_cast<Real>(options_.bulk_viscosity));
        viscous_status_.emplace(exec_);
    }

    if (options_.needs_gradients()) {
        gradient_.emplace(mesh_, connectivity_, mesh::device::AuxGeometryView<Space>{}, gr::Options{}, exec_);
        mpi::require(field_.size() >= gradient_->required_input_size(), "Steady flow: missing primitive boundary slots");
        boundary_status_.emplace(exec_);
        gradients_.emplace("flow_solver/gradients", cells, fields::FieldLocation::Cell, exec_);
        gradient_halo_.emplace(mesh_, halo_plan, comm, options_.gradient_tag, exec_);
        gradient_halo_->add_field(gradients_->view());
        if (options_.reconstruction.cell_limiter == ct::CellLimiter::BarthJespersen) {
            factors_.emplace("flow_solver/limiter", cells, fields::FieldLocation::Cell, exec_);
            gradient_halo_->add_field(factors_->view());
        }
        gradient_halo_->prepare();
    }
}

template<class Exec, class Real, class Properties, class EOS, class Storage, class Layout>
Options Solver<Exec, Real, Properties, EOS, Storage, Layout>::checked(Options options) {
    const auto& control = options.pseudo_time;
    constexpr double limit = static_cast<double>(std::numeric_limits<Real>::max());
    const auto positive_real = [](double value) {
        return value > 0 && value <= limit && static_cast<Real>(value) > Real{0};
    };
    mpi::require(positive_real(control.cfl) && positive_real(control.relaxation) && control.relaxation <= 1,
                 "Steady flow: require representable CFL > 0 and relaxation in (0, 1]");
    mpi::require(control.maximum_iterations > 0 && std::isfinite(control.absolute_tolerance) &&
                 control.absolute_tolerance >= 0 && control.relative_tolerance >= 0 &&
                 control.relative_tolerance < 1 && (control.absolute_tolerance > 0 || control.relative_tolerance > 0),
                 "Steady flow: invalid residual tolerance or update limit");
    mpi::require(!options.viscous_enabled || (options.bulk_viscosity >= 0 && options.bulk_viscosity <= limit),
                 "Steady flow: bulk viscosity must be finite, nonnegative and representable");

    const auto flux = options.flux;
    mpi::require(flux == ct::Flux::Hllc || flux == ct::Flux::Hll || flux == ct::Flux::Rusanov,
                 "Steady flow: unknown numerical flux");
    const auto& reconstruction = options.reconstruction;
    using Method = ct::ReconstructionMethod;
    mpi::require(reconstruction.method == Method::FirstOrder || reconstruction.method == Method::Linear ||
                 reconstruction.method == Method::MusclKappa || reconstruction.method == Method::Tvd,
                 "Steady flow: unknown reconstruction");
    const auto limiter = reconstruction.cell_limiter;
    mpi::require(limiter == ct::CellLimiter::None ||
                 (limiter == ct::CellLimiter::BarthJespersen && reconstruction.needs_gradients()),
                 "Steady flow: invalid cell limiter for the reconstruction");
    if (reconstruction.method == Method::MusclKappa)
        mpi::require(std::abs(reconstruction.kappa) <= 1, "Steady flow: MUSCL kappa must be in [-1, 1]");
    if (reconstruction.method == Method::Tvd)
        mpi::require(reconstruction.tvd_limiter >= ct::RatioLimiter::Minmod &&
                     reconstruction.tvd_limiter <= ct::RatioLimiter::Koren, "Steady flow: unknown TVD limiter");
    return options;
}

} // namespace krona::solvers::flow
