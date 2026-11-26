namespace krona::solvers::flow {

template<class Exec, class Real, class Properties, class EOS, class Storage, class Layout>
template<class Output>
void Solver<Exec, Real, Properties, EOS, Storage, Layout>::add_inner_residual(Output output) {
    const auto values = fields::pack(field_);
    if (!options_.reconstruction.needs_gradients()) {
        if (gradient_halo_) gradient_halo_->exchange();
        const auto profiles = rec::face_profiles(stencil_, values, rec::FirstOrder{});
        euler_.add_inner_residual_async(eu::Reconstruction{profiles}, output, status_);
        return;
    }

    const auto gradients = rec::gradient_pack(gradients_->view());
    detail::with_high_order_formula<Real>(options_.reconstruction, gradients, [&](auto formula) {
        const auto profiles = rec::face_profiles(stencil_, values, formula, rec::Linear{gradients});
        if (factors_) {
            const auto factors = fields::pack(factors_->view());
            rec::compute_limiters_async<Real>(exec_, profiles, factors);
            gradient_halo_->exchange();
            euler_.add_inner_residual_async(eu::Reconstruction{rec::limited(profiles, factors)}, output, status_);
        } else {
            gradient_halo_->exchange();
            euler_.add_inner_residual_async(eu::Reconstruction{profiles}, output, status_);
        }
    });
}

// Physical faces use the same linear profile in the residual and boundary Jacobian.
template<class Exec, class Real, class Properties, class EOS, class Storage, class Layout>
template<class Visit>
void Solver<Exec, Real, Properties, EOS, Storage, Layout>::with_boundary_reconstruction(const Visit& visit) const {
    const auto values = fields::pack(field_);
    if (!options_.reconstruction.needs_gradients())
        return visit(eu::Reconstruction{rec::face_profiles(stencil_, values, rec::FirstOrder{})});

    const auto linear = rec::Linear{rec::gradient_pack(gradients_->view())};
    const auto profiles = rec::face_profiles(stencil_, values, linear);
    if (factors_)
        return visit(eu::Reconstruction{rec::limited(profiles, fields::pack(factors_->view()))});
    visit(eu::Reconstruction{profiles});
}

template<class Exec, class Real, class Properties, class EOS, class Storage, class Layout>
template<class Boundary>
double Solver<Exec, Real, Properties, EOS, Storage, Layout>::assemble_residual(const Boundary& boundary) {
    mpi::require(boundary.execution_space() == exec_, "Steady flow: boundary uses another execution instance");

    // Boundary samples are prepared before the gradients and limiters read them.
    field_halo_.exchange();
    if (gradient_)
        detail::prepare_gradients(boundary, detail::primitive_views(field_), *gradient_, field_,
                                  gradients_->view(), properties_, options_.viscous_enabled, *boundary_status_);

    residual_.fill_async(exec_, AccumulationReal{0});
    const auto output = fields::pack(residual_.view());
    status_.begin_async();
    add_inner_residual(output);
    with_boundary_reconstruction([&](const auto& reconstruction) {
        euler_.add_boundary_residual_async(reconstruction, output, boundary, status_, eu::EvaluatedExterior<>{});
    });

    if (viscous_) {
        viscous_status_->begin_async();
        viscous_->add_residual_async(fields::pack(field_), fields::pack(gradients_->view()), output, boundary,
                                     *viscous_status_);
    }
    linalg::copy_from_fields(rhs_, output, exec_);
    const auto norm = static_cast<double>(rhs_.native()->norm2());
    mpi::require(std::isfinite(norm), "Steady flow: nonfinite spatial residual norm");
    return norm;
}

template<class Exec, class Real, class Properties, class EOS, class Storage, class Layout>
template<class Boundary>
void Solver<Exec, Real, Properties, EOS, Storage, Layout>::assemble_jacobian(const Boundary& boundary) {
    const auto target = transport::make_block_assembly_view(matrix_, assembly_, exec_);
    Kokkos::deep_copy(exec_, target.blocks, SolverReal{0});

    // The compact implicit stencil uses cell-centred values on inner faces.
    const auto values = fields::pack(field_);
    euler_.add_inner_jacobian_async(target, values, status_, eu::FrozenRusanov{eos_});
    with_boundary_reconstruction([&](const auto& reconstruction) {
        euler_.add_boundary_jacobian_async(target, reconstruction, boundary, status_, eu::EvaluatedExterior<>{});
    });

    const auto primitive = detail::primitive_views(field_);
    const auto cfl = static_cast<Real>(options_.pseudo_time.cfl);
    if (viscous_) {
        viscous_->add_jacobian_async(values, fields::pack(gradients_->view()), target, boundary, *viscous_status_);
        const detail::DiffusiveScale scale{properties_, static_cast<Real>(eos_.cv()),
                                           static_cast<Real>(options_.bulk_viscosity)};
        detail::add_pseudo_time_diagonal(exec_, mesh_, connectivity_, target, primitive, eos_, cfl, scale);
    } else {
        detail::add_pseudo_time_diagonal(exec_, mesh_, connectivity_, target, primitive, eos_, cfl);
    }
}

} // namespace krona::solvers::flow
