namespace krona::solvers::scalar {

namespace detail {

template<class Real, class Storage, class Field, class Candidate>
struct RelaxSolution {
    Field field;
    Candidate candidate;
    Real relaxation;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        const Real old = static_cast<Real>(field(i));
        field(i) = static_cast<Storage>(old + relaxation * (static_cast<Real>(candidate(i, 0)) - old));
    }
};

} // namespace detail


/// @brief Assembles the base matrix and right-hand side for the scalar equation.
/// Uses upwind convection and two-point diffusion.
/// Higher-order reconstruction and nonorthogonal diffusion corrections are added to the RHS during outer iterations.
template<class Exec, class Real, class Storage, class Layout>
template<class Boundary, class Flux, class Law, class Source, class TimeTerm>
void Solver<Exec, Real, Storage, Layout>::assemble(const Boundary& boundary, const Flux& flux,
                                                   const Law& law, const Source& source,
                                                   Real time, const TimeTerm& add_time) {
    mpi::require(boundary.execution_space() == exec_, "Scalar solver: boundary uses another execution instance");
    const auto target = ts::make_system_view(matrix_, rhs_, assembly_, exec_);
    Kokkos::deep_copy(exec_, target.values, SolverReal{0});
    Kokkos::deep_copy(exec_, target.rhs, SolverReal{0});
    status_.begin_async();
    add_time(target);

    // The implicit convection block uses upwind. Higher-order corrections enter the RHS below.
    if (convection_)
        convection_->add_system_async(target, boundary, status_, flux, cv::NoReconstruction{},
                                      cv::FixedValueInflow{}, time);
    if (diffusion_)
        diffusion_->add_system_async(target, boundary, status_, law, time);
    sources_.add_async(target.rhs, status_, source);

    (void)status_.check();
    if (gradient_) Kokkos::deep_copy(exec_, base_rhs_, target.rhs);
    matrix_.commit();
    rhs_.commit();
}

template<class Exec, class Real, class Storage, class Layout>
template<class Boundary, class Flux, class Law>
StepResult<Real> Solver<Exec, Real, Storage, Layout>::solve_updates(const Boundary& boundary, const Flux& flux,
                                                                    const Law& law, Real time) {
    StepResult<Real> result{time, {}};
    result.linear_iterations.reserve(static_cast<std::size_t>(options_.outer.iterations));
    if (!gradient_) {
        for (int outer = 0; outer < options_.outer.iterations; ++outer)
            result.linear_iterations.push_back(solve_linear_update());
        return result;
    }

    const auto gradients = rec::gradient_pack(gradients_->view());
    const auto correction = gr::GradientCorrection{gr::cell_gradients(gradients_->view())};
    const auto limit = options_.reconstruction.cell_limiter;
    boundary_status_.begin_async();
    status_.begin_async();

    const auto run = [&](const auto& reconstruction, const auto& prepare_factors) {
        for (int outer = 0; outer < options_.outer.iterations; ++outer) {
            field_halo_.exchange();
            const auto fill_boundary = [&](const auto& boundary_correction) {
                if (diffusion_)
                    boundary.apply_async(field_, boundary_status_, law, boundary_correction, time);
                else
                    boundary.apply_async(field_, boundary_status_, bc::NoFluxLaw{}, boundary_correction, time);
            };

            // Physical face slots supply gradient samples and limiter bounds.
            if (options_.nonorthogonal_correction) fill_boundary(correction);
            else fill_boundary(gr::NoCorrection{});

            gradient_->apply_async(field_, gradients_->view());
            if (limit != ts::CellLimiter::None && options_.nonorthogonal_correction) fill_boundary(correction);
            prepare_factors();
            gradient_halo_->exchange();

            const auto rhs = Kokkos::subview(rhs_.overwrite(exec_), Kokkos::ALL(), 0);
            Kokkos::deep_copy(exec_, rhs, base_rhs_);
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(reconstruction)>, cv::NoReconstruction>)
                convection_->add_correction_async(rhs, boundary, status_, flux, reconstruction, time);
            if (options_.nonorthogonal_correction)
                diffusion_->add_correction_async(rhs, boundary, status_, law, correction, time);
            rhs_.commit();
            result.linear_iterations.push_back(solve_linear_update());
        }
    };

    if (stencil_) {
        const auto values = fields::pack(field_);
        const auto factors = fields::pack(factors_->view());
        
        // Dispatch once per step. The selected type is fixed through all its outer updates.
        detail::with_reconstruction<Real>(options_.reconstruction, gradients, [&](auto formula) {
            const auto profiles = rec::face_profiles(*stencil_, values, formula, rec::Linear{gradients});
            const auto reconstruction = cv::make_reconstruction(rec::limited(profiles, factors));

            run(reconstruction, [&] {
                if (limit == ts::CellLimiter::BarthJespersen) {
                    rec::compute_limiters_async<Real>(exec_, profiles, factors);
                } else if (limit == ts::CellLimiter::Venkatakrishnan) {
                    const auto epsilon = static_cast<Real>(options_.reconstruction.regularization);
                    rec::compute_limiters_async<Real>(exec_, profiles, factors,
                        rec::Venkatakrishnan{rec::ConstantRegularization<Real>{epsilon}});
                }
            });
        });
    } else {
        run(cv::NoReconstruction{}, [] {});
    }

    boundary_status_.check();
    (void)status_.check();
    return result;
}

template<class Exec, class Real, class Storage, class Layout>
int Solver<Exec, Real, Storage, Layout>::solve_linear_update() {
    const auto field = field_.component(0);
    solution_.copy_from_component(field, exec_);
    const auto solved = linear_.solve(rhs_, solution_);
    mpi::require(solved.converged, "Scalar transport: linear solve did not converge");

    const auto relaxation = static_cast<Real>(options_.outer.relaxation);
    if (relaxation == Real{1}) {
        solution_.copy_to_component(field, exec_);
    } else {
        const auto candidate = solution_.read(exec_);
        Kokkos::parallel_for("scalar_solver/relax", Policy(exec_, 0, owned_),
            detail::RelaxSolution<Real, Storage, decltype(field), decltype(candidate)>{field, candidate, relaxation});
        solution_.commit();
    }
    return solved.iterations;
}

} // namespace krona::solvers::scalar
