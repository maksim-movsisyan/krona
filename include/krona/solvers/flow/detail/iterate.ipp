namespace krona::solvers::flow {

template<class Exec, class Real, class Properties, class EOS, class Storage>
template<class Boundary>
IterationResult Solver<Exec, Real, Properties, EOS, Storage>::iterate(const Boundary& boundary) {
    detail::IterationTimer timer{options_.measure_iteration_time};
    IterationResult iteration;
    assemble_residual(boundary, iteration, timer);
    finish_iteration(boundary, iteration, timer, true);
    return iteration;
}

template<class Exec, class Real, class Properties, class EOS, class Storage>
template<class Boundary, class Observe>
SolveResult Solver<Exec, Real, Properties, EOS, Storage>::solve_steady(const Boundary& boundary,
                                                                       const Observe& observe) {
    const auto& control = options_.pseudo_time;
    SolveResult result;
    while (true) {
        detail::IterationTimer timer{options_.measure_iteration_time};
        IterationResult iteration;
        assemble_residual(boundary, iteration, timer);
        result.final_residual = iteration.residual_norm;
        if (result.updates == 0) result.initial_residual = result.final_residual;
        const double tolerance = std::max(control.absolute_tolerance, control.relative_tolerance * result.initial_residual);
        result.converged = result.final_residual <= tolerance;
        const bool update = !result.converged && result.updates < control.maximum_iterations;
        finish_iteration(boundary, iteration, timer, update);
        observe(result.updates, iteration);
        if (!update) return result;
        ++result.updates;
    }
}

template<class Exec, class Real, class Properties, class EOS, class Storage>
template<class Boundary>
void Solver<Exec, Real, Properties, EOS, Storage>::finish_iteration(const Boundary& boundary, 
                                                                    IterationResult& result, 
                                                                    detail::IterationTimer& timer, 
                                                                    bool update) {
    if (update) assemble_jacobian(boundary);
    result.euler = status_.check();
    if (viscous_status_) viscous_status_->check();
    if (update) {
        matrix_.commit();
        result.timings.jacobian = timer.lap();
        solve_linear_update(result, timer);
    }
    result.timings.total = timer.total();
}

template<class Exec, class Real, class Properties, class EOS, class Storage>
void Solver<Exec, Real, Properties, EOS, Storage>::solve_linear_update(IterationResult& result, 
                                                                       detail::IterationTimer& timer) {
    const auto solved = linear_.solve_from_zero(rhs_, increment_, result.residual_norm, exec_);
    mpi::require(solved.converged, "Steady flow: linear solve did not converge");
    result.linear_iterations = solved.iterations;
    result.timings.linear_solve = timer.lap();
    detail::update_state<Real, Storage>(exec_, conserved_, increment_.read(exec_),
        detail::primitive_views(field_), eos_, static_cast<Real>(options_.pseudo_time.relaxation));
    increment_.commit();
    result.timings.update_state = timer.lap();
}

} // namespace krona::solvers::flow
