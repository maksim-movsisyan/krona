namespace krona::solvers::flow {

template<class Exec, class Real, class Properties, class EOS, class Storage>
template<class Boundary>
IterationResult Solver<Exec, Real, Properties, EOS, Storage>::iterate(const Boundary& boundary) {
    return finish_iteration(boundary, assemble_residual(boundary), true);
}

template<class Exec, class Real, class Properties, class EOS, class Storage>
template<class Boundary, class Observe>
SolveResult Solver<Exec, Real, Properties, EOS, Storage>::solve_steady(const Boundary& boundary,
                                                                             const Observe& observe) {
    const auto& control = options_.pseudo_time;
    SolveResult result;
    while (true) {
        result.final_residual = assemble_residual(boundary);
        if (result.updates == 0) result.initial_residual = result.final_residual;
        const double tolerance = std::max(control.absolute_tolerance, control.relative_tolerance * result.initial_residual);
        result.converged = result.final_residual <= tolerance;
        const bool update = !result.converged && result.updates < control.maximum_iterations;
        const auto iteration = finish_iteration(boundary, result.final_residual, update);
        observe(result.updates, iteration);
        if (!update) return result;
        ++result.updates;
    }
}

template<class Exec, class Real, class Properties, class EOS, class Storage>
template<class Boundary>
IterationResult Solver<Exec, Real, Properties, EOS, Storage>::finish_iteration(
    const Boundary& boundary, double norm, bool update) {
    if (update) assemble_jacobian(boundary);
    IterationResult result{norm, std::nullopt, status_.check()};
    if (viscous_status_) viscous_status_->check();
    if (update) {
        matrix_.commit();
        result.linear_iterations = solve_linear_update();
    }
    return result;
}

template<class Exec, class Real, class Properties, class EOS, class Storage>
int Solver<Exec, Real, Properties, EOS, Storage>::solve_linear_update() {
    increment_.fill(SolverReal{0}, exec_);
    const auto solved = linear_.solve(rhs_, increment_);
    mpi::require(solved.converged, "Steady flow: linear solve did not converge");
    detail::update_state<Real, Storage>(exec_, conserved_, increment_.read(exec_),
        detail::primitive_views(field_), eos_, static_cast<Real>(options_.pseudo_time.relaxation));
    increment_.commit();
    return solved.iterations;
}

} // namespace krona::solvers::flow
