#pragma once

// Test-only functions, included inside test_diffusion.cpp's anonymous namespace.
// Matrices/vectors are supplied by the caller when testing graph reuse.
template<class Matrix, class BoundaryType, class Law,
         class Source = ts::UniformSource<Real>, class Correction = numerics::gradient::NoCorrection>
void assemble_equation(const Attached& a, Matrix& matrix,
    linalg::Vector<typename Matrix::scalar_type, Exec>& rhs, const BoundaryType& boundary,
    Law law, Source source = {}, Correction correction = {})
{
    using Scalar = typename Matrix::scalar_type;
    const auto& exec = a.plan.execution_space_instance();
    const td::DiffusionOperator<Exec, Real> diffusion(a.plan);
    ts::SourceOperator<Exec, Real> sources(a.mesh.view(), exec);
    ts::AssemblyStatus<Exec> status(exec);
    const auto values = matrix.overwrite_values(exec);
    const auto right = rhs.overwrite(exec);
    const ts::SystemView target{values, Kokkos::subview(right, Kokkos::ALL(), 0), a.assembly.entry_map()};
    Kokkos::deep_copy(exec, target.values, Scalar{0});
    Kokkos::deep_copy(exec, target.rhs, Scalar{0});
    status.begin_async();
    diffusion.add_system_async(target, boundary, status, law, correction);
    sources.add_async(target.rhs, status, source);
    // This fixture solves pure steady diffusion; positive diagonals are its own policy.
    const auto flags = status.device_flags(exec);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("test/check_diffusion_system", Policy(exec, 0, std::max(target.values.extent(0), target.rhs.extent(0))),
        KOKKOS_LAMBDA(std::size_t i) {
            if (i < target.values.extent(0) && !Kokkos::isfinite(target.values(i))) ts::detail::flag(flags, ts::detail::BadSystem);
            if (i < target.rhs.extent(0) && (!Kokkos::isfinite(target.rhs(i)) || !(target.values(target.entries.diagonal(i)) > Scalar{0})))
                ts::detail::flag(flags, ts::detail::BadSystem);
        });
    const auto report = status.check(a.plan.communicator());
    matrix.commit(); rhs.commit();
    mpi::require(matrix.distribution().global_size() == 0 || report.has_diffusion_anchor,
                 "Diffusion test: pure Neumann problem requires an explicit reference");
}

template<class Scalar, class PrecScalar, class Component>
linalg::SolveResult solve_and_publish(linalg::Solver<linalg::CsrMatrix<Scalar, Exec>, PrecScalar>& solver,
    const linalg::Vector<Scalar, Exec>& rhs, linalg::Vector<Scalar, Exec>& solution,
    Component field, const Exec& exec)
{
    solution.copy_from_component_async(field, exec); solution.commit();
    const auto result = solver.solve(rhs, solution);
    if (result.converged) { solution.copy_to_component_async(field, exec); solution.commit(); }
    solution.require_committed();
    return result;
}

template<class Scalar = SolverReal, class PrecScalar = PreconditionerReal,
         class Component, class BoundaryType,
         class Law = transport::IsotropicDiffusion<transport::UniformCoefficient<Real>>,
         class Source = ts::UniformSource<Real>, class Correction = numerics::gradient::NoCorrection>
linalg::SolveResult solve_field(const Attached& a, Component field, const BoundaryType& boundary,
    linalg::SolverOptions options, Law law = {}, Source source = {}, Correction correction = {})
{
    linalg::CsrMatrix<Scalar, Exec> matrix(a.assembly.pattern());
    linalg::Vector<Scalar, Exec> rhs(matrix.distribution()), solution(matrix.distribution());
    linalg::Solver<decltype(matrix), PrecScalar> solver(matrix, std::move(options));
    assemble_equation(a, matrix, rhs, boundary, law, source, correction);
    return solve_and_publish(solver, rhs, solution, field, a.plan.execution_space_instance());
}
