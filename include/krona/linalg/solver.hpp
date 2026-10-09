#pragma once

#include <krona/linalg/csr_matrix.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <Ifpack2_Factory.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace krona::linalg {

struct SolverOptions {
    double relative_tolerance = std::is_same_v<SolverReal, float> ? 1.0e-5 : 1.0e-8;
    double absolute_tolerance = 0;
    int maximum_iterations = 500;
    int restart_length = 50;
    // Empty means no preconditioner; RELAXATION defaults to Jacobi.
    std::string preconditioner = "RELAXATION";
    Teuchos::ParameterList preconditioner_parameters;
};

struct SolveResult {
    bool converged = false;
    bool krylov_converged = false;
    int iterations = 0;
    double residual_norm = 0;
    double rhs_norm = 0;
    double relative_residual = 0;
};

/// @brief Restarted GMRES with optional Ifpack2 right preconditioning.
/// The matrix must outlive the solver; construction, prepare and solve are collective.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
class Solver {
    using T = Types<Scalar, Exec>;
    using MV = typename T::multivector_type;
    using OP = typename T::operator_type;
    using Problem = Belos::LinearProblem<Scalar, MV, OP>;
    using Manager = Belos::PseudoBlockGmresSolMgr<Scalar, MV, OP>;
    using Prec = Ifpack2::Preconditioner<Scalar, LocalOrdinal, GlobalOrdinal, typename T::node_type>;
    using RowMatrix = Tpetra::RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, typename T::node_type>;

public:
    explicit Solver(const CsrMatrix<Scalar, Exec>& matrix, SolverOptions options = {})
        : matrix_(matrix), options_(std::move(options))
    {
        detail::guarded([&] {
            detail::require(std::isfinite(options_.relative_tolerance) && options_.relative_tolerance > 0 &&
                options_.relative_tolerance < 1 && std::isfinite(options_.absolute_tolerance) &&
                options_.absolute_tolerance >= 0 && options_.maximum_iterations > 0 && options_.restart_length > 0,
                "Linalg: invalid solver tolerances or iteration limits");
            problem_ = Teuchos::rcp(new Problem);
            problem_->setOperator(matrix_.native());
            manager_ = Teuchos::rcp(new Manager);
            residual_ = Teuchos::rcp(new typename T::vector_type(matrix_.pattern().distribution().native(), false));
        });
    }
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    /// @brief Reuses symbolic setup; recomputes numeric preconditioning after coefficient edits.
    void prepare()
    {
        matrix_.require_committed();
        detail::guarded([&] {
            if (options_.preconditioner.empty()) { prepared_revision_ = matrix_.revision(); prepared_ = true; return; }
            if (preconditioner_.is_null()) {
                preconditioner_ = Ifpack2::Factory::create<RowMatrix>(options_.preconditioner, matrix_.native());
                auto parameters = options_.preconditioner_parameters;
                if (options_.preconditioner == "RELAXATION" && !parameters.isParameter("relaxation: type"))
                    parameters.set("relaxation: type", "Jacobi");
                preconditioner_->setParameters(parameters);
                preconditioner_->initialize();
                problem_->setRightPrec(preconditioner_);
            }
            const int local_dirty = !prepared_ || prepared_revision_ != matrix_.revision() ? 1 : 0;
            int global_dirty = 0;
            Teuchos::reduceAll(*matrix_.native()->getComm(), Teuchos::REDUCE_MAX, 1, &local_dirty, &global_dirty);
            if (global_dirty != 0) preconditioner_->compute();
            prepared_revision_ = matrix_.revision(); prepared_ = true;
            Kokkos::fence("Linalg: preconditioner preparation complete");
        });
    }

    /// @brief Blocking solve with the supplied initial guess and independently checked true residual.
    [[nodiscard]] SolveResult solve(const Vector<Scalar, Exec>& rhs, Vector<Scalar, Exec>& solution)
    {
        matrix_.require_committed(); rhs.require_committed(); solution.require_committed();
        const auto* map = matrix_.pattern().distribution().native().getRawPtr();
        detail::require(rhs.distribution().native().getRawPtr() == map && solution.distribution().native().getRawPtr() == map,
                        "Linalg: solve vectors must share the matrix distribution");
        detail::require(rhs.native().getRawPtr() != solution.native().getRawPtr(), "Linalg: RHS and solution alias");
        return detail::guarded([&] {
            SolveResult result;
            result.rhs_norm = static_cast<double>(rhs.native()->norm2());
            detail::require(std::isfinite(result.rhs_norm), "Linalg: nonfinite RHS norm");
            const double threshold = std::max(options_.absolute_tolerance, options_.relative_tolerance*result.rhs_norm);
            auto check_residual = [&] {
                matrix_.native()->apply(*solution.native(), *residual_);
                residual_->update(Scalar{1}, *rhs.native(), Scalar{-1});
                result.residual_norm = static_cast<double>(residual_->norm2());
                result.relative_residual = result.rhs_norm > 0 ? result.residual_norm/result.rhs_norm :
                    (result.residual_norm == 0 ? 0 : std::numeric_limits<double>::infinity());
                result.converged = std::isfinite(result.residual_norm) && result.residual_norm <= threshold;
            };
            if (matrix_.pattern().distribution().global_size() == 0) {
                result.converged = true; result.krylov_converged = true; return result;
            }
            prepare();
            problem_->setLHS(solution.native());
            problem_->setRHS(rhs.native());
            detail::require(problem_->setProblem(), "Linalg: Belos rejected the linear problem");
            auto parameters = Teuchos::rcp(new Teuchos::ParameterList);
            const double tolerance = result.rhs_norm > 0 ? threshold/result.rhs_norm :
                (options_.absolute_tolerance > 0 ? options_.absolute_tolerance : options_.relative_tolerance);
            const auto native_tolerance = static_cast<Scalar>(tolerance);
            detail::require(std::isfinite(native_tolerance) && native_tolerance > Scalar{0},
                            "Linalg: stopping tolerance cannot be represented by the solver scalar");
            parameters->set("Convergence Tolerance", native_tolerance);
            parameters->set("Maximum Iterations", options_.maximum_iterations);
            parameters->set("Num Blocks", options_.restart_length);
            parameters->set("Implicit Residual Scaling", result.rhs_norm > 0 ? "Norm of RHS" : "None");
            parameters->set("Explicit Residual Scaling", result.rhs_norm > 0 ? "Norm of RHS" : "None");
            manager_->setParameters(parameters);
            manager_->setProblem(problem_);
            result.krylov_converged = manager_->solve() == Belos::Converged;
            result.iterations = manager_->getNumIters();
            check_residual();
            Kokkos::fence("Linalg: solve complete");
            return result;
        });
    }

private:
    const CsrMatrix<Scalar, Exec>& matrix_;
    SolverOptions options_;
    Teuchos::RCP<Problem> problem_;
    Teuchos::RCP<Manager> manager_;
    Teuchos::RCP<Prec> preconditioner_;
    Teuchos::RCP<typename T::vector_type> residual_;
    std::uint64_t prepared_revision_ = 0;
    bool prepared_ = false;
};

} // namespace krona::linalg
