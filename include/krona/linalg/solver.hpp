#pragma once

#include <krona/linalg/preconditioner.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory_Tpetra.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace krona::linalg {

enum class PreconditioningSide { Automatic, Left, Right };

struct SolverOptions {
    // Native Belos names; FGMRES is also accepted as FLEXIBLE GMRES.
    std::string method = "GMRES";
    // Unset chooses 1e-5 for float or 1e-10 for double, using the actual Solver scalar.
    std::optional<double> relative_tolerance;
    double absolute_tolerance = 0;
    int maximum_iterations = 500;
    int restart_length = 50;
    PreconditioningSide side = PreconditioningSide::Automatic;
    Teuchos::ParameterList solver_parameters;
    PreconditionerOptions preconditioner;
};

struct SolveResult {
    // Authoritative test: independently recomputed ||b-Ax||_2 <= residual_threshold.
    bool converged = false;
    bool krylov_converged = false;
    bool iterations_started = false;
    int iterations = 0;
    double residual_norm = 0;
    double initial_residual_norm = 0;
    double rhs_norm = 0;
    double relative_residual = 0;
    double residual_threshold = 0;
};

/// @brief Configurable Belos solve with independent true-residual verification.
/// Collective and blocking; matrix must outlive the solver. Not reentrant.
template<class Matrix, class PrecScalar = PreconditionerReal>
class Solver {
    using Scalar = typename Matrix::scalar_type;
    using Exec = typename Matrix::execution_space;
    using T = Types<Scalar, Exec>;
    using MV = typename T::multivector_type;
    using OP = typename T::operator_type;
    using Problem = Belos::LinearProblem<Scalar, MV, OP>;
public:
    using scalar_type = Scalar;
    using preconditioner_scalar_type = PrecScalar;
    using preconditioner_type = Preconditioner<Matrix, PrecScalar>;
    using manager_type = Belos::SolverManager<Scalar, MV, OP>;

    explicit Solver(const Matrix& matrix, SolverOptions options = {},
                    typename preconditioner_type::Factory custom_factory = {})
        : matrix_(matrix), options_(std::move(options)),
          preconditioner_(matrix, options_.preconditioner, std::move(custom_factory))
    {
        mpi::guarded([&] {
            relative_tolerance_ = options_.relative_tolerance.value_or(std::is_same_v<Scalar, float> ? 1.0e-5 : 1.0e-10);
            mpi::require(std::isfinite(relative_tolerance_) && relative_tolerance_ >= 0 && relative_tolerance_ < 1 &&
                std::isfinite(options_.absolute_tolerance) && options_.absolute_tolerance >= 0 &&
                (relative_tolerance_ > 0 || options_.absolute_tolerance > 0) &&
                options_.maximum_iterations >= 0,
                "Linalg: invalid solver tolerances or iteration limits");
            auto parameters = Teuchos::rcp(new Teuchos::ParameterList(options_.solver_parameters));
            Belos::SolverFactory<Scalar, MV, OP> factory;
            method_ = detail::upper_name(options_.method);
            if (method_ == "FGMRES") method_ = "FLEXIBLE GMRES";
            manager_ = factory.create(method_, parameters);
            const auto valid = manager_->getValidParameters();
            mpi::require(valid->isParameter("Convergence Tolerance") && valid->isParameter("Maximum Iterations"),
                         "Linalg: selected Belos manager does not expose the common stopping controls");
            mpi::require(!valid->isParameter("Num Blocks") || options_.solver_parameters.isParameter("Num Blocks") ||
                         options_.restart_length > 0, "Linalg: nonpositive Krylov restart length");
            problem_ = Teuchos::rcp(new Problem);
            problem_->setOperator(matrix_.native());
            const bool symmetric = method_.ends_with("CG") || method_ == "MINRES";
            problem_->setHermitian(symmetric);
            side_ = options_.side == PreconditioningSide::Automatic ?
                (symmetric ? PreconditioningSide::Left : PreconditioningSide::Right) : options_.side;
            mpi::require(!preconditioner_.enabled() || method_ != "BICGSTAB" || side_ == PreconditioningSide::Right,
                         "Linalg: Belos BICGSTAB requires right preconditioning");
            mpi::require(!preconditioner_.enabled() || method_ != "FLEXIBLE GMRES" || side_ == PreconditioningSide::Right,
                         "Linalg: flexible GMRES requires right preconditioning");
            manager_->setProblem(problem_);
        });
    }
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    /// @brief Optional explicit warm-up; solve also prepares after coefficient revisions.
    void prepare()
    {
        // Let a rebuild release the old hierarchy before allocating a new one.
        problem_->setLeftPrec(Teuchos::null);
        problem_->setRightPrec(Teuchos::null);
        preconditioner_.prepare();
        if (side_ == PreconditioningSide::Left) problem_->setLeftPrec(preconditioner_.native());
        else problem_->setRightPrec(preconditioner_.native());
    }

    [[nodiscard]] preconditioner_type& preconditioner() noexcept { return preconditioner_; }
    [[nodiscard]] const preconditioner_type& preconditioner() const noexcept { return preconditioner_; }
    [[nodiscard]] const Teuchos::RCP<manager_type>& native() const noexcept { return manager_; }

    /// @brief Uses the supplied initial guess; RHS=0 uses ||r_initial|| as the relative reference.
    [[nodiscard]] SolveResult solve(const Vector<Scalar, Exec>& rhs, Vector<Scalar, Exec>& solution)
    {
        matrix_.require_committed(); rhs.require_committed(); solution.require_committed();
        const auto* map = matrix_.distribution().native().getRawPtr();
        mpi::require(rhs.distribution().native().getRawPtr() == map && solution.distribution().native().getRawPtr() == map,
                     "Linalg: solve vectors must share the matrix distribution");
        mpi::require(rhs.native().getRawPtr() != solution.native().getRawPtr(), "Linalg: RHS and solution alias");
        return mpi::guarded([&] {
            SolveResult result;
            result.rhs_norm = static_cast<double>(rhs.native()->norm2());
            mpi::require(std::isfinite(result.rhs_norm), "Linalg: nonfinite RHS norm");
            if (matrix_.distribution().global_size() == 0) {
                result.converged = true;
                return result;
            }
            if (residual_.is_null())
                residual_ = Teuchos::rcp(new typename T::vector_type(matrix_.distribution().native(), false));
            const auto residual_norm = [&] {
                matrix_.native()->apply(*solution.native(), *residual_);
                residual_->update(Scalar{1}, *rhs.native(), Scalar{-1});
                return static_cast<double>(residual_->norm2());
            };
            result.initial_residual_norm = residual_norm();
            mpi::require(std::isfinite(result.initial_residual_norm), "Linalg: nonfinite initial residual");
            const double reference = result.rhs_norm > 0 ? result.rhs_norm : result.initial_residual_norm;
            result.residual_threshold = std::max(options_.absolute_tolerance, relative_tolerance_*reference);
            mpi::require(std::isfinite(result.residual_threshold), "Linalg: stopping threshold overflow");
            const auto record_residual = [&](double norm) {
                result.residual_norm = norm;
                result.relative_residual = reference > 0 ? norm/reference :
                    (norm == 0 ? 0 : std::numeric_limits<double>::infinity());
                result.converged = std::isfinite(norm) && norm <= result.residual_threshold;
            };
            record_residual(result.initial_residual_norm);
            if (result.converged || options_.maximum_iterations == 0) return result;

            prepare();
            problem_->setLHS(solution.native());
            problem_->setRHS(rhs.native());
            // Reuse the true residual instead of doing another initial SpMV.
            problem_->setInitResVec(residual_);
            mpi::require(problem_->setProblem(), "Linalg: Belos rejected the linear problem");
            auto parameters = Teuchos::rcp(new Teuchos::ParameterList(*manager_->getCurrentParameters()));
            const auto valid = manager_->getValidParameters();
            const bool unscaled = valid->isParameter("Implicit Residual Scaling") || valid->isParameter("Residual Scaling");
            const double tolerance = unscaled ? result.residual_threshold :
                result.residual_threshold/result.initial_residual_norm;
            const Scalar native_tolerance = static_cast<Scalar>(tolerance);
            mpi::require(std::isfinite(native_tolerance) && native_tolerance > Scalar{0},
                         "Linalg: stopping tolerance cannot be represented by the solver scalar");
            parameters->set("Convergence Tolerance", native_tolerance);
            parameters->set("Maximum Iterations", options_.maximum_iterations);
            if (valid->isParameter("Implicit Residual Scaling")) parameters->set("Implicit Residual Scaling", "None");
            // CG/BICGSTAB give the legacy name priority when both names occur.
            if (valid->isParameter("Residual Scaling")) parameters->set("Residual Scaling", "None");
            if (valid->isParameter("Explicit Residual Scaling")) parameters->set("Explicit Residual Scaling", "None");
            if (valid->isParameter("Num Blocks") && !options_.solver_parameters.isParameter("Num Blocks"))
                parameters->set("Num Blocks", options_.restart_length);
            if (valid->isParameter("Maximum Restarts") && valid->isParameter("Num Blocks") &&
                !options_.solver_parameters.isParameter("Maximum Restarts")) {
                const int length = parameters->get<int>("Num Blocks");
                mpi::require(length > 0, "Linalg: nonpositive Krylov restart length");
                parameters->set("Maximum Restarts", options_.maximum_iterations/length +
                    static_cast<int>(options_.maximum_iterations%length != 0));
            }
            manager_->setParameters(parameters);
            result.iterations_started = true;
            result.krylov_converged = manager_->solve() == Belos::Converged;
            result.iterations = manager_->getNumIters();
            record_residual(residual_norm());
            Kokkos::fence("Linalg: solve complete");
            return result;
        });
    }

private:
    const Matrix& matrix_;
    SolverOptions options_;
    preconditioner_type preconditioner_;
    Teuchos::RCP<Problem> problem_;
    Teuchos::RCP<manager_type> manager_;
    Teuchos::RCP<typename T::vector_type> residual_;
    std::string method_;
    PreconditioningSide side_ = PreconditioningSide::Right;
    double relative_tolerance_ = 0;
};

} // namespace krona::linalg