#pragma once

#include <krona/linalg/csr_matrix.hpp>
#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/detail/matrix_access.hpp>
#include <krona/linalg/detail/precision_operator.hpp>

#include <Ifpack2_Factory.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <cctype>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace krona::linalg {

struct PreconditionerOptions {
    // Empty/NONE: identity; AMG/MUELU: MueLu; otherwise a native Ifpack2 name.
    std::string type = "AMG";
    Teuchos::ParameterList parameters;
};

namespace detail {
inline std::string upper_name(std::string name)
{
    for (auto& c : name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return name;
}

template<class Source, class Destination>
struct ConvertMatrixValues {
    Source source;
    Destination destination;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, std::size_t& bad) const
    {
        using Value = typename Destination::non_const_value_type;
        const Value value = static_cast<Value>(source(i));
        destination(i) = value;
        if (!Kokkos::isfinite(value)) ++bad;
    }
};
} // namespace detail

/// @brief Revision-aware native preconditioning, independent of the Krylov algorithm.
/// Collective prepare; the matrix must outlive this object. Not reentrant.
template<class Matrix, class PrecScalar = PreconditionerReal>
class Preconditioner {
    using Scalar = typename Matrix::scalar_type;
    using Exec = typename Matrix::execution_space;
    using T = Types<Scalar, Exec>;
    using P = Types<PrecScalar, Exec>;
    using IfpackPrec = Ifpack2::Preconditioner<PrecScalar, LocalOrdinal, GlobalOrdinal, typename P::node_type>;
    using AmgPrec = MueLu::TpetraOperator<PrecScalar, LocalOrdinal, GlobalOrdinal, typename P::node_type>;
    using RowMatrix = Tpetra::RowMatrix<PrecScalar, LocalOrdinal, GlobalOrdinal, typename P::node_type>;
public:
    using scalar_type = Scalar;
    using preconditioner_scalar_type = PrecScalar;
    using operator_type = typename T::operator_type;
    using matrix_type = typename Matrix::template native_type_for<PrecScalar>;
    using inner_operator_type = typename P::operator_type;
    // Called initially and after collectively detected coefficient changes.
    // A callback may reuse the previous operator or return a replacement.
    using Factory = std::function<Teuchos::RCP<inner_operator_type>(
        const Teuchos::RCP<const matrix_type>&, const Teuchos::RCP<inner_operator_type>&)>;

    explicit Preconditioner(const Matrix& matrix,
                            PreconditionerOptions options = {}, Factory factory = {})
        : matrix_(matrix), options_(std::move(options)), factory_(std::move(factory)),
          type_(detail::upper_name(options_.type)) {}
    Preconditioner(const Preconditioner&) = delete;
    Preconditioner& operator=(const Preconditioner&) = delete;

    [[nodiscard]] bool enabled() const noexcept
    { return static_cast<bool>(factory_) || (!type_.empty() && type_ != "NONE"); }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] const Teuchos::RCP<operator_type>& native() const
    {
        mpi::require(prepared_, "Linalg: prepare the preconditioner before requesting its operator");
        return operator_;
    }
    [[nodiscard]] const Teuchos::RCP<inner_operator_type>& native_inner() const
    {
        mpi::require(prepared_, "Linalg: preconditioner is not prepared");
        return inner_;
    }
    void invalidate() noexcept { prepared_ = false; }

    void prepare()
    {
        matrix_.require_committed();
        mpi::guarded([&] {
            if (!enabled() || matrix_.distribution().global_size() == 0) {
                prepared_revision_ = matrix_.revision();
                prepared_ = true;
                return;
            }
            const int dirty = !prepared_ || prepared_revision_ != matrix_.revision() ? 1 : 0;
            int any_dirty = 0;
            Teuchos::reduceAll(*matrix_.native()->getComm(), Teuchos::REDUCE_MAX, 1, &dirty, &any_dirty);
            if (!any_dirty) return;

            update_matrix();
            const auto previous = inner_.getRawPtr();
            if (factory_) {
                const Teuchos::RCP<const matrix_type> matrix = preconditioner_matrix_;
                inner_ = factory_(matrix, inner_);
                mpi::require(!inner_.is_null(), "Linalg: preconditioner factory returned null");
            } else if (type_ == "AMG" || type_ == "MUELU") {
                const auto& settings = options_.parameters;
                const bool reuse = settings.isParameter("reuse: type") && settings.get<std::string>("reuse: type") != "none";
                if (amg_.is_null() || !reuse) {
                    operator_ = Teuchos::null;
                    inner_ = Teuchos::null;
                    amg_ = Teuchos::null;
                    auto parameters = amg_parameters();
                    const Teuchos::RCP<const inner_operator_type> matrix = preconditioner_matrix_;
                    amg_ = MueLu::CreateTpetraPreconditioner<PrecScalar, LocalOrdinal, GlobalOrdinal,
                                                          typename P::node_type>(matrix, parameters);
                } else {
                    MueLu::ReuseTpetraPreconditioner<PrecScalar, LocalOrdinal, GlobalOrdinal,
                                                    typename P::node_type>(preconditioner_matrix_, *amg_);
                }
                inner_ = amg_;
            } else {
                if (ifpack_.is_null()) {
                    ifpack_ = Ifpack2::Factory::create<RowMatrix>(type_, preconditioner_matrix_);
                    auto parameters = options_.parameters;
                    if (type_ == "RELAXATION" && !parameters.isParameter("relaxation: type"))
                        parameters.set("relaxation: type", "Jacobi");
                    if constexpr (Matrix::block_storage) {
                        if (type_ == "SCHWARZ") {
                            if (!parameters.isParameter("inner preconditioner name"))
                                parameters.set("inner preconditioner name", "RBILUK");
                            if (!parameters.isParameter("schwarz: overlap level"))
                                parameters.set("schwarz: overlap level", 0);
                            mpi::require(parameters.get<int>("schwarz: overlap level") == 0,
                                "Linalg: Trilinos 17.2.1 block SCHWARZ requires zero overlap");
                        }
                    }
                    ifpack_->setParameters(parameters);
                    ifpack_->initialize();
                }
                ifpack_->compute();
                inner_ = ifpack_;
            }

            const auto& map = matrix_.distribution().native();
            mpi::require(!inner_->getDomainMap().is_null() && !inner_->getRangeMap().is_null(),
                         "Linalg: preconditioner has null maps");
            const bool domain_ok = inner_->getDomainMap()->isSameAs(*map);
            const bool range_ok = inner_->getRangeMap()->isSameAs(*map);
            mpi::require(domain_ok && range_ok,
                "Linalg: preconditioner maps do not match the matrix distribution; use SCHWARZ for local subdomain solvers");
            if constexpr (std::is_same_v<Scalar, PrecScalar>) operator_ = inner_;
            else if (operator_.is_null() || previous != inner_.getRawPtr())
                operator_ = Teuchos::rcp(new detail::PrecisionOperator<Scalar, PrecScalar, Exec>(inner_));
            prepared_revision_ = matrix_.revision();
            prepared_ = true;
            ++generation_;
            Kokkos::fence("Linalg: preconditioner preparation complete");
        });
    }

private:
    const Matrix& matrix_;
    PreconditionerOptions options_;
    Factory factory_;
    std::string type_;
    Teuchos::RCP<matrix_type> preconditioner_matrix_;
    Teuchos::RCP<IfpackPrec> ifpack_;
    Teuchos::RCP<AmgPrec> amg_;
    Teuchos::RCP<inner_operator_type> inner_;
    Teuchos::RCP<operator_type> operator_;
    std::uint64_t prepared_revision_ = 0, generation_ = 0;
    bool prepared_ = false;

    void update_matrix()
    {
        if constexpr (std::is_same_v<Scalar, PrecScalar>) {
            preconditioner_matrix_ = matrix_.native();
        } else {
            if (preconditioner_matrix_.is_null()) {
                preconditioner_matrix_ = matrix_.template create_native_structure<PrecScalar>();
            }
            const auto source = detail::read_coefficients(std::as_const(*matrix_.native()));
            const auto target = detail::overwrite_coefficients(*preconditioner_matrix_);
            using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
            std::size_t bad = 0;
            Kokkos::parallel_reduce("linalg/preconditioner_coefficients", Policy(Exec{}, 0, target.extent(0)),
                detail::ConvertMatrixValues<decltype(source), decltype(target)>{source, target}, bad);
            mpi::require(bad == 0, "Linalg: matrix coefficients are nonfinite in the preconditioner precision");
        }
    }

    Teuchos::ParameterList amg_parameters() const
    {
        auto parameters = options_.parameters;
        const auto set_default = [&](const char* name, const auto& value) {
            if (!parameters.isParameter(name)) parameters.set(name, value);
        };
        // Block storage uses an unsmoothed, block-preserving hierarchy by default.
        set_default("multigrid algorithm", Matrix::block_storage ? "unsmoothed" : "sa");
        set_default("cycle type", "V");
        set_default("smoother: type", Matrix::block_storage ? "RELAXATION" : "CHEBYSHEV");
        if constexpr (Matrix::block_storage) {
            set_default("number of equations", matrix_.block_size());
            mpi::require(parameters.get<int>("number of equations") == matrix_.block_size(),
                         "Linalg: AMG equation count differs from the BSR block size");
            set_default("coarse: type", "RELAXATION");
            if (detail::upper_name(parameters.get<std::string>("smoother: type")) == "RELAXATION") {
                auto& smoother = parameters.sublist("smoother: params");
                if (!smoother.isParameter("relaxation: type")) smoother.set("relaxation: type", "Jacobi");
            }
            if (detail::upper_name(parameters.get<std::string>("coarse: type")) == "RELAXATION") {
                auto& coarse = parameters.sublist("coarse: params");
                if (!coarse.isParameter("relaxation: type")) coarse.set("relaxation: type", "Jacobi");
                if (!coarse.isParameter("relaxation: sweeps")) coarse.set("relaxation: sweeps", 8);
            }
        }
        set_default("use kokkos refactor", !Matrix::block_storage);
        if constexpr (Matrix::block_storage)
            mpi::require(!parameters.get<bool>("use kokkos refactor"),
                         "Linalg: Trilinos 17.2.1 block AMG requires use kokkos refactor=false");
        set_default("coarse: max size", 200);
        set_default("verbosity", "none");
        set_default("reuse: type", "none");
        return parameters;
    }
};

} // namespace krona::linalg