#pragma once

#include <krona/linalg/csr_pattern.hpp>
#include <krona/linalg/vector.hpp>

#include <cstdint>
#include <span>

namespace krona::linalg {
namespace detail {
template<class Input, class Values, class Permutation>
struct SetCsrValues {
    Input input;
    Values values;
    Permutation permutation;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    { values(permutation(i)) = static_cast<typename Values::non_const_value_type>(input(i)); }
};
} // namespace detail

/// @brief Native Tpetra coefficients with a shared, immutable CSR graph.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
class CsrMatrix {
public:
    using native_type = typename Types<Scalar, Exec>::matrix_type;

    explicit CsrMatrix(const CsrPattern<Exec>& pattern) : pattern_(pattern)
    {
        mpi::guarded([&] {
            matrix_ = Teuchos::rcp(new native_type(pattern_.native()));
            matrix_->setAllToScalar(Scalar{0});
            matrix_->fillComplete(pattern_.distribution().native(), pattern_.distribution().native());
            Kokkos::fence("Linalg: matrix initialization complete");
        });
    }
    CsrMatrix(const CsrMatrix&) = delete;
    CsrMatrix& operator=(const CsrMatrix&) = delete;

    [[nodiscard]] const CsrPattern<Exec>& pattern() const noexcept { return pattern_; }
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }

    /// @brief Native CSR order; use pattern.input_to_entry() for original input slots.
    [[nodiscard]] auto edit_values(const Exec& exec = {})
    {
        require_committed();
        pending_exec_ = exec; pending_ = true; ++revision_;
        return mpi::guarded([&] { return matrix_->getLocalMatrixDevice().values; });
    }
    void commit()
    {
        if (pending_) { pending_exec_.fence("Linalg: matrix writes committed"); pending_ = false; }
    }
    void require_committed() const
    { mpi::require(!pending_, "Linalg: matrix has pending writes; call commit()"); }

    /// @brief Blocking host assembly in the original CSR input order.
    void set_values(std::span<const Scalar> input)
    {
        require_committed();
        mpi::require(input.size() == pattern_.local_nonzeros(), "Linalg: wrong coefficient count");
        mpi::guarded([&] {
            auto values = matrix_->getLocalMatrixHost().values;
            auto permutation = pattern_.host_input_to_entry();
            for (std::size_t i = 0; i < input.size(); ++i) values(permutation[i]) = input[i];
            ++revision_;
        });
    }

    /// @brief Asynchronous device assembly in the original CSR input order.
    template<class Input>
    void set_values_async(Input input, const Exec& exec = {})
    {
        static_assert(Input::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Input::memory_space>::accessible);
        mpi::require(input.extent(0) == pattern_.local_nonzeros(), "Linalg: wrong coefficient count");
        auto values = edit_values(exec);
        auto permutation = pattern_.input_to_entry();
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("linalg/set_csr_values", Policy(exec, 0, input.extent(0)),
            detail::SetCsrValues<Input, decltype(values), decltype(permutation)>{input, values, permutation});
    }

    /// @brief Collective, blocking y = alpha*A*x + beta*y, including native remote imports.
    void apply(const Vector<Scalar, Exec>& x, Vector<Scalar, Exec>& y,
               Scalar alpha = Scalar{1}, Scalar beta = Scalar{0}) const
    {
        require_committed(); x.require_committed(); y.require_committed();
        const auto* map = pattern_.distribution().native().getRawPtr();
        mpi::require(x.distribution().native().getRawPtr() == map && y.distribution().native().getRawPtr() == map,
                        "Linalg: vectors must share the matrix distribution");
        mpi::require(x.native().getRawPtr() != y.native().getRawPtr(), "Linalg: in-place matrix apply is unsupported");
        mpi::guarded([&] {
            matrix_->apply(*x.native(), *y.native(), Teuchos::NO_TRANS, alpha, beta);
            Kokkos::fence("Linalg: matrix apply complete");
        });
    }

    /// @brief Advanced interoperability; call mark_modified() after native coefficient edits.
    [[nodiscard]] const Teuchos::RCP<native_type>& native() const noexcept { return matrix_; }
    void mark_modified() { require_committed(); ++revision_; }

private:
    CsrPattern<Exec> pattern_;
    Teuchos::RCP<native_type> matrix_;
    Exec pending_exec_{};
    std::uint64_t revision_ = 0;
    bool pending_ = false;
};

} // namespace krona::linalg
