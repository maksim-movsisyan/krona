#pragma once

#include <krona/linalg/csr_pattern.hpp>
#include <krona/linalg/vector.hpp>

#include <cstdint>
#include <span>
#include <utility>

namespace krona::linalg {
namespace detail {
// Each original input position maps to a distinct native CSR coefficient.
template<class Input, class Values, class Permutation>
struct SetCsrValues {
    Input input;
    Values values;
    Permutation permutation;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    { values(permutation(i)) = static_cast<typename Values::non_const_value_type>(input(i)); }
};
} // namespace detail

/// @brief Owns the coefficients of a Tpetra::CrsMatrix on a shared CSR graph.
/// @tparam Scalar Precision of each scalar coefficient.
/// @tparam Exec Kokkos execution space used by Tpetra and coefficient assembly.
/// @details Each finalized graph entry has one coefficient. Reassembling values leaves the graph in place.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
class CsrMatrix {
public:
    using scalar_type = Scalar;
    using execution_space = Exec;
    using memory_space = typename Types<Scalar, Exec>::memory_space;
    using native_type = typename Types<Scalar, Exec>::matrix_type;
    template<class OtherScalar>
    using native_type_for = typename Types<OtherScalar, Exec>::matrix_type;
    static constexpr bool block_storage = false;

    /// @brief Collectively allocates coefficients and shares the pattern's graph and maps.
    explicit CsrMatrix(const CsrPattern<Exec>& pattern) : pattern_(pattern) {
        mpi::guarded([&] {
            matrix_ = create_native_structure<Scalar>();
            Kokkos::fence("Linalg: matrix initialization complete");
        });
    }
    CsrMatrix(const CsrMatrix&) = delete;
    CsrMatrix& operator=(const CsrMatrix&) = delete;
    ~CsrMatrix() { commit(); }

    [[nodiscard]] const CsrPattern<Exec>& pattern() const noexcept { return pattern_; }
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return pattern_.distribution(); }
    [[nodiscard]] static constexpr LocalOrdinal block_size() noexcept { return 1; }
    /// @brief Change counter used to decide whether the preconditioner needs refreshing.
    /// Requesting a write view advances it immediately. The associated kernels may still be pending.
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }

    /// @brief Allocates a native matrix of another precision on the same graph and maps.
    /// @tparam OtherScalar Precision of the new coefficients, used for mixed-precision preconditioning.
    template<class OtherScalar>
    [[nodiscard]] Teuchos::RCP<native_type_for<OtherScalar>> create_native_structure() const {
        auto matrix = Teuchos::rcp(new native_type_for<OtherScalar>(pattern_.native()));
        // Finalize the native operator on the existing graph. Row and vector maps are the same.
        matrix->fillComplete(distribution().native(), distribution().native());
        return matrix;
    }

    /// @brief Gets a write view indexed by native CSR entry, preserving current values.
    /// Use pattern().input_to_entry() to translate original CSR slots to these positions.
    [[nodiscard]] auto edit_values(const Exec& exec = {}) {
        begin_access(exec);
        ++revision_;
        return mpi::guarded([&] { return matrix_->getLocalValuesDevice(Tpetra::Access::ReadWrite); });
    }
    /// @brief Full coefficient overwrite; skips synchronization of old host values.
    [[nodiscard]] auto overwrite_values(const Exec& exec = {}) {
        begin_access(exec);
        ++revision_;
        return mpi::guarded([&] { return matrix_->getLocalValuesDevice(Tpetra::Access::OverwriteAll); });
    }
    /// @brief Gets a read-only view indexed by native CSR entry on the batch's execution instance.
    [[nodiscard]] auto read_values(const Exec& exec = {}) const {
        begin_access(exec);
        return mpi::guarded([&] { return std::as_const(*matrix_).getLocalValuesDevice(Tpetra::Access::ReadOnly); });
    }
    /// @brief Waits for user kernels submitted on the batch's execution instance.
    /// Release access views before switching to host access or a native Tpetra operation.
    void commit() const {
        if (pending_) { pending_exec_.fence("Linalg: matrix accesses committed"); pending_ = false; }
    }
    void require_committed() const
    { mpi::require(!pending_, "Linalg: matrix has pending reads or writes; call commit()"); }

    /// @brief Blocking host assembly in the original CSR input order.
    void set_values(std::span<const Scalar> input) {
        require_committed();
        mpi::require(input.size() == pattern_.local_nonzeros(), "Linalg: wrong coefficient count");
        mpi::guarded([&] {
            auto values = matrix_->getLocalValuesHost(Tpetra::Access::OverwriteAll);
            auto permutation = pattern_.host_input_to_entry();
            for (std::size_t i = 0; i < input.size(); ++i) values(permutation[i]) = input[i];
            ++revision_;
        });
    }

    /// @brief Submits a device copy from the original CSR order into native coefficient slots.
    /// @tparam Input One-dimensional Kokkos view with one value per original input entry.
    /// @details Keep the input valid until commit() completes the copy.
    template<class Input>
    void set_values_async(Input input, const Exec& exec = {}) {
        static_assert(Input::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Input::memory_space>::accessible);
        mpi::require(input.extent(0) == pattern_.local_nonzeros(), "Linalg: wrong coefficient count");
        auto values = overwrite_values(exec);
        auto permutation = pattern_.input_to_entry();
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("linalg/set_csr_values", Policy(exec, 0, input.extent(0)),
            detail::SetCsrValues<Input, decltype(values), decltype(permutation)>{input, values, permutation});
    }

    /// @brief Collective, blocking y = alpha*A*x + beta*y, including native remote imports.
    void apply(const Vector<Scalar, Exec>& x, Vector<Scalar, Exec>& y,
               Scalar alpha = Scalar{1}, Scalar beta = Scalar{0}) const {
        require_committed();
        x.require_committed();
        y.require_committed();
        // Use the exact map shared by the matrix, so local vector positions match its rows.
        const auto* map = distribution().native().getRawPtr();
        mpi::require(x.distribution().native().getRawPtr() == map && y.distribution().native().getRawPtr() == map,
                     "Linalg: vectors must share the matrix distribution");
        mpi::require(x.native().getRawPtr() != y.native().getRawPtr(), "Linalg: in-place matrix apply is unsupported");
        mpi::guarded([&] {
            matrix_->apply(*x.native(), *y.native(), Teuchos::NO_TRANS, alpha, beta);
            Kokkos::fence("Linalg: matrix apply complete");
        });
    }

    /// @brief Returns the underlying Tpetra matrix for interoperability.
    /// Finish native coefficient edits before calling mark_modified() to invalidate cached preconditioning.
    [[nodiscard]] const Teuchos::RCP<native_type>& native() const noexcept { return matrix_; }
    void mark_modified() { require_committed(); ++revision_; }

private:
    CsrPattern<Exec> pattern_;
    Teuchos::RCP<native_type> matrix_;
    mutable Exec pending_exec_{};
    std::uint64_t revision_ = 0;
    mutable bool pending_ = false;

    void begin_access(const Exec& exec) const {
        mpi::require(!pending_ || pending_exec_ == exec,
                     "Linalg: commit before switching execution instances");
        pending_exec_ = exec;
        pending_ = true;
    }
};

} // namespace krona::linalg
