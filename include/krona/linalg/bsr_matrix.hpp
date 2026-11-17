#pragma once

#include <krona/linalg/bsr_pattern.hpp>
#include <krona/linalg/vector.hpp>

#include <cstdint>
#include <span>
#include <utility>

namespace krona::linalg {
namespace detail {

template<class Native, class Values>
auto block_values(Values values, std::size_t blocks, LocalOrdinal size) {
    // Adjacent graph entries hold adjacent dense blocks. Inside each block use Tpetra's
    // row and column strides so the view works with either native layout.
    using Layout = typename Native::little_block_type::array_layout;
    const std::size_t width = static_cast<std::size_t>(size);
    const std::size_t row_stride = std::is_same_v<Layout, Kokkos::LayoutLeft> ? 1 : width;
    const std::size_t col_stride = std::is_same_v<Layout, Kokkos::LayoutLeft> ? width : 1;
    using View = Kokkos::View<typename Values::value_type***, Kokkos::LayoutStride,
                              typename Values::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    return View(values.data(), Kokkos::LayoutStride(blocks, width*width,
                                                   width, row_stride, width, col_stride));
}

template<class Input, class Blocks, class Permutation>
struct SetBsrValues {
    Input input;
    Blocks blocks;
    Permutation permutation;
    std::size_t width;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        // Decode a row-major input position, then let the destination view handle native strides.
        const auto entry = i/(width*width);
        const auto row = (i/width)%width;
        const auto column = i%width;
        using Value = typename Blocks::non_const_value_type;
        if constexpr (Input::rank == 1)
            blocks(permutation(entry), row, column) = static_cast<Value>(input(i));
        else blocks(permutation(entry), row, column) = static_cast<Value>(input(entry, row, column));
    }
};

} // namespace detail

/// @brief Owns the coefficients of a Tpetra::BlockCrsMatrix on a graph of block rows.
/// @tparam Scalar Precision of the values stored inside each dense block.
/// @tparam Exec Kokkos execution space used by Tpetra and coefficient assembly.
/// @details The graph indexes entire blocks. distribution() gives the scalar map used by vectors.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
class BsrMatrix {
public:
    using scalar_type = Scalar;
    using execution_space = Exec;
    using memory_space = typename Types<Scalar, Exec>::memory_space;
    using native_type = typename Types<Scalar, Exec>::block_matrix_type;
    template<class OtherScalar>
    using native_type_for = typename Types<OtherScalar, Exec>::block_matrix_type;
    static constexpr bool block_storage = true;

    /// @brief Collectively allocates block coefficients using the pattern's graph and maps.
    explicit BsrMatrix(const BsrPattern<Exec>& pattern) : pattern_(pattern) {
        mpi::guarded([&] {
            matrix_ = create_native_structure<Scalar>();
            Kokkos::fence("Linalg: block matrix initialization complete");
        });
    }
    BsrMatrix(const CsrPattern<Exec>& graph, LocalOrdinal block_size)
        : BsrMatrix(BsrPattern<Exec>(graph, block_size)) {}
    BsrMatrix(const BsrMatrix&) = delete;
    BsrMatrix& operator=(const BsrMatrix&) = delete;
    ~BsrMatrix() { commit(); }

    [[nodiscard]] const BsrPattern<Exec>& pattern() const noexcept { return pattern_; }
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return pattern_.distribution(); }
    [[nodiscard]] LocalOrdinal block_size() const noexcept { return pattern_.block_size(); }
    [[nodiscard]] std::size_t local_values() const noexcept
    { return pattern_.local_values(); }
    /// @brief Change counter used by the preconditioner. A write view advances it when requested.
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }

    /// @brief Allocates a native block matrix on the same logical graph and scalar maps.
    /// @tparam OtherScalar Precision of the new coefficients, used for mixed-precision preconditioning.
    template<class OtherScalar>
    [[nodiscard]] Teuchos::RCP<native_type_for<OtherScalar>> create_native_structure() const {
        return Teuchos::rcp(new native_type_for<OtherScalar>(*pattern_.native(),
            *distribution().native(), *distribution().native(), block_size()));
    }

    /// @brief Flat native coefficient order; prefer edit_blocks() for layout-independent assembly.
    [[nodiscard]] auto edit_values(const Exec& exec = {}) {
        begin_access(exec);
        ++revision_;
        return mpi::guarded([&] { return matrix_->getValuesDeviceNonConst(); });
    }
    /// @brief Gets a view for replacing all coefficients.
    /// The native BlockCrsMatrix accessor uses ReadWrite, so it can still synchronize old values.
    [[nodiscard]] auto overwrite_values(const Exec& exec = {}) { return edit_values(exec); }
    [[nodiscard]] auto read_values(const Exec& exec = {}) const {
        begin_access(exec);
        return mpi::guarded([&] { return std::as_const(*matrix_).getValuesDevice(); });
    }

    /// @brief Gets a writable view indexed by (native block entry, equation row, unknown column).
    /// The view aliases native storage. Use it only while this matrix and the batch are active.
    [[nodiscard]] auto edit_blocks(const Exec& exec = {})
    { return detail::block_values<native_type>(edit_values(exec), pattern_.local_nonzeros(), block_size()); }
    [[nodiscard]] auto overwrite_blocks(const Exec& exec = {})
    { return detail::block_values<native_type>(overwrite_values(exec), pattern_.local_nonzeros(), block_size()); }
    [[nodiscard]] auto read_blocks(const Exec& exec = {}) const
    { return detail::block_values<native_type>(read_values(exec), pattern_.local_nonzeros(), block_size()); }

    /// @brief Waits for user kernels submitted on the batch's execution instance.
    /// Release access views before switching to host access or a native Tpetra operation.
    void commit() const {
        if (pending_) { pending_exec_.fence("Linalg: block matrix accesses committed"); pending_ = false; }
    }
    void require_committed() const
    { mpi::require(!pending_, "Linalg: matrix has pending reads or writes; call commit()"); }

    /// @brief Blocking host input: original graph order, then row-major dense blocks.
    void set_values(std::span<const Scalar> input) {
        require_committed();
        mpi::require(input.size() == local_values(), "Linalg: wrong BSR coefficient count");
        mpi::guarded([&] {
            auto raw = matrix_->getValuesHostNonConst();
            auto blocks = detail::block_values<native_type>(raw, pattern_.local_nonzeros(), block_size());
            const auto permutation = pattern_.host_input_to_entry();
            const auto width = static_cast<std::size_t>(block_size());
            for (std::size_t entry = 0; entry < permutation.size(); ++entry)
                for (std::size_t row = 0; row < width; ++row)
                    for (std::size_t column = 0; column < width; ++column)
                        blocks(permutation[entry], row, column) = input[(entry*width+row)*width+column];
            ++revision_;
        });
    }

    /// @brief Submits a device copy of blocks in the original graph input order.
    /// @tparam Input A flat row-major view or a Kokkos view indexed by (input entry, row, column).
    /// @details Keep the input valid until commit() completes the copy.
    template<class Input>
    void set_values_async(Input input, const Exec& exec = {}) {
        static_assert(Input::rank == 1 || Input::rank == 3);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Input::memory_space>::accessible);
        const auto width = static_cast<std::size_t>(block_size());
        if constexpr (Input::rank == 1)
            mpi::require(input.extent(0) == local_values(), "Linalg: wrong BSR coefficient count");
        else mpi::require(input.extent(0) == pattern_.local_nonzeros() &&
                          input.extent(1) == width && input.extent(2) == width,
                          "Linalg: wrong BSR block extents");
        const auto permutation = pattern_.input_to_entry();
        auto blocks = overwrite_blocks(exec);
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("linalg/set_bsr_values", Policy(exec, 0, local_values()),
            detail::SetBsrValues<Input, decltype(blocks), decltype(permutation)>{input, blocks, permutation, width});
    }

    /// @brief Collective, blocking y = alpha*A*x + beta*y, including native remote imports.
    void apply(const Vector<Scalar, Exec>& x, Vector<Scalar, Exec>& y,
               Scalar alpha = Scalar{1}, Scalar beta = Scalar{0}) const {
        require_committed();
        x.require_committed();
        y.require_committed();
        // Native apply consumes scalar DOFs. The block-row map is used only by the graph.
        const auto* map = distribution().native().getRawPtr();
        mpi::require(x.distribution().native().getRawPtr() == map && y.distribution().native().getRawPtr() == map,
                     "Linalg: vectors must share the matrix distribution");
        mpi::require(x.native().getRawPtr() != y.native().getRawPtr(), "Linalg: in-place matrix apply is unsupported");
        mpi::guarded([&] {
            matrix_->apply(*x.native(), *y.native(), Teuchos::NO_TRANS, alpha, beta);
            Kokkos::fence("Linalg: block matrix apply complete");
        });
    }

    /// @brief Returns the underlying Tpetra block matrix for interoperability.
    /// Finish native coefficient edits before calling mark_modified() to invalidate cached preconditioning.
    [[nodiscard]] const Teuchos::RCP<native_type>& native() const noexcept { return matrix_; }
    void mark_modified() { require_committed(); ++revision_; }

private:
    BsrPattern<Exec> pattern_;
    Teuchos::RCP<native_type> matrix_;
    mutable Exec pending_exec_{};
    std::uint64_t revision_ = 0;
    mutable bool pending_ = false;

    void begin_access(const Exec& exec) const {
        mpi::require(!pending_ || pending_exec_ == exec, "Linalg: commit before switching execution instances");
        pending_exec_ = exec;
        pending_ = true;
    }
};

} // namespace krona::linalg
