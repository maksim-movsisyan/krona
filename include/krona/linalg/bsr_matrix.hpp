#pragma once

#include <krona/linalg/bsr_pattern.hpp>
#include <krona/linalg/vector.hpp>

#include <cstdint>
#include <span>
#include <utility>

namespace krona::linalg {
namespace detail {

template<class Native, class Values>
auto block_values(Values values, std::size_t blocks, LocalOrdinal size)
{
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

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
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

/// @brief Fixed-size dense blocks on an immutable graph of block rows.
/// Graph distribution indexes blocks; distribution() indexes cell-major scalar DOFs.
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

    /// @brief Collective; copies the pattern descriptors and allocates coefficients.
    explicit BsrMatrix(const BsrPattern<Exec>& pattern) : pattern_(pattern)
    {
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
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }

    /// @brief Native interoperability: independent coefficients sharing graph and point maps.
    template<class OtherScalar>
    [[nodiscard]] Teuchos::RCP<native_type_for<OtherScalar>> create_native_structure() const
    {
        return Teuchos::rcp(new native_type_for<OtherScalar>(*pattern_.native(),
            *distribution().native(), *distribution().native(), block_size()));
    }

    /// @brief Flat native coefficient order; prefer edit_blocks() for layout-independent assembly.
    [[nodiscard]] auto edit_values(const Exec& exec = {})
    {
        begin_access(exec); ++revision_;
        return mpi::guarded([&] { return matrix_->getValuesDeviceNonConst(); });
    }
    /// @brief Declares a full overwrite. Native BlockCrsMatrix still uses ReadWrite access.
    [[nodiscard]] auto overwrite_values(const Exec& exec = {}) { return edit_values(exec); }
    [[nodiscard]] auto read_values(const Exec& exec = {}) const
    {
        begin_access(exec);
        return mpi::guarded([&] { return std::as_const(*matrix_).getValuesDevice(); });
    }

    /// @brief Unmanaged (native graph entry, equation, variable) coefficient view.
    [[nodiscard]] auto edit_blocks(const Exec& exec = {})
    { return detail::block_values<native_type>(edit_values(exec), pattern_.local_nonzeros(), block_size()); }
    [[nodiscard]] auto overwrite_blocks(const Exec& exec = {})
    { return detail::block_values<native_type>(overwrite_values(exec), pattern_.local_nonzeros(), block_size()); }
    [[nodiscard]] auto read_blocks(const Exec& exec = {}) const
    { return detail::block_values<native_type>(read_values(exec), pattern_.local_nonzeros(), block_size()); }

    void commit() const
    {
        if (pending_) { pending_exec_.fence("Linalg: block matrix accesses committed"); pending_ = false; }
    }
    void require_committed() const
    { mpi::require(!pending_, "Linalg: matrix has pending reads or writes; call commit()"); }

    /// @brief Blocking host input: original graph order, then row-major dense blocks.
    void set_values(std::span<const Scalar> input)
    {
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

    /// @brief Async original-order input: flat row-major blocks or a rank-three View.
    template<class Input>
    void set_values_async(Input input, const Exec& exec = {})
    {
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
               Scalar alpha = Scalar{1}, Scalar beta = Scalar{0}) const
    {
        require_committed(); x.require_committed(); y.require_committed();
        const auto* map = distribution().native().getRawPtr();
        mpi::require(x.distribution().native().getRawPtr() == map && y.distribution().native().getRawPtr() == map,
                     "Linalg: vectors must share the matrix distribution");
        mpi::require(x.native().getRawPtr() != y.native().getRawPtr(), "Linalg: in-place matrix apply is unsupported");
        mpi::guarded([&] {
            matrix_->apply(*x.native(), *y.native(), Teuchos::NO_TRANS, alpha, beta);
            Kokkos::fence("Linalg: block matrix apply complete");
        });
    }

    /// @brief Advanced interoperability; call mark_modified() after native coefficient edits.
    [[nodiscard]] const Teuchos::RCP<native_type>& native() const noexcept { return matrix_; }
    void mark_modified() { require_committed(); ++revision_; }

private:
    BsrPattern<Exec> pattern_;
    Teuchos::RCP<native_type> matrix_;
    mutable Exec pending_exec_{};
    std::uint64_t revision_ = 0;
    mutable bool pending_ = false;

    void begin_access(const Exec& exec) const
    {
        mpi::require(!pending_ || pending_exec_ == exec, "Linalg: commit before switching execution instances");
        pending_exec_ = exec; pending_ = true;
    }
};

} // namespace krona::linalg
