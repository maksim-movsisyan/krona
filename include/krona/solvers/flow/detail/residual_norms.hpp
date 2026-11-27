#pragma once

#include <krona/linalg/vector.hpp>
#include <krona/mpi/collectives.hpp>

#include <array>
#include <span>
#include <utility>

namespace krona::solvers::flow::detail {

/// @brief Packs one cell's residual into the linear RHS and sums the stored values squared.
template<class Pack, class Rhs>
struct PackResidual {
    using value_type = Kokkos::Array<double, 5>;
    Pack residual;
    Rhs rhs;

    template<std::size_t C>
    KOKKOS_INLINE_FUNCTION void add(std::size_t cell, value_type& squares) const {
        const auto value = static_cast<typename Rhs::non_const_value_type>(residual.template get<C>(cell));
        rhs(5 * cell + C, 0) = value;
        const auto wide = static_cast<double>(value);
        squares[C] += wide * wide;
    }

    template<std::size_t... C>
    KOKKOS_INLINE_FUNCTION void add_cell(std::size_t cell, value_type& squares, std::index_sequence<C...>) const {
        (add<C>(cell, squares), ...);
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell, value_type& squares) const {
        add_cell(cell, squares, std::make_index_sequence<5>{});
    }

    KOKKOS_INLINE_FUNCTION void init(value_type& squares) const { squares = {}; }

    KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
        for (std::size_t c = 0; c < dst.size(); ++c) dst[c] += src[c];
    }
};

/// @brief Commits the RHS and returns global sums of squares in conservative component order.
template<class Exec, class Scalar, class Pack>
[[nodiscard]] std::array<double, 5> pack_residual(const Exec& exec, mpi::CommView comm, Pack residual,
                                                  linalg::Vector<Scalar, Exec>& rhs) {
    static_assert(Pack::components == 5);
    mpi::require(rhs.local_size() == 5 * residual.size(), "Steady flow: RHS and residual sizes disagree");
    Kokkos::Array<double, 5> local{};
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    mpi::guarded([&] {
        const auto target = rhs.overwrite(exec);
        Kokkos::parallel_reduce("flow_solver/pack_residual", Policy(exec, 0, residual.size()),
                                PackResidual{residual, target}, local);
    });
    rhs.commit();
    std::array<double, 5> squares;
    mpi::allreduce<mpi::Reduction::sum>(comm, std::span{local.data(), local.size()}, std::span{squares});
    return squares;
}

} // namespace krona::solvers::flow::detail
