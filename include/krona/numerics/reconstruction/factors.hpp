#pragma once

#include <krona/numerics/reconstruction/detail/contracts.hpp>
#include <krona/numerics/reconstruction/detail/factors_kernel.hpp>
#include <krona/mpi/error.hpp>

namespace krona::numerics::reconstruction {
namespace detail {
template<class Exec, class P>
void accessible() {
    if constexpr (requires { typename P::memory_space; })
        static_assert(Kokkos::SpaceAccessibility<Exec, typename P::memory_space>::accessible);
}
} // namespace detail

/// @brief Computes owned-cell factors asynchronously into caller-provided storage; no temporary arrays or halo exchange.
/// Stencil: size(), required_input_size(), size(cell), and (cell,k)->StencilEntry.
/// Use the same unlimited scheme and evaluation points when preparing and consuming the factors.
/// Values (including ghost/boundary samples) and owned gradients must be current. Exchange factors/gradients before
/// reconstructing a remote side; keep all inputs/outputs alive until work and dependent exchanges complete.
template<class Real = ComputeReal, class Exec, class Stencil, class D, class Output,
         class Limiter = BarthJespersen, class Scheme = SecondOrderUpwind, class Coupling = Componentwise>
void compute_limiters_async(const Exec& exec, Stencil stencil, D input, Output output,
    Limiter limiter = {}, Scheme scheme = {}, Coupling = {}, CellRange range = {}) {
    static_assert(std::is_floating_point_v<Real>);
    static_assert((Scheme::requirements & Factors) == 0, "Prepare factors with an unlimited reconstruction");
    detail::check_input_components<Scheme, D>();
    static_assert(std::is_same_v<Coupling, Componentwise> || std::is_same_v<Coupling, SharedMinimum>);
    static_assert(Output::components == D::components ||
        (std::is_same_v<Coupling, SharedMinimum> && Output::components == 1));
    detail::accessible<Exec, Stencil>();
    detail::accessible<Exec, decltype(input.values)>(); detail::accessible<Exec, Output>();
    mpi::require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
                 "Reconstruction: inactive Kokkos");
    const auto count = stencil.size();
    if (range.end == std::numeric_limits<std::size_t>::max()) range.end = count;
    mpi::require(range.begin <= range.end && range.end <= count &&
                 output.size() >= count && input.values.size() >= stencil.required_input_size(),
        "Reconstruction: invalid range or input/output extents");

    if constexpr ((Scheme::requirements & Gradients) != 0) {
        detail::accessible<Exec, decltype(input.gradients)>();
        if constexpr (requires { input.gradients.size(); }) {
            mpi::require(input.gradients.size() >= count,
                         "Reconstruction: gradient extent is too short");
        }
    }
    if constexpr ((Scheme::requirements & Hessians) != 0) {
        detail::accessible<Exec, decltype(input.hessians)>();

        if constexpr (requires { input.hessians.size(); }) {
            mpi::require(
                input.hessians.size() >= count,
                "Reconstruction: Hessian extent is too short");
        }

    }
    if constexpr ((Scheme::requirements & Moments) != 0) {
        detail::accessible<Exec, decltype(input.moments)>();
        if constexpr (requires { input.moments.size(); }) {
            mpi::require(
                input.moments.size() >= count,
                "Reconstruction: moment extent is too short");
        }
    }
    if (range.begin == range.end) return;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    mpi::guarded([&] {
        Kokkos::parallel_for("reconstruction/limiters", Policy(exec, range.begin, range.end),
            detail::PrepareFactors<Real, Stencil, D, Output, Limiter, Scheme, Coupling>{stencil, input, output, limiter, scheme});
    });
}

} // namespace krona::numerics::reconstruction
