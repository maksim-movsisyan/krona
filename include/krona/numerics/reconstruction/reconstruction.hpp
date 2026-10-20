#pragma once

#include <krona/numerics/reconstruction/pack.hpp>
#include <krona/numerics/reconstruction/schemes.hpp>

namespace krona::numerics::reconstruction {
namespace detail {
template<class R, class Scheme, class Side, class D, std::size_t... C>
KOKKOS_INLINE_FUNCTION auto side_values(Scheme scheme, const Side& side, const D& d, std::index_sequence<C...>)
{ return Kokkos::Array<R, D::components>{(static_cast<R>(d.values(side.cell, C))+scheme.template increment<R>(side, d, C))...}; }

template<class R, class Scheme, class Face, class D, std::size_t... C>
KOKKOS_INLINE_FUNCTION auto face_values(Scheme scheme, const Face& face, const D& d, std::index_sequence<C...>)
{
    return FaceValues<R, D::components>{{(static_cast<R>(d.values(face.left.cell, C))+scheme.template increment<R>(face.left, d, C))...},
                                       {(static_cast<R>(d.values(face.right.cell, C))+scheme.template increment<R>(face.right, d, C))...}};
}
template<class Exec, class P>
void accessible()
{
    if constexpr (requires { typename P::memory_space; })
        static_assert(Kokkos::SpaceAccessibility<Exec, typename P::memory_space>::accessible);
}

template<class R, class Stencil, class D, class Output, class Limiter, class Scheme, class Coupling>
struct PrepareFactors {
    Stencil stencil; D data; Output output; Limiter limiter; Scheme scheme;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const
    {
        constexpr auto N = D::components;
        Kokkos::Array<R, N> base{}, minimum{}, maximum{}, theta{};
        for (std::size_t c = 0; c < N; ++c) {
            base[c] = minimum[c] = maximum[c] = static_cast<R>(data.values(cell, c));
            theta[c] = R{1};
        }
        const auto count = stencil.size(cell);
        // Read every sample once for the entire component batch.
        for (std::size_t k = 0; k < count; ++k) {
            const auto entry = stencil(cell, k);
            for (std::size_t c = 0; c < N; ++c) {
                const R value = static_cast<R>(data.values(entry.sample, c));
                minimum[c] = Kokkos::min(minimum[c], value);
                maximum[c] = Kokkos::max(maximum[c], value);
            }
        }
        for (std::size_t k = 0; k < count; ++k) {
            const auto entry = stencil(cell, k);
            for (std::size_t c = 0; c < N; ++c) {
                const R increment = scheme.template increment<R>(entry.side, data, c);
                const R factor = static_cast<R>(limiter(LimiterContext<R>{cell, c, base[c], minimum[c], maximum[c], increment}));
                KOKKOS_ASSERT(Kokkos::isfinite(factor) && factor >= R{0} && factor <= R{1});
                theta[c] = Kokkos::min(theta[c], factor);
            }
        }
        if constexpr (std::is_same_v<Coupling, SharedMinimum>) {
            R shared = R{1};
            for (std::size_t c = 0; c < N; ++c) shared = Kokkos::min(shared, theta[c]);
            for (std::size_t c = 0; c < Output::components; ++c) output(cell, c) = static_cast<typename Output::value_type>(shared);
        } else {
            for (std::size_t c = 0; c < N; ++c) output(cell, c) = static_cast<typename Output::value_type>(theta[c]);
        }
    }
};
} // namespace detail

/// @brief A custom policy supplies requirements and increment<Real>(side, data, component).
/// Evaluate inside the caller's kernel: no allocation, submission, synchronization, or MPI.
/// Finite valid inputs and live backing storage are caller preconditions. States are in component-pack order.
template<class Real = ComputeReal, class Scheme, class Side, class D>
KOKKOS_INLINE_FUNCTION auto evaluate_side(Scheme scheme, const Side& side, const D& d)
{ return detail::side_values<Real>(scheme, side, d, std::make_index_sequence<D::components>{}); }

template<class Real = ComputeReal, class Scheme, class Face, class D>
KOKKOS_INLINE_FUNCTION auto evaluate(Scheme scheme, const Face& face, const D& d)
{ return detail::face_values<Real>(scheme, face, d, std::make_index_sequence<D::components>{}); }

/// @brief Computes owned-cell factors asynchronously into caller-provided storage; no temporary arrays or halo exchange.
/// Stencil: size(), required_input_size(), size(cell), and (cell,k)->StencilEntry.
/// Use the same unlimited scheme and evaluation points when preparing and consuming the factors.
/// Values (including ghost/boundary samples) and owned gradients must be current. Exchange factors/gradients before
/// reconstructing a remote side; keep all inputs/outputs alive until work and dependent exchanges complete.
template<class Real = ComputeReal, class Exec, class Stencil, class D, class Output,
         class Limiter = BarthJespersen, class Scheme = SecondOrderUpwind, class Coupling = Componentwise>
void compute_limiters_async(const Exec& exec, Stencil stencil, D input, Output output,
    Limiter limiter = {}, Scheme scheme = {}, Coupling = {}, CellRange range = {})
{
    static_assert(std::is_floating_point_v<Real>);
    static_assert((Scheme::requirements & Factors) == 0, "Prepare factors with an unlimited reconstruction");
    static_assert(!std::is_const_v<typename Output::value_type>);
    static_assert(std::is_same_v<Coupling, Componentwise> || std::is_same_v<Coupling, SharedMinimum>);
    static_assert(Output::components == D::components ||
        (std::is_same_v<Coupling, SharedMinimum> && Output::components == 1));
    detail::accessible<Exec, Stencil>();
    detail::accessible<Exec, decltype(input.values)>(); detail::accessible<Exec, Output>();
    if (!Kokkos::is_initialized() || Kokkos::is_finalized()) throw std::logic_error("Reconstruction: inactive Kokkos");
    const auto count = stencil.size();
    if (range.end == std::numeric_limits<std::size_t>::max()) range.end = count;
    if (range.begin > range.end || range.end > count || output.size() < count || input.values.size() < stencil.required_input_size())
        throw std::invalid_argument("Reconstruction: invalid range or input/output extents");
    if constexpr ((Scheme::requirements & Gradients) != 0) {
        detail::accessible<Exec, decltype(input.gradients)>();
        if constexpr (requires { input.gradients.size(); })
            if (input.gradients.size() < count) throw std::invalid_argument("Reconstruction: gradient extent is too short");
    }
    if constexpr ((Scheme::requirements & Hessians) != 0) {
        detail::accessible<Exec, decltype(input.hessians)>(); detail::accessible<Exec, decltype(input.moments)>();
        if constexpr (requires { input.hessians.size(); })
            if (input.hessians.size() < count) throw std::invalid_argument("Reconstruction: Hessian extent is too short");
        if constexpr (requires { input.moments.size(); })
            if (input.moments.size() < count) throw std::invalid_argument("Reconstruction: moment extent is too short");
    }
    if (range.begin == range.end) return;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("reconstruction/limiters", Policy(exec, range.begin, range.end),
        detail::PrepareFactors<Real, Stencil, D, Output, Limiter, Scheme, Coupling>{stencil, input, output, limiter, scheme});
}

} // namespace krona::numerics::reconstruction
