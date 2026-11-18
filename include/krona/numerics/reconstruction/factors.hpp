#pragma once

#include <krona/numerics/reconstruction/detail/factor_validation.hpp>
#include <krona/numerics/reconstruction/detail/factors_kernel.hpp>
#include <krona/numerics/reconstruction/face_profiles.hpp>
#include <krona/numerics/reconstruction/limiters.hpp>

namespace krona::numerics::reconstruction {
namespace detail {

/// @brief Launches the limiter kernel for a validated range of cells.
/// @tparam Real Floating-point type used to compute bounds, increments, and factors.
/// @tparam Exec Kokkos execution space that can access the input and output arrays.
/// @tparam Profiles Bundle of cell values, sample indices, and formulas for the test points.
/// @tparam Output Writable accessor to the factor field.
/// @tparam Limiter Rule that turns the bounds and the unlimited increment at a point into a factor in [0, 1].
/// @tparam Coupling Tag type selecting a factor per component or a shared minimum across components.
/// @param exec Execution instance on which to enqueue the kernel.
/// @param profiles Inputs for computing the bounds and evaluating the unlimited reconstruction.
/// @param output Destination for the factors of the selected cells.
/// @param limiter Configured limiter, such as BarthJespersen or Venkatakrishnan.
/// @param range Half-open cell range [begin, end) returned by validation.
/// @details The unnamed Coupling argument lets the compiler deduce the tag type from the call.
/// An empty range returns immediately. Keep the backing arrays alive until the kernel finishes.
template <class Real, class Exec, class Profiles, class Output, class Limiter, class Coupling>
void submit_factors(const Exec& exec, Profiles profiles, Output output, Limiter limiter, Coupling,
                    CellRange range) {
    static_assert(std::is_floating_point_v<Real>);
    if (range.begin == range.end)
        return;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    mpi::guarded([&] {
        Kokkos::parallel_for(
            "reconstruction/limiters", Policy(exec, range.begin, range.end),
            PrepareFactors<Real, Profiles, Output, Limiter, Coupling>{profiles, output, limiter});
    });
}

/// @brief Adapts a custom cell stencil to the common limiter kernel.
/// @tparam Stencil Provides sample indices for the bounds and descriptors for the points to test.
/// @tparam Values Readable field pack indexed as values(cell_or_sample, component).
/// @tparam Formula Unlimited reconstruction formula applied at each test point.
/// @details The stencil addresses points by cell and local slot. It can supply these points
/// directly, for example in a test or when the caller has prepared its own geometry.
template <class Stencil, class Values, class Formula>
struct CellProfiles {
    Stencil stencil;
    Values values;
    Formula formula;

    /// @brief Returns the number of samples used alongside the base cell value to find the bounds.
    /// @param cell Index of the cell whose factor is being computed.
    KOKKOS_INLINE_FUNCTION std::size_t sample_count(std::size_t cell) const {
        return stencil.size(cell);
    }

    /// @brief Returns the number of points at which to test the reconstruction.
    /// @param cell Index of the cell whose factor is being computed.
    /// @details Uses stencil.point_count(cell) when supplied. Otherwise uses the sample count.
    KOKKOS_INLINE_FUNCTION std::size_t point_count(std::size_t cell) const {
        if constexpr (requires { stencil.point_count(cell); })
            return stencil.point_count(cell);
        else
            return stencil.size(cell);
    }

    /// @brief Passes the formula and one point descriptor to a callback.
    /// @tparam Visitor Callable that accepts (formula, point).
    /// @param cell Index of the cell whose profile is evaluated.
    /// @param slot Local point index in [0, point_count(cell)).
    /// @param visit Callback that evaluates the increment and updates the running limiter factor.
    template <class Visitor>
    KOKKOS_INLINE_FUNCTION void visit_point(std::size_t cell, std::size_t slot, const Visitor& visit) const {
        visit(formula, stencil.point(cell, slot));
    }
};

} // namespace detail

/// @brief Writes limiter factors for the given profiles on this execution instance.
/// @tparam Real Arithmetic type used to compute the bounds and factors.
/// @tparam Exec Execution space used to read the inputs and write the factors.
/// @tparam Stencil Face stencil with incident faces and bound samples for owned cells.
/// @tparam Values Provider of cell and physical-face values used in the bounds.
/// @tparam Inner Unlimited formula for inner faces.
/// @tparam Boundary Unlimited formula for physical faces.
/// @tparam Output Writable accessor for shared or per-component factors.
/// @tparam Limiter Callable reducing one point increment to a factor in [0,1].
/// @tparam Coupling Selects separate component factors or a shared minimum.
/// @details Queue after updating sample values and derivative fields. Keep the arrays
/// alive until the work completes. Exchange owned factors before using MPI ghost traces.
template <class Real = ComputeReal, class Exec, class Stencil, class Values, class Inner, class Boundary,
          class Output, class Limiter = BarthJespersen, class Coupling = Componentwise>
void compute_limiters_async(const Exec& exec, FaceProfiles<Stencil, Values, Inner, Boundary> profiles,
                            Output output, Limiter limiter = {}, Coupling coupling = {}, CellRange range = {}) {
    range = detail::validate_factor_launch<Exec>(profiles, output, range, coupling);
    const auto cells = profiles.stencil.size();
    detail::validate_unlimited_formula<Exec>(profiles.inner, profiles.values, cells);
    detail::validate_unlimited_formula<Exec>(profiles.boundary, profiles.values, cells);
    detail::submit_factors<Real>(exec, profiles, output, limiter, coupling, range);
}

/// @brief Prepares factors on a custom stencil of bound samples and evaluation points.
/// @tparam Real Arithmetic type used to compute the factors.
/// @tparam Exec Execution space used to submit the kernel.
/// @tparam Stencil Supplies size(), size(cell), required_input_size(), sample(cell,slot), and point(cell,slot).
/// It may supply point_count(cell) when the number of evaluation points differs from size(cell).
/// @tparam Values Provider of the base and sample values.
/// @tparam Formula Unlimited formula used at the supplied points.
/// @tparam Output Writable accessor for factors.
/// @tparam Limiter Callable mapping a LimiterContext to a factor.
/// @tparam Coupling Selects separate component factors or a shared minimum.
template <class Real = ComputeReal, class Exec, class Stencil, class Values, class Formula, class Output,
          class Limiter = BarthJespersen, class Coupling = Componentwise>
void compute_limiters_async(const Exec& exec, Stencil stencil, Values values, Formula formula, Output output,
                            Limiter limiter = {}, Coupling coupling = {}, CellRange range = {}) {
    const auto profiles = detail::CellProfiles{stencil, values, formula};
    range = detail::validate_factor_launch<Exec>(profiles, output, range, coupling);
    detail::validate_unlimited_formula<Exec>(formula, values, stencil.size());
    detail::submit_factors<Real>(exec, profiles, output, limiter, coupling, range);
}

} // namespace krona::numerics::reconstruction