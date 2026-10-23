#pragma once

#include <krona/numerics/time/derivative.hpp>
#include <krona/transport/scalar/status.hpp>
#include <krona/transport/scalar/system_view.hpp>
#include <krona/transport/scalar/transient/mass.hpp>

#include <cmath>

namespace krona::transport::scalar::transient {
namespace detail {
using scalar::detail::flag;
using scalar::detail::BadSystem;
using scalar::detail::BadTemporalMass;
using scalar::detail::BadTemporalState;

template<class Real, class Mass, class Flags>
KOKKOS_INLINE_FUNCTION Real read_mass(Mass mass, std::size_t i, Flags flags)
{
    const Real value = static_cast<Real>(mass(i));
    if (!Kokkos::isfinite(value) || !(value > Real{0})) {
        flag(flags, BadTemporalMass); return Real{0};
    }
    return value;
}
 
template<class Level, class Tail>
template<class Real, std::size_t J, class Weights, class Flags>
KOKKOS_INLINE_FUNCTION Real History<Level, Tail>::rhs(std::size_t i, const Weights& weights, Flags flags) const
{
    const Real mass = read_mass<Real>(level.mass, i, flags);
    const Real value = static_cast<Real>(level.field(i));
    if (!Kokkos::isfinite(value)) { flag(flags, BadTemporalState); return Real{0}; }
    const Real result = -weights[J]*mass*value + tail.template rhs<Real, J+1>(i, weights, flags);
    if (!Kokkos::isfinite(result)) flag(flags, BadSystem);
    return result;
}

template<class Real, class Target, class Weights, class Mass, class History, class Flags>
struct AddSystem {
    Target target; Weights weights; Mass mass; History history; Flags flags;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        const std::size_t entry = static_cast<std::size_t>(target.entries.diagonal(i));
        if (entry >= target.values.extent(0)) { flag(flags, BadSystem); return; }
        const Real diagonal = weights[0]*read_mass<Real>(mass, i, flags);
        using Value = typename decltype(target.values)::non_const_value_type;
        if (!(static_cast<Value>(diagonal) > Value{0})) { flag(flags, BadSystem); return; }
        scalar::detail::add_checked(target.values, entry, diagonal, flags);
        scalar::detail::add_checked(target.rhs, i, history.template rhs<Real, 1>(i, weights, flags), flags);
    }
};

template<class Real, class Field, class Output, class Weights, class Mass, class History, class Flags>
struct AddResidual {
    Field field; Output output; Weights weights; Mass mass; History history; Flags flags;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        const Real current_mass = read_mass<Real>(mass, i, flags);
        const Real value = static_cast<Real>(field(i));
        if (!Kokkos::isfinite(value)) { flag(flags, BadTemporalState); return; }
        const Real residual = history.template rhs<Real, 1>(i, weights, flags)-weights[0]*current_mass*value;
        scalar::detail::add_checked(output, i, residual, flags);
    }
};
} // namespace detail

/// @brief Additive implicit d(M*phi)/dt on owned cells, M=rho*V. No mesh/algebra/history ownership.
/// Inputs are frozen during submitted work and have no overlap with written targets.
/// Each history layer retains its own mass, scalar values and owned-cell ordering.
/// Static topology only: changed cell numbering requires external history remapping.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class TransientOperator {
    static_assert(std::is_floating_point_v<Real>);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;

    explicit TransientOperator(std::size_t owned, Exec exec = {}) : owned_(owned), exec_(exec)
    { execution::require_kokkos(); }

    /// @brief Adds w0*M_current to A_ii and -sum_{j>0} wj*M_j*phi_j to b.
    /// Diagonal entries must be distinct, valid native graph entries, one per owned row.
    template<class Target, class C, std::size_t N, class Mass, class History>
    void add_system_async(Target target, AssemblyStatus<Exec>& status,
        numerics::time::Derivative<C, N> derivative, Mass mass, History history) const
    {
        const auto weights = validate(derivative, mass, history);
        scalar::detail::validate_output<Exec>(target.values, target.values.extent(0));
        scalar::detail::validate_output<Exec>(target.rhs, owned_);
        scalar::detail::validate_view<Exec>(target.entries.diagonal, owned_);
        mpi::require(target.rhs.extent(0) == owned_ && target.entries.diagonal.extent(0) == owned_,
                     "Scalar transient: incompatible system slice/entry map");
        const auto flags = status.device_flags(exec_);
        Kokkos::parallel_for("transient/add_system", Policy(exec_, 0, owned_),
            detail::AddSystem<Real, Target, decltype(weights), Mass, History, decltype(flags)>{
                target, weights, mass, history, flags});
    }

    /// @brief Adds the negative integrated time derivative to r=b-A*phi; no volume normalization.
    /// No ghosts are read. No resets, allocation, fence, MPI, commit or history rotation.
    template<class Field, class Output, class C, std::size_t N, class Mass, class History>
    void add_residual_async(Field field, Output output, AssemblyStatus<Exec>& status,
        numerics::time::Derivative<C, N> derivative, Mass mass, History history) const
    {
        const auto weights = validate(derivative, mass, history);
        scalar::detail::validate_view<Exec>(field, owned_);
        scalar::detail::validate_output<Exec>(output, owned_);
        mpi::require(owned_ == 0 || static_cast<const void*>(field.data()) != static_cast<const void*>(output.data()),
                     "Scalar transient: in-place residual is unsupported");
        const auto flags = status.device_flags(exec_);
        Kokkos::parallel_for("transient/add_residual", Policy(exec_, 0, owned_),
            detail::AddResidual<Real, Field, Output, decltype(weights), Mass, History, decltype(flags)>{
                field, output, weights, mass, history, flags});
    }

    [[nodiscard]] std::size_t local_size() const noexcept { return owned_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
private:
    std::size_t owned_;
    Exec exec_;

    template<class C, std::size_t N, class Mass, class History>
    auto validate(numerics::time::Derivative<C, N> derivative, const Mass& mass, const History& history) const
    {
        static_assert(N == History::size, "Time derivative requires exactly one input layer per history coefficient");
        Kokkos::Array<Real, N+1> weights;
        for (std::size_t j = 0; j <= N; ++j) {
            weights[j] = static_cast<Real>(derivative.weights[j]);
            mpi::require(std::isfinite(weights[j]) && (j != 0 || weights[j] > Real{0}),
                         "Scalar transient: invalid derivative coefficients in computation precision");
        }
        detail::validate_mass<Exec>(mass, owned_);
        detail::validate_history<Exec>(history, owned_);
        return weights;
    }
};

} // namespace krona::transport::scalar::transient
