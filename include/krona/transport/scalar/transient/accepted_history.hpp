#pragma once

#include <krona/transport/scalar/detail/views.hpp>
#include <krona/transport/scalar/transient/mass.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace krona::transport::scalar::transient {

/// @brief Keeps the two most recent accepted scalar fields for backward Euler or BDF2.
/// After accepting a step, its field is stored at t^n and the previous t^n becomes t^(n-1).
/// Keep this object and the source views alive until queued copies and consumers finish. Use
/// the same execution instance for both, or synchronize before handing views to another instance.
/// @tparam Scalar Floating-point storage type of the accepted scalar.
/// @tparam Exec Kokkos execution space used for snapshots.
/// @tparam MassScalar Storage type for cell mass snapshots. Use void for time-independent mass.
template<class Scalar, class Exec = execution::ExecutionSpace, class MassScalar = void>
class AcceptedHistory {
    static_assert(std::is_floating_point_v<Scalar>);
    static constexpr bool stores_mass = !std::is_void_v<MassScalar>;
    static_assert(!stores_mass || std::is_floating_point_v<MassScalar>);

    using Space = typename Exec::memory_space;
    using FieldView = Kokkos::View<Scalar*, Space>;
    using ConstFieldView = Kokkos::View<const Scalar*, Space>;
    using MassValue = std::conditional_t<stores_mass, MassScalar, ComputeReal>;
    using MassView = Kokkos::View<MassValue*, Space>;
    using ConstMassView = Kokkos::View<const MassValue*, Space>;

public:
    /// @brief Allocates history buffers for the owned cells of this rank.
    /// Call accept_step_async() with the initial condition before requesting history.
    /// @param owned Number of owned cells at every time level.
    /// @param exec Execution instance shared with the transient operator.
    explicit AcceptedHistory(std::size_t owned, Exec exec = {}) : exec_(exec), owned_(owned) {
        execution::require_kokkos();
        mpi::guarded([&] {
            for (auto& field : fields_)
                field = FieldView(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "transient/accepted_field"),
                                  owned_);
            if constexpr (stores_mass) {
                for (auto& mass : masses_)
                    mass = MassView(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "transient/accepted_mass"),
                                    owned_);
            }
        });
    }

    AcceptedHistory(const AcceptedHistory&) = delete;
    AcceptedHistory& operator=(const AcceptedHistory&) = delete;

    /// @brief Returns the number of accepted layers available, capped at two.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// @brief Saves an accepted scalar when the same mass provider applies to both history levels.
    /// @tparam Field Rank-one view of Scalar values, starting with the owned cells.
    /// @param field Accepted scalar or initial condition, supplied from the working field.
    template<class Field> requires (!stores_mass)
    void accept_step_async(Field field) {
        validate_input<Scalar>(field);
        const auto slot = 1 - latest_;
        mpi::guarded([&] { copy_owned(fields_[slot], field); });
        publish(slot);
    }

    /// @brief Saves the accepted scalar and integrated cell mass from the same step.
    /// @tparam Field Rank-one view of Scalar values, starting with the owned cells.
    /// @tparam MassField Rank-one view of MassScalar values, starting with the owned cells.
    /// @param field Accepted scalar or initial condition, supplied from the working field.
    /// @param mass Accepted integrated masses M = rho*V for those same cells.
    template<class Field, class MassField> requires (stores_mass)
    void accept_step_async(Field field, MassField mass) {
        validate_input<Scalar>(field);
        validate_input<MassScalar>(mass);
        const auto slot = 1 - latest_;
        mpi::guarded([&] {
            copy_owned(fields_[slot], field);
            copy_owned(masses_[slot], mass);
        });
        publish(slot);
    }

    /// @brief Returns accepted layers with a shared, time-independent mass provider.
    /// Submit consumers before accepting another step, which can reuse the history buffers.
    /// @tparam Layers Number of accepted levels required by the formula, one for Euler or two for BDF2.
    /// @tparam Mass Device-callable provider of integrated mass in the owned cells.
    /// @param mass The same provider used by all requested levels.
    template<std::size_t Layers, class Mass> requires (!stores_mass)
    [[nodiscard]] auto history(const Mass& mass) const {
        return build_history<Layers>([&](std::size_t slot) {
            return TimeLevel{mass, ConstFieldView{fields_[slot]}};
        });
    }

    /// @brief Returns accepted layers with their saved integrated masses.
    /// Submit consumers before accepting another step, which can reuse the history buffers.
    /// @tparam Layers Number of accepted levels required by the formula, one for Euler or two for BDF2.
    template<std::size_t Layers> requires (stores_mass)
    [[nodiscard]] auto history() const {
        return build_history<Layers>([&](std::size_t slot) {
            return TimeLevel{CellMass{ConstMassView{masses_[slot]}}, ConstFieldView{fields_[slot]}};
        });
    }

private:
    Exec exec_;
    std::size_t owned_;
    std::array<FieldView, 2> fields_;
    std::array<MassView, stores_mass ? 2 : 0> masses_;
    std::size_t latest_ = 1;
    std::size_t size_ = 0;

    template<class Value, class View>
    void validate_input(const View& view) const {
        scalar::detail::validate_view<Exec>(view, owned_);
        static_assert(std::is_same_v<Value, typename View::non_const_value_type>,
                      "Scalar transient: snapshot source and history storage types must match");
    }

    template<class Destination, class Source>
    void copy_owned(const Destination& destination, const Source& source) const {
        if (owned_ != 0) {
            const auto prefix = Kokkos::subview(source, std::pair<std::size_t, std::size_t>{0, owned_});
            Kokkos::deep_copy(exec_, destination, prefix);
        }
    }

    void publish(std::size_t slot) noexcept {
        // The old t^n already occupies the other slot. Leave it in place as t^(n-1).
        latest_ = slot;
        if (size_ < 2)
            ++size_;
    }

    template<std::size_t Layers, class MakeLevel>
    auto build_history(const MakeLevel& level) const {
        static_assert(Layers == 1 || Layers == 2, "Scalar transient: request one or two accepted layers");
        mpi::require(size_ >= Layers, "Scalar transient: insufficient accepted history");
        if constexpr (Layers == 1)
            return time_history(level(latest_));
        else
            return time_history(level(latest_), level(1 - latest_));
    }
};

} // namespace krona::transport::scalar::transient
