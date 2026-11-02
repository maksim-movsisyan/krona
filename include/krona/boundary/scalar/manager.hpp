#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/boundary/scalar/config.hpp>
#include <krona/boundary/scalar/detail/kernels.hpp>
#include <krona/boundary/layout.hpp>
#include <krona/numerics/gradient/correction.hpp>
#include <krona/boundary/patches.hpp>
#include <krona/fields/fields_view.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::scalar {

/// @brief Host dispatch per patch; typed conditions execute in portable Kokkos kernels.
/// Borrows a shared BoundaryLayout; owns conditions, never a field. Finalize before dispatch.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
class Manager {
    static_assert(std::is_floating_point_v<Real>);
    using memory_space = typename Exec::memory_space;
    using policy_type = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using condition_type = std::variant<std::monostate, FixedValue<Real>, NormalGradient<Real>,
                                        FixedFlux<Real>, std::remove_cvref_t<CustomConditions>...>;

public:
    using layout_type = BoundaryLayout<Exec, Real>;
    explicit Manager(const layout_type& layout) : layout_(&layout)
    {
        execution::require_kokkos();
        mpi::guarded([&] { conditions_.resize(catalog().patches().size()); });
    }
    Manager(const layout_type& layout, const Configuration& config) : Manager(layout)
    { configure(config); }
    
    Manager(layout_type&&) = delete;
    Manager(layout_type&&, const Configuration&) = delete;

    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager& operator=(Manager&&) = delete;

    /// @brief Assigns a built-in or one of CustomConditions; copied parameters may change between applies.
    template<class C>
    void set(mesh::PatchId patch_id, C condition)
    {
        static_assert(std::is_constructible_v<condition_type, C>, "Add the custom condition type to Manager's template parameters");
        mpi::require(patch_id >= 0 && static_cast<std::size_t>(patch_id) < conditions_.size(),
                     "Scalar boundary: unknown patch ID");
        mpi::guarded([&] {
            validate_condition(condition, catalog().patches()[static_cast<std::size_t>(patch_id)].range);
            conditions_[static_cast<std::size_t>(patch_id)].template emplace<C>(std::move(condition));
            finalized_ = false;
        });
    }

    template<class C>
    void set_unlabelled(C condition)
    {
        static_assert(std::is_constructible_v<condition_type, C>);
        mpi::guarded([&] {
            validate_condition(condition, catalog().unlabelled());
            unlabelled_.template emplace<C>(std::move(condition));
            finalized_ = false;
        });
    }

    /// @brief Binds all labelled patches by ID and exact name; requires explicit unlabelled handling when present.
    void configure(const Configuration& config)
    {
        mpi::guarded([&] {
            mpi::require(config.patches.size() == conditions_.size(),
                         "Scalar boundary: configuration must describe every mesh patch");
            std::vector<bool> seen(conditions_.size(), false);
            for (const auto& entry : config.patches) {
                mpi::require(entry.patch_id >= 0 && static_cast<std::uint64_t>(entry.patch_id) < conditions_.size(),
                             "Scalar boundary: configuration patch ID is out of range");
                const std::size_t p = static_cast<std::size_t>(entry.patch_id);
                mpi::require(!seen[p], "Scalar boundary: duplicate configuration patch ID");
                mpi::require(entry.name == catalog().patches()[p].name, "Scalar boundary: patch name does not match the mesh");
                seen[p] = true;
                assign_config(conditions_[p], entry.condition);
            }
            if (config.unlabelled) assign_config(unlabelled_, *config.unlabelled);
            else unlabelled_.template emplace<std::monostate>();
            finalized_ = false;
        });
    }

    /// @brief Host-only completeness/provider checks. No kernels, fences or MPI communication.
    void finalize()
    {
        mpi::guarded([&] {
            require_complete();
            needs_law_ = false;
            const auto inspect = [&](PatchRange range, const condition_type& condition) {
                std::visit([&](const auto& typed) {
                    using C = std::remove_cvref_t<decltype(typed)>;
                    if constexpr (!std::is_same_v<C, std::monostate>) {
                        validate_condition(typed, range);
                        needs_law_ = needs_law_ || requires_flux_law<C>;
                    }
                }, condition);
            };
            for (std::size_t p = 0; p < conditions_.size(); ++p)
                inspect(catalog().patches()[p].range, conditions_[p]);
            inspect(catalog().unlabelled(), unlabelled_);
            finalized_ = true;
        });
    }
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }

    /// @brief Calls visitor(range, typed_condition) on the host for every nonempty local patch.
    /// The same dispatch supports field filling and solver-specific implicit assembly.
    template<class Visitor>
    void for_each_patch(Visitor&& visitor) const
    {
        require_finalized();
        mpi::guarded([&] {
            for (std::size_t p = 0; p < conditions_.size(); ++p)
                visit(catalog().patches()[p].range, conditions_[p], visitor);
            visit(catalog().unlabelled(), unlabelled_, visitor);
        });
    }

    /// @brief Calls a device-callable sink(context, typed_condition); no reconstruction or flux law is imposed.
    template<class Sink>
    void for_each_face_async(Sink sink, Real time = 0) const
    {
        mpi::require(std::isfinite(time), "Scalar boundary: time must be finite");
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/scalar/for_each_face", policy_type(execution_space(), range.begin, range.end),
                boundary::detail::ForEachFace<memory_space, Real, Sink, C>{geometry_view(), sink, condition, time});
        });
    }

    /// @brief Fills boundary slots only; FixedFlux requires an explicit flux-to-gradient law.
    template<class Field, class Law = NoFluxLaw, class Correction = numerics::gradient::NoCorrection>
    void apply_async(Field field, Status<Exec>& status, Law law = {}, Correction correction = {}, Real time = 0) const
    {
        static_assert(Field::components == 1, "Use apply_component_async for a selected component");
        apply_component_async(field.component(0), status, law, correction, time);
    }

    /// @brief Supports writable rank-one Kokkos Views, including strided component subviews.
    template<class View, class Law = NoFluxLaw, class Correction = numerics::gradient::NoCorrection>
    void apply_component_async(View field, Status<Exec>& status, Law law = {}, Correction correction = {}, Real time = 0) const
    {
        validate_view(field, required_field_size());
        static_assert(!std::is_const_v<typename View::value_type>, "Boundary output must be writable");
        static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
        validate_relation_inputs<Law>(correction);
        for_each_face_async(detail::FillBoundary<Real, View, Law, Correction, typename Status<Exec>::flags_type>{
            field, law, correction, status.device_flags(execution_space())}, time);
    }

    /// @brief Calls sink(context, relation); returned normal gradients already include the correction.
    template<class Sink, class Law = NoFluxLaw, class Correction = numerics::gradient::NoCorrection>
    void evaluate_async(Sink sink, Status<Exec>& status, Law law = {}, Correction correction = {}, Real time = 0) const
    {
        validate_relation_inputs<Law>(correction);
        for_each_face_async(detail::EvaluateBoundary<Real, Sink, Law, Correction, typename Status<Exec>::flags_type>{
            sink, law, correction, status.device_flags(execution_space())}, time);
    }

    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return layout_->geometry_view(); }
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return layout_->catalog(); }
    [[nodiscard]] const Exec& execution_space() const noexcept { return layout_->execution_space_instance(); }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return layout_->required_field_size(); }
    [[nodiscard]] const layout_type& layout() const noexcept { return *layout_; }

private:
    const layout_type* layout_;
    std::vector<condition_type> conditions_;
    condition_type unlabelled_;
    bool finalized_ = false, needs_law_ = false;

    void require_finalized() const
    { mpi::require(finalized_, "Scalar boundary: call finalize() after assigning conditions"); }

    void require_complete() const
    {
        for (const auto& condition : conditions_)
            mpi::require(!std::holds_alternative<std::monostate>(condition),
                         "Scalar boundary: missing condition on a labelled patch");
        mpi::require(!catalog().needs_unlabelled() || !std::holds_alternative<std::monostate>(unlabelled_),
                     "Scalar boundary: unlabelled faces need an explicit condition");
    }

    template<class Visitor>
    static void visit(PatchRange range, const condition_type& condition, Visitor& visitor)
    {
        if (range.empty()) return;
        std::visit([&](const auto& typed) {
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(typed)>, std::monostate>)
                visitor(range, typed);
        }, condition);
    }

    template<class C>
    static void validate_condition(const C& condition, PatchRange range)
    {
        if constexpr (is_fixed_value<C>) profile::validate<Exec>(condition.value, range);
        else if constexpr (is_normal_gradient<C>) profile::validate<Exec>(condition.gradient, range);
        else if constexpr (is_fixed_flux<C>) profile::validate<Exec>(condition.flux, range);
        else if constexpr (requires { condition.template validate<Exec>(range); })
            condition.template validate<Exec>(range);
    }

    static void assign_config(condition_type& destination, ConditionConfig config)
    {
        mpi::require(std::isfinite(config.parameter) &&
                     std::abs(config.parameter) <= static_cast<double>(std::numeric_limits<Real>::max()),
                     "Scalar boundary: parameter is not representable in the selected precision");
        const Real value = static_cast<Real>(config.parameter);
        switch (config.kind) {
            case Kind::FixedValue: destination.template emplace<FixedValue<Real>>(FixedValue<Real>{value}); return;
            case Kind::NormalGradient: destination.template emplace<NormalGradient<Real>>(NormalGradient<Real>{value}); return;
            case Kind::FixedFlux: destination.template emplace<FixedFlux<Real>>(FixedFlux<Real>{value}); return;
        }
        mpi::fatal("Scalar boundary: unknown condition kind");
    }

    template<class View>
    static void validate_view(const View& view, std::size_t count)
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible,
                      "The selected execution space cannot access this View");
        mpi::require(view.extent(0) >= count, "Scalar boundary: field/coefficient/gradient extent is too small");
        mpi::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0),
                        "Scalar boundary: invalid View storage");
    }

    template<class Law, class Correction>
    void validate_relation_inputs(const Correction& correction) const
    {
        require_finalized();
        if constexpr (!FluxLaw<Law, Real>)
            mpi::require(!needs_law_,
                "Scalar boundary: FIXED_FLUX requires a flux law with normal_gradient(context, flux)");
        mpi::guarded([&] {
            if constexpr (requires { correction.gradients.template validate<Exec>(layout_->owned_cells()); })
                correction.gradients.template validate<Exec>(layout_->owned_cells());
            else if constexpr (requires { correction.template validate<Exec>(layout_->owned_cells()); })
                correction.template validate<Exec>(layout_->owned_cells());
        });
    }
};

} // namespace krona::boundary::scalar
