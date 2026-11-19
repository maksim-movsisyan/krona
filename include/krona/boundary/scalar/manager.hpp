#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/boundary/scalar/config.hpp>
#include <krona/boundary/scalar/detail/condition_binding.hpp>
#include <krona/boundary/scalar/detail/kernels.hpp>
#include <krona/boundary/layout.hpp>
#include <krona/numerics/gradient/correction.hpp>
#include <krona/boundary/patches.hpp>
#include <krona/fields/fields_view.hpp>

#include <cmath>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::scalar {

/// @brief Stores the patch conditions for one scalar and dispatches their concrete types to Kokkos.
/// @tparam Exec Kokkos execution space used for boundary work.
/// @tparam Real Floating-point type used for face geometry and relations. Field storage may use another type.
/// @tparam CustomConditions Additional condition types, including built-in formulas with custom providers.
/// @details Share a BoundaryLayout between managers on the same mesh. Each manager owns its
/// condition objects and receives the field or callback at dispatch. Assign conditions and call
/// finalize() before dispatch. Keep the layout, mesh and borrowed input/output buffers alive through use.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
class Manager {
    static_assert(std::is_floating_point_v<Real>);
    using memory_space = typename Exec::memory_space;
    using policy_type = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;

    // monostate marks an unassigned patch. The host picks the concrete type before launch.
    using condition_type = std::variant<std::monostate, FixedValue<Real>, NormalGradient<Real>,
                                        FixedFlux<Real>, std::remove_cvref_t<CustomConditions>...>;

public:
    /// @brief Shared patch catalog, geometry and execution instance used by this manager.
    using layout_type = BoundaryLayout<Exec, Real>;

    /// @brief Creates a manager with an empty condition slot for each labelled patch.
    /// @param layout Existing layout that must outlive the manager and its submitted work.
    explicit Manager(const layout_type& layout) : layout_(&layout) {
        execution::require_kokkos();
        mpi::guarded([&] { conditions_.resize(catalog().patches().size()); });
    }

    /// @brief Creates a manager and binds numeric conditions from one scalar's configuration.
    /// @param layout Existing layout that supplies the patch IDs and names.
    /// @param config Conditions to bind. Call finalize() before using them.
    Manager(const layout_type& layout, const Configuration& config) : Manager(layout)
    { configure(config); }

    // Guard against binding a temporary layout to the manager's borrowed pointer.
    Manager(layout_type&&) = delete;
    Manager(layout_type&&, const Configuration&) = delete;

    // Keep the conditions and their finalization state in one host object.
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager& operator=(Manager&&) = delete;

    /// @brief Replaces the condition on a labelled patch. Call finalize() before the next dispatch.
    /// @tparam C Built-in condition or one of the types registered in CustomConditions.
    /// @param patch_id ID from the mesh patch catalog.
    /// @param condition Condition to bind. Its parameters are copied into later kernel submissions.
    template<class C>
    void set(mesh::PatchId patch_id, C condition) {
        static_assert(std::is_constructible_v<condition_type, C>, "Add the custom condition type to Manager's template parameters");
        mpi::require(patch_id >= 0 && static_cast<std::size_t>(patch_id) < conditions_.size(),
                     "Scalar boundary: unknown patch ID");
        mpi::guarded([&] {
            detail::validate_condition<Exec>(condition, catalog().patches()[static_cast<std::size_t>(patch_id)].range);
            conditions_[static_cast<std::size_t>(patch_id)].template emplace<C>(std::move(condition));
            finalized_ = false;
        });
    }

    /// @brief Replaces the shared condition for boundary faces without a label.
    /// @tparam C Built-in condition or one of the types registered in CustomConditions.
    /// @param condition Condition to bind. Call finalize() before the next dispatch.
    template<class C>
    void set_unlabelled(C condition) {
        static_assert(std::is_constructible_v<condition_type, C>);
        mpi::guarded([&] {
            detail::validate_condition<Exec>(condition, catalog().unlabelled());
            unlabelled_.template emplace<C>(std::move(condition));
            finalized_ = false;
        });
    }

    /// @brief Replaces all patch bindings using numeric parameters from one scalar's configuration.
    /// @param config One entry per labelled mesh patch, plus the unlabelled condition when supplied.
    /// @details Match both the ID and the name of each patch. Use configure_tabulated() for CSV
    /// profiles. Finalize after binding, with an explicit condition for any unlabelled faces.
    void configure(const Configuration& config) {
        configure(config, [](const ConditionConfig&, PatchRange, const auto&) {
            mpi::fatal("Scalar boundary: CSV profile requires configure_tabulated");
        });
    }

    /// @brief Binds the configuration and uses a host callback to turn each profile path into a condition.
    /// @tparam Resolver Callback accepting (spec, range, assign) or (spec, assign).
    /// @param config Conditions for all labelled patches and optionally the unlabelled group.
    /// @param resolver Called for profile paths, including empty local ranges. Call assign(condition)
    /// exactly once and use a condition type supported by the manager.
    template<class Resolver>
    void configure(const Configuration& config, Resolver&& resolver) {
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
                detail::assign_resolved<Exec, Real>(conditions_[p], entry.condition, catalog().patches()[p].range, resolver);
            }
            if (config.unlabelled)
                detail::assign_resolved<Exec, Real>(unlabelled_, *config.unlabelled, catalog().unlabelled(), resolver);
            else unlabelled_.template emplace<std::monostate>();
            finalized_ = false;
        });
    }

    /// @brief Checks the patch bindings and records whether relation evaluation requires a flux law.
    /// @details This step is local and runs on the host. It validates every labelled patch, including
    /// empty local ranges. Changing the time or the contents of an existing provider buffer needs
    /// only correct execution ordering. Replacing a binding requires another finalize().
    void finalize() {
        mpi::guarded([&] {
            require_complete();
            needs_law_ = false;
            const auto inspect = [&](PatchRange range, const condition_type& condition) {
                std::visit([&](const auto& typed) {
                    using C = std::remove_cvref_t<decltype(typed)>;
                    if constexpr (!std::is_same_v<C, std::monostate>) {
                        detail::validate_condition<Exec>(typed, range);
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

    /// @brief Reports whether the current bindings have passed finalize().
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }

    /// @brief Calls a host visitor once per nonempty local patch, including the unlabelled range.
    /// @tparam Visitor Callable that accepts (range, condition) for every condition type in the manager.
    /// @param visitor Receives the local face range and its concrete condition.
    template<class Visitor>
    void for_each_patch(Visitor&& visitor) const {
        require_finalized();
        mpi::guarded([&] {
            for (std::size_t p = 0; p < conditions_.size(); ++p)
                visit(catalog().patches()[p].range, conditions_[p], visitor);
            visit(catalog().unlabelled(), unlabelled_, visitor);
        });
    }

    /// @brief Queues a callback for each local boundary face. The callback gets the condition itself.
    /// @tparam Sink Device-callable object accepting (context, condition) for every supported condition type.
    /// @param sink Callback that decides how to use the face and condition.
    /// @param time Time passed to every face context in this submission.
    /// @details Submits one kernel per nonempty local patch on the layout's execution instance and
    /// returns after submission. Use evaluate_async() when the callback needs the affine relation.
    template<class Sink>
    void for_each_face_async(Sink sink, Real time = 0) const {
        mpi::require(std::isfinite(time), "Scalar boundary: time must be finite");
        // Resolve the condition type on the host before dispatching a typed kernel for the patch.
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/scalar/for_each_face", policy_type(execution_space(), range.begin, range.end),
                boundary::detail::ForEachFace<memory_space, Real, Sink, C>{geometry_view(), sink, condition, time});
        });
    }

    /// @brief Queues writes to the boundary face slots of a scalar field.
    /// @tparam Field Writable single-component field view.
    /// @tparam Law Converts prescribed flux density to a normal derivative when a FixedFlux patch is bound.
    /// @tparam Correction Device-callable object returning the explicit tangential correction for each face.
    /// @param field Extended field with owned cells, MPI ghosts and boundary face slots in that order.
    /// @param status Open diagnostic batch on the layout's execution instance. Call check() to finish it.
    /// @param law The flux law used by flux conditions.
    /// @param correction Explicit correction, typically the owner gradient dotted with the tangential displacement.
    /// @param time Time supplied to the condition and its providers through the face context.
    template<class Field, class Law = NoFluxLaw, class Correction = numerics::gradient::NoCorrection>
    void apply_async(Field field, Status<Exec>& status, Law law = {}, Correction correction = {}, Real time = 0) const {
        static_assert(Field::components == 1, "Use apply_component_async for a selected component");
        apply_component_async(field.component(0), status, law, correction, time);
    }

    /// @brief Queues writes to the boundary slots of a selected scalar component.
    /// @tparam View Writable rank-one Kokkos view. Strided component subviews are supported.
    /// @tparam Law Converts prescribed flux density to a normal derivative when needed.
    /// @tparam Correction Device-callable object returning the explicit tangential correction for each face.
    /// @param field Component with at least required_field_size() entries. Cell values stay in place.
    /// @param status Open diagnostic batch on the layout's execution instance.
    /// @param law Flux-to-gradient conversion for prescribed fluxes.
    /// @param correction Explicit correction to the owner value at the face.
    /// @param time Time supplied to the condition and its providers through the face context.
    template<class View, class Law = NoFluxLaw, class Correction = numerics::gradient::NoCorrection>
    void apply_component_async(View field, Status<Exec>& status, Law law = {}, Correction correction = {}, Real time = 0) const {
        validate_view(field, required_field_size());
        static_assert(!std::is_const_v<typename View::value_type>, "Boundary output must be writable");
        static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
        validate_relation_inputs<Law>(correction);
        for_each_face_async(detail::FillBoundary<Real, View, Law, Correction, typename Status<Exec>::flags_type>{
            field, law, correction, status.device_flags(execution_space())}, time);
    }

    /// @brief Queues a callback with the affine face value and normal derivative, useful for matrix assembly.
    /// @tparam Sink Device-callable object accepting (context, Relation<Real>).
    /// @tparam Law Converts prescribed flux density to a normal derivative when needed.
    /// @tparam Correction Device-callable object returning the explicit tangential correction for each face.
    /// @param sink Callback receiving the face context and the corrected relation coefficients.
    /// @param status Open diagnostic batch on the layout's execution instance.
    /// @param law Flux-to-gradient conversion for prescribed fluxes.
    /// @param correction Explicit correction to the owner value at the face.
    /// @param time Time supplied to the condition and its providers through the face context.
    template<class Sink, class Law = NoFluxLaw, class Correction = numerics::gradient::NoCorrection>
    void evaluate_async(Sink sink, Status<Exec>& status, Law law = {}, Correction correction = {}, Real time = 0) const {
        validate_relation_inputs<Law>(correction);
        for_each_face_async(detail::EvaluateBoundary<Real, Sink, Law, Correction, typename Status<Exec>::flags_type>{
            sink, law, correction, status.device_flags(execution_space())}, time);
    }

    /// @brief Returns the borrowed geometry used to build face contexts.
    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return layout_->geometry_view(); }
    /// @brief Returns the layout's host catalog of patch IDs, names and local ranges.
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return layout_->catalog(); }
    /// @brief Returns the execution instance on which the manager submits work.
    [[nodiscard]] const Exec& execution_space() const noexcept { return layout_->execution_space_instance(); }
    /// @brief Returns the number of local cells, including MPI ghosts, plus the boundary face count.
    [[nodiscard]] std::size_t required_field_size() const noexcept { return layout_->required_field_size(); }
    /// @brief Returns the layout shared with any other managers on this mesh.
    [[nodiscard]] const layout_type& layout() const noexcept { return *layout_; }

private:
    const layout_type* layout_;             ///< Shared layout borrowed from the caller.
    std::vector<condition_type> conditions_; ///< Labelled conditions indexed by patch ID.
    condition_type unlabelled_;              ///< Shared binding for local boundary faces without a label.
    bool finalized_ = false;                ///< The current patch bindings passed finalize().
    bool needs_law_ = false;                ///< Cached by finalize() for relation-based submissions.

    /// @brief Reports a dispatch attempted before finalizing the current bindings.
    void require_finalized() const
    { mpi::require(finalized_, "Scalar boundary: call finalize() after assigning conditions"); }

    /// @brief Checks that every labelled patch and any globally required unlabelled group has a condition.
    void require_complete() const {
        for (const auto& condition : conditions_)
            mpi::require(!std::holds_alternative<std::monostate>(condition),
                         "Scalar boundary: missing condition on a labelled patch");
        mpi::require(!catalog().needs_unlabelled() || !std::holds_alternative<std::monostate>(unlabelled_),
                     "Scalar boundary: unlabelled faces need an explicit condition");
    }

    /// @brief Passes the concrete condition to the host visitor when the patch has work on this rank.
    /// @tparam Visitor Host callback accepting the range and concrete condition.
    /// @param range Local interval of face IDs.
    /// @param condition Condition stored for this range.
    /// @param visitor Visitor supplied to for_each_patch().
    template<class Visitor>
    static void visit(PatchRange range, const condition_type& condition, Visitor& visitor) {
        if (range.empty()) return;
        std::visit([&](const auto& typed) {
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(typed)>, std::monostate>)
                visitor(range, typed);
        }, condition);
    }

    /// @brief Checks that the execution space can access enough usable entries in the component view.
    /// @tparam View Rank-one Kokkos view for a field component.
    /// @param view Component to be written by the boundary kernels.
    /// @param count Minimum number of entries, including boundary face slots.
    template<class View>
    static void validate_view(const View& view, std::size_t count) {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible,
                      "The selected execution space cannot access this View");
        mpi::require(view.extent(0) >= count, "Scalar boundary: field/coefficient/gradient extent is too small");
        mpi::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0),
                        "Scalar boundary: invalid View storage");
    }

    /// @brief Checks the bound conditions, whether a flux law is available, and optional correction metadata.
    /// @tparam Law Type supplied for flux-to-gradient conversion.
    /// @tparam Correction Tangential correction with an optional host validation method.
    /// @param correction Correction whose storage is checked against the owned-cell count when supported.
    template<class Law, class Correction>
    void validate_relation_inputs(const Correction& correction) const {
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
