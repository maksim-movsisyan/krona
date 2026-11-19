#pragma once

#include <krona/boundary/detail/config_binding.hpp>
#include <krona/boundary/layout.hpp>
#include <krona/boundary/physical/config.hpp>
#include <krona/boundary/physical/detail/condition_cast.hpp>
#include <krona/boundary/physical/detail/kernels.hpp>
#include <krona/boundary/physical/detail/validation.hpp>

#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::physical {

/// @brief Assigns physical conditions to mesh patches and dispatches their concrete types to Kokkos.
/// @tparam Model Flow model, for example Compressible with an EOS and a characteristic policy.
/// @tparam Exec Kokkos execution space used for boundary kernels.
/// @tparam Real Floating-point type of the face context and boundary calculations.
/// @tparam CustomConditions Additional condition types, including built-in conditions with profiled parameters.
/// @details The manager owns the model and condition objects and borrows the layout and mesh arrays.
/// Fields and callbacks are supplied for each launch. Assign conditions and call finalize() before use.
template<class Model, class Exec = execution::ExecutionSpace, class Real = ComputeReal,
         class... CustomConditions>
class Manager {
  public:
    /// @brief Memory space of the mesh arrays.
    using memory_space = typename Exec::memory_space;
    /// @brief Flow model stored by the manager.
    using model_type = Model;
    /// @brief Primitive state with temperature when the flow model includes it.
    using state_type = Primitive<Real, Model::thermal>;
    /// @brief Shared catalog, mesh geometry and execution instance.
    using layout_type = BoundaryLayout<Exec, Real>;

  private:
    using policy_type = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using condition_type =
        detail::ConditionVariant<Real, std::monostate, std::remove_cvref_t<CustomConditions>...>;
    static_assert(std::is_copy_constructible_v<condition_type>,
                  "Physical boundary conditions/providers must be copy constructible");

  public:
    /// @brief Checks the model and creates an unassigned slot for every patch.
    /// @param layout Layout that must outlive the manager and its submissions.
    /// @param model Flow model to use for this boundary.
    Manager(const layout_type& layout, Model model) : layout_(&layout), model_(std::move(model)) {
        execution::require_kokkos();
        mpi::guarded([&] {
            model_.validate();
            const auto count = catalog().patches().size();
            conditions_.reserve(count);
            // A captured lambda can be copy-constructible without supporting assignment.
            for (std::size_t patch = 0; patch < count; ++patch)
                conditions_.emplace_back(std::in_place_type<std::monostate>);
        });
    }
    /// @brief Creates a manager with numeric conditions from a configuration.
    /// @param layout Shared mesh layout used to match patch IDs and names.
    /// @param model Flow model to use for this boundary.
    /// @param config Conditions for the mesh patches. Call finalize() after binding.
    Manager(const layout_type& layout, Model model, const Configuration& config)
        : Manager(layout, std::move(model)) {
        configure(config);
    }
    Manager(layout_type&&, Model) = delete;
    Manager(layout_type&&, Model, const Configuration&) = delete;
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager& operator=(Manager&&) = delete;

    /// @brief Replaces the condition of a labelled patch and marks the bindings for finalization.
    /// @tparam C Built-in condition or an additional type registered in CustomConditions.
    /// @param id Mesh patch ID as stored in the catalog.
    /// @param condition Condition to store, with constant direction vectors normalized when applicable.
    template<class C> void set(mesh::PatchId id, C condition) {
        static_assert(std::is_constructible_v<condition_type, C>, "Register the condition type in Manager");
        mpi::require(id >= 0 && static_cast<std::size_t>(id) < conditions_.size(),
                     "Physical boundary: unknown patch ID");
        mpi::guarded([&] {
            auto prepared = detail::prepare(std::move(condition), model_);
            validate_metadata(prepared, catalog().patches()[static_cast<std::size_t>(id)].range);
            conditions_[static_cast<std::size_t>(id)].template emplace<C>(std::move(prepared));
            finalized_ = false;
        });
    }
    /// @brief Sets the common condition for boundary faces that have no label.
    /// @tparam C Built-in condition or an additional type registered in CustomConditions.
    /// @param condition Condition to bind. Finalize before submitting boundary work.
    template<class C> void set_unlabelled(C condition) {
        static_assert(std::is_constructible_v<condition_type, C>);
        mpi::guarded([&] {
            auto prepared = detail::prepare(std::move(condition), model_);
            validate_metadata(prepared, catalog().unlabelled());
            unlabelled_.template emplace<C>(std::move(prepared));
            finalized_ = false;
        });
    }
    /// @brief Binds numeric conditions for all labelled patches and the optional unlabelled group.
    /// @param config Conditions to match to the layout by patch ID and name.
    void configure(const Configuration& config) {
        configure(config, [](const TabulatedConditionConfig&, PatchRange, const auto&) {
            mpi::fatal("Physical boundary: CSV profile requires configure_tabulated");
        });
    }

    /// @brief Binds a configuration and delegates file-backed conditions to a host resolver.
    /// @tparam Resolver Callback that calls assign(typed_condition) exactly once for a profile.
    /// @param config Conditions to match to the layout by patch ID and name.
    /// @param resolver Receives the file-backed specification, local face range and assignment callback.
    template<class Resolver> void configure(const Configuration& config, Resolver&& resolver) {
        mpi::guarded([&] {
            const auto resolve = [&](const ConfigCondition& condition, PatchRange range, const auto& assign) {
                std::visit(
                    [&](const auto& c) {
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(c)>,
                                                     TabulatedConditionConfig>) {
                            boundary::detail::resolve_profile_once(resolver, c, range, assign,
                                                                   "Physical boundary");
                        } else
                            assign(detail::condition_cast<Real>(c));
                    },
                    condition);
            };
            boundary::detail::bind_configured_patches(
                catalog(), config.patches,
                [&](const PatchConfig& patch, PatchRange range) {
                    resolve(patch.condition, range,
                            [&](auto c) { set(static_cast<mesh::PatchId>(patch.patch_id), std::move(c)); });
                });
            unlabelled_.template emplace<std::monostate>();
            if (config.unlabelled)
                resolve(*config.unlabelled, catalog().unlabelled(),
                        [&](auto c) { set_unlabelled(std::move(c)); });
            finalized_ = false;
        });
    }
    /// @brief Checks that all required patches are bound and records whether a heat-flux law is needed.
    /// @details Runs locally on the host and includes provider metadata checks for empty local ranges.
    /// Replacing a binding requires another finalize(). Time and updated buffer contents can vary per launch.
    void finalize() {
        mpi::guarded([&] {
            require_complete();
            needs_law_ = false;
            const auto inspect = [&](PatchRange range, const condition_type& condition) {
                std::visit(
                    [&](const auto& c) {
                        using C = std::remove_cvref_t<decltype(c)>;
                        if constexpr (!std::is_same_v<C, std::monostate>) {
                            validate_metadata(c, range);
                            needs_law_ = needs_law_ || requires_flux_law<C>;
                        }
                    },
                    condition);
            };
            for (std::size_t p = 0; p < conditions_.size(); ++p)
                inspect(catalog().patches()[p].range, conditions_[p]);
            inspect(catalog().unlabelled(), unlabelled_);
            finalized_ = true;
        });
    }
    /// @brief Reports whether the current patch bindings have passed finalize().
    [[nodiscard]] bool finalized() const noexcept {
        return finalized_;
    }

    /// @brief Calls a host callback once for each nonempty local patch.
    /// @tparam Visitor Callable accepting (range, condition) for every supported condition type.
    /// @param visitor Receives the patch range and the concrete condition bound to it.
    template<class Visitor> void for_each_patch(Visitor&& visitor) const {
        require_finalized();
        mpi::guarded([&] {
            for (std::size_t p = 0; p < conditions_.size(); ++p)
                visit(catalog().patches()[p].range, conditions_[p], visitor);
            visit(catalog().unlabelled(), unlabelled_, visitor);
        });
    }
    /// @brief Launches a device callback on every local boundary face, passing its typed condition.
    /// @tparam Sink Device-callable object accepting (context, condition) for every supported condition type.
    /// @param sink Callback that decides how to process the face.
    /// @param time Time copied into each face context for profile evaluation.
    /// @details Selects the condition on the host and submits one kernel per nonempty patch.
    template<class Sink> void for_each_face_async(Sink sink, Real time = 0) const {
        mpi::require(std::isfinite(time), "Physical boundary: time must be finite");
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/physical/for_each_face",
                                 policy_type(execution_space(), range.begin, range.end),
                                 boundary::detail::ForEachFace<memory_space, Real, Sink, C>{
                                     geometry_view(), sink, condition, time});
        });
    }

    /// @brief Queues evaluation of both boundary states for a caller-supplied device callback.
    /// @tparam Sink Device-callable object accepting (context, evaluation).
    /// @tparam Reader Device callable returning the interior state from (context, condition).
    /// @tparam Law Converts a prescribed heat-flux density to a normal temperature derivative.
    /// @tparam Correction Groups the tangential increments for pressure, velocity and temperature.
    /// @param sink Runs for each face whose evaluation succeeds.
    /// @param reader Supplies the interior state, optionally reconstructed by a numerical operator.
    /// @param status Open diagnostic batch on the layout's execution instance.
    /// @param law Heat-flux conversion used by the condition.
    /// @param correction Explicit corrections used for the physical trace.
    /// @param time Time at which to evaluate the profiles.
    template<class Sink, class Reader, class Law = scalar::NoFluxLaw, class Correction = Corrections<>>
    void evaluate_async(Sink sink, Reader reader, Status<Exec>& status, Law law = {},
                        Correction correction = {}, Real time = 0) const {
        validate_inputs<Law>(correction);
        mpi::guarded([&] {
            if constexpr (requires { reader.template validate<Exec>(layout_->owned_cells()); })
                reader.template validate<Exec>(layout_->owned_cells());
        });
        submit(detail::CheckedSink<Sink>{sink}, reader, status, law, correction, time);
    }
    /// @brief Queues writes of the physical trace into the boundary slots of extended primitive fields.
    /// @tparam Fields Writable binding of primitive components with the same thermal capability as Model.
    /// @tparam Law Converts a prescribed heat-flux density to a normal temperature derivative.
    /// @tparam Correction Groups the tangential increments for pressure, velocity and temperature.
    /// @param fields Supplies the owner state and receives the physical trace.
    /// @param status Open diagnostic batch on the layout's execution instance.
    /// @param law Heat-flux conversion used by the selected condition.
    /// @param correction Explicit corrections used for the physical trace.
    /// @param time Time at which to evaluate the profiles.
    template<class Fields, class Law = scalar::NoFluxLaw, class Correction = Corrections<>>
    void apply_async(Fields fields, Status<Exec>& status, Law law = {}, Correction correction = {},
                     Real time = 0) const {
        static_assert(Fields::thermal == Model::thermal,
                      "Primitive binding must match the model's thermal capability");
        mpi::guarded([&] { fields.template validate<Exec, true>(required_field_size()); });
        validate_inputs<Law>(correction);
        submit(detail::Fill<Fields>{fields}, OwnerReader{fields}, status, law, correction, time);
    }

    /// @brief Returns the model used by direct kernel consumers such as the Euler operator.
    [[nodiscard]] const Model& model() const noexcept {
        return model_;
    }
    /// @brief Returns the execution instance on which this manager submits work.
    [[nodiscard]] const Exec& execution_space() const noexcept {
        return layout_->execution_space_instance();
    }
    /// @brief Returns the layout's host patch catalog.
    [[nodiscard]] const PatchCatalog& catalog() const noexcept {
        return layout_->catalog();
    }
    /// @brief Returns the borrowed geometry used to build face contexts.
    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept {
        return layout_->geometry_view();
    }
    /// @brief Returns the count of owned cells, MPI ghosts and physical face slots.
    [[nodiscard]] std::size_t required_field_size() const noexcept {
        return layout_->required_field_size();
    }
    /// @brief Returns the layout shared with any scalar or physical managers on this mesh.
    [[nodiscard]] const layout_type& layout() const noexcept {
        return *layout_;
    }

  private:
    const layout_type* layout_;
    Model model_;
    std::vector<condition_type> conditions_;
    condition_type unlabelled_ = std::monostate{};
    bool finalized_ = false, needs_law_ = false;

    template<class C> static void validate_metadata(const C& c, PatchRange range) {
        profile::validate<Exec>(c, range);
    }
    void require_finalized() const {
        mpi::require(finalized_, "Physical boundary: call finalize() after assigning conditions");
    }
    void require_complete() const {
        for (const auto& c : conditions_)
            mpi::require(!std::holds_alternative<std::monostate>(c),
                         "Physical boundary: missing condition on a labelled patch");
        mpi::require(!catalog().needs_unlabelled() || !std::holds_alternative<std::monostate>(unlabelled_),
                     "Physical boundary: unlabelled faces need an explicit condition");
    }
    template<class Visitor>
    static void visit(PatchRange range, const condition_type& condition, Visitor& visitor) {
        if (range.empty())
            return;
        std::visit(
            [&](const auto& c) {
                if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(c)>, std::monostate>)
                    visitor(range, c);
            },
            condition);
    }
    template<class C> void validate_correction(const C& c) const {
        if constexpr (requires { c.gradients.template validate<Exec>(layout_->owned_cells()); })
            c.gradients.template validate<Exec>(layout_->owned_cells());
        else if constexpr (requires { c.template validate<Exec>(layout_->owned_cells()); })
            c.template validate<Exec>(layout_->owned_cells());
    }
    template<class X, class Y, class Z> void validate_correction(const VectorCorrection<X, Y, Z>& c) const {
        validate_correction(c.x);
        validate_correction(c.y);
        validate_correction(c.z);
    }
    template<class Law, class C> void validate_inputs(const C& correction) const {
        require_finalized();
        if constexpr (!scalar::FluxLaw<Law, Real>)
            mpi::require(!needs_law_, "Physical boundary: heat-flux wall requires an explicit flux law");
        mpi::guarded([&] {
            validate_correction(correction.pressure);
            validate_correction(correction.velocity);
            if constexpr (Model::thermal)
                validate_correction(correction.temperature);
        });
    }
    template<class Sink, class Reader, class Law, class Correction>
    void submit(Sink sink, Reader reader, Status<Exec>& status, Law law, Correction correction,
                Real time) const {
        using Flags = typename Status<Exec>::flags_type;
        for_each_face_async(
            detail::Evaluate<Model, Reader, Sink, Law, Correction, Flags>{
                model_, reader, sink, law, correction, status.device_flags(execution_space())},
            time);
    }
};

} // namespace krona::boundary::physical
