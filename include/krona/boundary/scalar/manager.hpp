#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/boundary/scalar/config.hpp>
#include <krona/boundary/scalar/detail/kernels.hpp>
#include <krona/boundary/detail/kernels.hpp>
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
/// Setup is blocking and local. Apply is asynchronous; borrowed buffers must outlive submitted work.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
class Manager {
    static_assert(std::is_floating_point_v<Real>);
    using memory_space = typename Exec::memory_space;
    using mesh_view_type = mesh::compute::MeshView<memory_space>;
    using policy_type = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using condition_type = std::variant<std::monostate, FixedValue<Real>, NormalGradient<Real>,
                                        FixedFlux<Real>, CustomConditions...>;

public:
    /// @brief Copies patch labels/ranges and validates borrowed compute geometry once.
    Manager(const mesh::MeshPart& host_mesh, mesh_view_type mesh_view, Exec exec) : catalog_(host_mesh), exec_(exec)
    {
        execution::require_kokkos();
        const auto& s = mesh_view.sizes;
        mpi::require(s.faces_grouped && s.n_own == host_mesh.n_own && s.n_cells == host_mesh.n_cells &&
                     s.n_faces == host_mesh.n_faces && s.n_inner_faces == host_mesh.n_inner_faces,
                     "Scalar boundary: host/compute mesh counts differ");
        const std::size_t cells = static_cast<std::size_t>(s.n_cells);
        const std::size_t boundary_count = static_cast<std::size_t>(s.n_faces - s.n_inner_faces);
        mpi::require(cells <= std::numeric_limits<std::size_t>::max() - boundary_count,
                     "Scalar boundary: extended size overflows size_t");
        required_size_ = cells + boundary_count;
        n_own_ = s.n_own;
        mpi::guarded([&] {
            conditions_.resize(catalog_.patches().size());
            execution::complete(exec_, [&] {
                geometry_ = boundary::detail::attach_geometry<Real>(exec_, mesh_view);
            });
        });
    }

    Manager(const mesh::MeshPart& host_mesh, mesh_view_type mesh_view, const Configuration& config, Exec exec)
        : Manager(host_mesh, mesh_view, exec) { configure(config); }

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
        validate_condition(condition);
        mpi::guarded([&] { conditions_[static_cast<std::size_t>(patch_id)] = std::move(condition); });
    }

    template<class C>
    void set_unlabelled(C condition)
    {
        static_assert(std::is_constructible_v<condition_type, C>);
        validate_condition(condition);
        mpi::guarded([&] { unlabelled_ = std::move(condition); });
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
                mpi::require(entry.name == catalog_.patches()[p].name, "Scalar boundary: patch name does not match the mesh");
                seen[p] = true;
                conditions_[p] = from_config(entry.condition);
            }
            unlabelled_ = config.unlabelled ? from_config(*config.unlabelled) : condition_type{};
            require_complete();
        });
    }

    /// @brief Calls visitor(range, typed_condition) on the host for every nonempty local patch.
    /// The same dispatch supports field filling and solver-specific implicit assembly.
    template<class Visitor>
    void for_each_patch(Visitor&& visitor) const
    {
        require_complete();
        for (std::size_t p = 0; p < conditions_.size(); ++p)
            visit(catalog_.patches()[p].range, conditions_[p], visitor);
        visit(catalog_.unlabelled(), unlabelled_, visitor);
    }

    /// @brief Calls a device-callable sink(context, typed_condition); no reconstruction or flux law is imposed.
    template<class Sink>
    void for_each_face_async(Sink sink, Real time = 0) const
    {
        mpi::require(std::isfinite(time), "Scalar boundary: time must be finite");
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/scalar/for_each_face", policy_type(exec_, range.begin, range.end),
                boundary::detail::ForEachFace<memory_space, Real, Sink, C>{geometry_, sink, condition, time});
        });
    }

    /// @brief Fills boundary slots only; FixedFlux requires an explicit flux-to-gradient law.
    template<class Field, class Law = NoFluxLaw, class Correction = NoCorrection>
    void apply_async(Field field, Law law = {}, Correction correction = {}, Real time = 0) const
    {
        static_assert(Field::components == 1, "Use apply_component_async for a selected component");
        apply_component_async(field.component(0), law, correction, time);
    }

    /// @brief Supports writable rank-one Kokkos Views, including strided component subviews.
    template<class View, class Law = NoFluxLaw, class Correction = NoCorrection>
    void apply_component_async(View field, Law law = {}, Correction correction = {}, Real time = 0) const
    {
        validate_view(field, required_size_);
        static_assert(!std::is_const_v<typename View::value_type>, "Boundary output must be writable");
        static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
        validate_relation_inputs<Law>(correction);
        for_each_face_async(detail::FillBoundary<Real, View, Law, Correction>{field, law, correction}, time);
    }

    /// @brief Calls sink(context, relation); returned normal gradients already include the correction.
    template<class Sink, class Law = NoFluxLaw, class Correction = NoCorrection>
    void evaluate_async(Sink sink, Law law = {}, Correction correction = {}, Real time = 0) const
    {
        validate_relation_inputs<Law>(correction);
        for_each_face_async(detail::EvaluateBoundary<Real, Sink, Law, Correction>{sink, law, correction}, time);
    }

    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return catalog_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return required_size_; }

private:
    PatchCatalog catalog_;
    Exec exec_;
    GeometryView<memory_space> geometry_;
    std::vector<condition_type> conditions_;
    condition_type unlabelled_;
    LocalIndex n_own_ = 0;
    std::size_t required_size_ = 0;

    void require_complete() const
    {
        for (const auto& condition : conditions_)
            mpi::require(!std::holds_alternative<std::monostate>(condition),
                         "Scalar boundary: missing condition on a labelled patch");
        mpi::require(!catalog_.needs_unlabelled() || !std::holds_alternative<std::monostate>(unlabelled_),
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

    static void validate_condition(FixedValue<Real> c)
    { mpi::require(std::isfinite(c.value), "Scalar boundary: value must be finite"); }
    static void validate_condition(NormalGradient<Real> c)
    { mpi::require(std::isfinite(c.gradient), "Scalar boundary: normal gradient must be finite"); }
    static void validate_condition(FixedFlux<Real> c)
    { mpi::require(std::isfinite(c.flux), "Scalar boundary: flux must be finite"); }
    // Custom conditions define their own parameter and backing-storage contracts.
    template<class C> static void validate_condition(const C&) {}

    static condition_type from_config(ConditionConfig config)
    {
        mpi::require(std::isfinite(config.parameter) &&
                     std::abs(config.parameter) <= static_cast<double>(std::numeric_limits<Real>::max()),
                     "Scalar boundary: parameter is not representable in the selected precision");
        const Real value = static_cast<Real>(config.parameter);
        switch (config.kind) {
            case Kind::FixedValue: return FixedValue<Real>{value};
            case Kind::NormalGradient: return NormalGradient<Real>{value};
            case Kind::FixedFlux: return FixedFlux<Real>{value};
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
        if constexpr (!FluxLaw<Law, Real>) {
            bool needs_law = std::holds_alternative<FixedFlux<Real>>(unlabelled_);
            for (const auto& condition : conditions_)
                needs_law = needs_law || std::holds_alternative<FixedFlux<Real>>(condition);
            mpi::require(!needs_law,
                "Scalar boundary: FIXED_FLUX requires a flux law with normal_gradient(context, flux)");
        }
        // Law/provider values are caller contracts; no application-time reduction or fence.
        validate_correction(correction);
    }

    static void validate_correction(NoCorrection) {}
    template<class X, class Y, class Z>
    void validate_correction(GradientCorrection<X, Y, Z> correction) const
    {
        const std::size_t owned = static_cast<std::size_t>(n_own_);
        validate_view(correction.x, owned); validate_view(correction.y, owned); validate_view(correction.z, owned);
    }
    template<class C> static void validate_correction(const C&) {}
};

} // namespace krona::boundary::scalar
