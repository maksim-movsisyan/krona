#pragma once

#include <krona/boundary/scalar/config.hpp>
#include <krona/boundary/scalar/detail/kernels.hpp>
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
    using Space = typename Exec::memory_space;
    using Mesh = mesh::compute::MeshView<Space>;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using Condition = std::variant<std::monostate, FixedValue<Real>, NormalGradient<Real>,
                                   FixedFlux<Real>, CustomConditions...>;

public:
    /// @brief Copies patch labels/ranges and validates borrowed compute geometry once.
    Manager(const mesh::MeshPart& host_mesh, Mesh mesh_view, Exec exec) : catalog_(host_mesh), exec_(exec)
    {
        detail::require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
                        "Scalar boundary: requires active Kokkos");
        const auto& s = mesh_view.sizes;
        detail::require(s.faces_grouped && s.n_own == host_mesh.n_own && s.n_cells == host_mesh.n_cells &&
                        s.n_faces == host_mesh.n_faces && s.n_inner_faces == host_mesh.n_inner_faces,
                        "Scalar boundary: host/compute mesh counts differ");
        detail::require(mesh_view.has(mesh::compute::MeshData::FaceCells | mesh::compute::MeshData::Geometry),
                        "Scalar boundary: attach face connectivity and cell/face geometry");
        const std::size_t faces = static_cast<std::size_t>(s.n_faces);
        const std::size_t owned = static_cast<std::size_t>(s.n_own);
        const std::size_t cells = static_cast<std::size_t>(s.n_cells);
        const std::size_t boundary = static_cast<std::size_t>(s.n_faces - s.n_inner_faces);
        detail::require(cells <= std::numeric_limits<std::size_t>::max() - boundary,
                        "Scalar boundary: extended size overflows size_t");
        required_size_ = cells + boundary;
        n_own_ = s.n_own;
        n_faces_ = s.n_faces;
        geometry_ = {s.n_cells, s.n_inner_faces, mesh_view.connectivity.face_owner,
                     mesh_view.geometry.cell_centroid_x, mesh_view.geometry.cell_centroid_y,
                     mesh_view.geometry.cell_centroid_z, mesh_view.geometry.face_centroid_x,
                     mesh_view.geometry.face_centroid_y, mesh_view.geometry.face_centroid_z,
                     mesh_view.geometry.face_normal_x, mesh_view.geometry.face_normal_y,
                     mesh_view.geometry.face_normal_z, mesh_view.geometry.face_area};
        detail::require(geometry_.face_owner.extent(0) == faces && geometry_.face_x.extent(0) == faces &&
                        geometry_.face_y.extent(0) == faces && geometry_.face_z.extent(0) == faces &&
                        geometry_.normal_x.extent(0) == faces && geometry_.normal_y.extent(0) == faces &&
                        geometry_.normal_z.extent(0) == faces && geometry_.face_area.extent(0) == faces &&
                        geometry_.cell_x.extent(0) >= owned && geometry_.cell_y.extent(0) >= owned &&
                        geometry_.cell_z.extent(0) >= owned,
                        "Scalar boundary: geometry extent mismatch");
        detail::guarded([&] {
            conditions_.resize(catalog_.patches().size());
            std::size_t bad = 0;
            Kokkos::parallel_reduce("boundary/scalar/validate_geometry", Policy(exec_, s.n_inner_faces, s.n_faces),
                detail::ValidateGeometry<Space, Real>{geometry_, s.n_own}, bad);
            detail::require(bad == 0, "Scalar boundary: invalid owner, unit normal, area or positive normal distance");
            exec_.fence("KRONA scalar boundary: setup complete");
        });
    }

    Manager(const mesh::MeshPart& host_mesh, Mesh mesh_view, const Configuration& config, Exec exec)
        : Manager(host_mesh, mesh_view, exec) { configure(config); }

    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager& operator=(Manager&&) = delete;

    /// @brief Assigns a built-in or one of CustomConditions; copied parameters may change between applies.
    template<class C>
    void set(mesh::PatchId patch_id, C condition)
    {
        static_assert(std::is_constructible_v<Condition, C>, "Add the custom condition type to Manager's template parameters");
        detail::require(patch_id >= 0 && static_cast<std::size_t>(patch_id) < conditions_.size(),
                        "Scalar boundary: unknown patch ID");
        validate_condition(condition);
        detail::guarded([&] { conditions_[static_cast<std::size_t>(patch_id)] = std::move(condition); });
    }

    template<class C>
    void set_unlabelled(C condition)
    {
        static_assert(std::is_constructible_v<Condition, C>);
        validate_condition(condition);
        detail::guarded([&] { unlabelled_ = std::move(condition); });
    }

    /// @brief Binds all labelled patches by ID and exact name; requires explicit unlabelled handling when present.
    void configure(const Configuration& config)
    {
        detail::guarded([&] {
            detail::require(config.patches.size() == conditions_.size(),
                            "Scalar boundary: configuration must describe every mesh patch");
            std::vector<bool> seen(conditions_.size(), false);
            for (const auto& entry : config.patches) {
                detail::require(entry.patch_id >= 0 && static_cast<std::uint64_t>(entry.patch_id) < conditions_.size(),
                                "Scalar boundary: configuration patch ID is out of range");
                const auto p = static_cast<std::size_t>(entry.patch_id);
                detail::require(!seen[p], "Scalar boundary: duplicate configuration patch ID");
                detail::require(entry.name == catalog_.patches()[p].name,
                                "Scalar boundary: patch name does not match the mesh");
                seen[p] = true;
                conditions_[p] = from_config(entry.condition);
            }
            unlabelled_ = config.unlabelled ? from_config(*config.unlabelled) : Condition{};
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

    /// @brief Fills only boundary-face slots of a scalar FieldView; leaves owned/MPI ghost values unchanged.
    template<class Field, class Diffusivity = UniformDiffusivity<Real>, class Correction = NoCorrection>
    void apply_async(Field field, Diffusivity diffusion = {}, Correction correction = {}, Real time = 0) const
    {
        static_assert(Field::components == 1, "Use apply_component_async for a selected component");
        apply_component_async(field.component(0), diffusion, correction, time);
    }

    /// @brief Supports rank-one Kokkos Views, including LayoutStride component subviews.
    template<class View, class Diffusivity = UniformDiffusivity<Real>, class Correction = NoCorrection>
    void apply_component_async(View field, Diffusivity diffusion = {}, Correction correction = {}, Real time = 0) const
    {
        validate_view(field, required_size_);
        static_assert(!std::is_const_v<typename View::value_type>, "Boundary output must be writable");
        static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
        detail::require(std::isfinite(time), "Scalar boundary: time must be finite");
        validate_diffusivity(diffusion);
        validate_correction(correction);
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/scalar/fill", Policy(exec_, range.begin, range.end),
                detail::FillBoundary<Space, Real, View, C, Diffusivity, Correction>{
                    geometry_, field, condition, diffusion, correction, time});
        });
    }

    /// @brief Calls a device-callable sink(context, relation) per boundary face, without coefficient buffers.
    template<class Sink, class Diffusivity = UniformDiffusivity<Real>, class Correction = NoCorrection>
    void evaluate_async(Sink sink, Diffusivity diffusion = {}, Correction correction = {}, Real time = 0) const
    {
        detail::require(std::isfinite(time), "Scalar boundary: time must be finite");
        validate_diffusivity(diffusion);
        validate_correction(correction);
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/scalar/evaluate", Policy(exec_, range.begin, range.end),
                detail::EvaluateBoundary<Space, Real, Sink, C, Diffusivity, Correction>{
                    geometry_, sink, condition, diffusion, correction, time});
        });
    }

    [[nodiscard]] GeometryView<Space> geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return catalog_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return required_size_; }

private:
    PatchCatalog catalog_;
    Exec exec_;
    GeometryView<Space> geometry_;
    std::vector<Condition> conditions_;
    Condition unlabelled_;
    LocalIndex n_own_ = 0;
    LocalIndex n_faces_ = 0;
    std::size_t required_size_ = 0;

    void require_complete() const
    {
        for (const auto& condition : conditions_)
            detail::require(!std::holds_alternative<std::monostate>(condition),
                            "Scalar boundary: missing condition on a labelled patch");
        detail::require(!catalog_.needs_unlabelled() || !std::holds_alternative<std::monostate>(unlabelled_),
                        "Scalar boundary: unlabelled faces need an explicit condition");
    }

    template<class Visitor>
    static void visit(PatchRange range, const Condition& condition, Visitor& visitor)
    {
        if (range.empty()) return;
        std::visit([&](const auto& typed) {
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(typed)>, std::monostate>)
                visitor(range, typed);
        }, condition);
    }

    static void validate_condition(FixedValue<Real> c)
    { detail::require(std::isfinite(c.value), "Scalar boundary: value must be finite"); }
    static void validate_condition(NormalGradient<Real> c)
    { detail::require(std::isfinite(c.gradient), "Scalar boundary: normal gradient must be finite"); }
    static void validate_condition(FixedFlux<Real> c)
    { detail::require(std::isfinite(c.flux), "Scalar boundary: flux must be finite"); }
    // Custom conditions define their own parameter and backing-storage contracts.
    template<class C> static void validate_condition(const C&) {}

    static Condition from_config(ConditionConfig config)
    {
        detail::require(std::isfinite(config.parameter) &&
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
        detail::require(view.extent(0) >= count, "Scalar boundary: field/coefficient/gradient extent is too small");
        detail::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0),
                        "Scalar boundary: invalid View storage");
    }

    template<class R>
    void validate_diffusivity(UniformDiffusivity<R> diffusion) const
    {
        const Real value = static_cast<Real>(diffusion.value);
        detail::require(std::isfinite(value) && value >= Real{0},
                        "Scalar boundary: diffusivity must be finite and nonnegative");
        bool fixed_flux = std::holds_alternative<FixedFlux<Real>>(unlabelled_);
        for (const auto& condition : conditions_)
            fixed_flux = fixed_flux || std::holds_alternative<FixedFlux<Real>>(condition);
        detail::require(!fixed_flux || value > Real{0}, "Scalar boundary: FIXED_FLUX requires positive diffusivity");
    }

    template<class V>
    void validate_diffusivity(FaceDiffusivity<V> diffusion) const
    {
        validate_view(diffusion.values, static_cast<std::size_t>(n_faces_));
        // Per-face numerical validity is a caller contract; do not force an application-time reduction/fence.
    }
    template<class D> static void validate_diffusivity(const D&) {}
    static void validate_correction(NoCorrection) {}
    template<class X, class Y, class Z>
    void validate_correction(GradientCorrection<X, Y, Z> correction) const
    {
        const auto owned = static_cast<std::size_t>(n_own_);
        validate_view(correction.x, owned); validate_view(correction.y, owned); validate_view(correction.z, owned);
    }
    template<class C> static void validate_correction(const C&) {}
};

} // namespace krona::boundary::scalar
