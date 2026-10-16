#pragma once

#include <krona/boundary/patches.hpp>
#include <krona/boundary/physical/config.hpp>
#include <krona/boundary/physical/detail/kernels.hpp>
#include <krona/execution/utils.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::physical {

/// @brief Mesh buffers required by the borrowed boundary geometry; no auxiliary arrays.
[[nodiscard]] constexpr mesh::compute::MeshData requirements() noexcept
{ return mesh::compute::MeshData::FaceCells | mesh::compute::MeshData::Geometry; }

/// @brief Host patch dispatch and portable face kernels, independent of fields, flow models and EOS.
/// Setup is local/blocking. Submission is asynchronous on exec; mesh/field buffers remain borrowed.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
class Manager {
public:
    using Space = typename Exec::memory_space;
    using Mesh = mesh::compute::MeshView<Space>;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using Condition = std::variant<std::monostate, VelocityInlet<Real>, TotalInlet<Real>,
        SupersonicInlet<Real>, PressureOutlet<Real>, InletOutlet<Real>, SupersonicOutlet,
        AdiabaticWall<Real>, IsothermalWall<Real>, SlipWall<Real>, Symmetry, Extrapolate, CustomConditions...>;
    static_assert(std::is_floating_point_v<Real>);

    Manager(const mesh::MeshPart& host_mesh, Mesh view, Exec exec)
        : catalog_(host_mesh), exec_(exec)
    {
        execution::require_kokkos();
        const auto& s = view.sizes;
        mpi::require(s.faces_grouped && s.n_own == host_mesh.n_own && s.n_cells == host_mesh.n_cells &&
                     s.n_faces == host_mesh.n_faces && s.n_inner_faces == host_mesh.n_inner_faces &&
                     s.n_internal_faces == host_mesh.n_internal_faces,
                     "Physical boundary: host/compute mesh counts differ");
        mpi::require(view.has(requirements()), "Physical boundary: attach face connectivity and geometry");
        const auto faces = static_cast<std::size_t>(s.n_faces), owned = static_cast<std::size_t>(s.n_own);
        const auto cells = static_cast<std::size_t>(s.n_cells);
        const auto boundary = static_cast<std::size_t>(s.n_faces - s.n_inner_faces);
        mpi::require(cells <= std::numeric_limits<std::size_t>::max() - boundary,
                     "Physical boundary: extended field size overflows");
        required_size_ = cells + boundary;
        n_own_ = s.n_own;
        geometry_.storage = {s.n_cells, s.n_inner_faces, view.connectivity.face_owner,
            view.geometry.cell_centroid_x, view.geometry.cell_centroid_y, view.geometry.cell_centroid_z,
            view.geometry.face_centroid_x, view.geometry.face_centroid_y, view.geometry.face_centroid_z,
            view.geometry.face_normal_x, view.geometry.face_normal_y, view.geometry.face_normal_z,
            view.geometry.face_area};
        const auto& g = geometry_.storage;
        require_array(g.face_owner, faces); require_array(g.face_x, faces); require_array(g.face_y, faces);
        require_array(g.face_z, faces); require_array(g.normal_x, faces); require_array(g.normal_y, faces);
        require_array(g.normal_z, faces); require_array(g.face_area, faces);
        require_array(g.cell_x, owned, false); require_array(g.cell_y, owned, false);
        require_array(g.cell_z, owned, false);
        mpi::guarded([&] {
            conditions_.resize(catalog_.patches().size());
            std::size_t bad = 0;
            // Validate in geometry precision before any conversion to fluid arithmetic.
            Kokkos::parallel_reduce("boundary/physical/validate_geometry",
                Policy(exec_, s.n_inner_faces, s.n_faces),
                boundary::detail::ValidateGeometry<Space, GeometryReal>{g, s.n_own}, bad);
            mpi::require(bad == 0, "Physical boundary: invalid owner, normal, area or normal distance");
            exec_.fence("KRONA physical boundary: setup complete");
        });
    }

    Manager(const mesh::MeshPart& host_mesh, Mesh view, const Configuration& config, Exec exec)
        : Manager(host_mesh, view, exec) { configure(config); }
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager& operator=(Manager&&) = delete;

    template<class C>
    void set(mesh::PatchId id, C condition)
    {
        static_assert(std::is_constructible_v<Condition, C>, "Register custom types in Manager's template arguments");
        mpi::require(id >= 0 && static_cast<std::size_t>(id) < conditions_.size(),
                     "Physical boundary: unknown patch ID");
        validate_condition(condition);
        mpi::guarded([&] { conditions_[static_cast<std::size_t>(id)] = std::move(condition); });
    }
    template<class C>
    void set_unlabelled(C condition)
    {
        static_assert(std::is_constructible_v<Condition, C>);
        validate_condition(condition);
        mpi::guarded([&] { unlabelled_ = std::move(condition); });
    }

    /// @brief Binds IDs and exact mesh names; every global labelled patch must be specified.
    void configure(const Configuration& config)
    {
        mpi::guarded([&] {
            mpi::require(config.patches.size() == conditions_.size(),
                         "Physical boundary: configuration must describe every patch");
            std::vector<bool> seen(conditions_.size(), false);
            for (const auto& entry : config.patches) {
                mpi::require(entry.patch_id >= 0 && static_cast<std::uint64_t>(entry.patch_id) < conditions_.size(),
                             "Physical boundary: configuration patch ID out of range");
                const auto p = static_cast<std::size_t>(entry.patch_id);
                mpi::require(!seen[p] && entry.name == catalog_.patches()[p].name,
                             "Physical boundary: duplicate patch ID or mismatched name");
                seen[p] = true; conditions_[p] = from_config(entry.condition);
            }
            unlabelled_ = config.unlabelled ? from_config(*config.unlabelled) : Condition{};
            require_complete();
        });
    }

    /// @brief Host callback(range, typed_condition) for each nonempty local patch.
    template<class Visitor>
    void for_each_patch(Visitor&& visitor) const
    {
        require_complete();
        for (std::size_t p = 0; p < conditions_.size(); ++p)
            dispatch(catalog_.patches()[p].range, conditions_[p], visitor);
        dispatch(catalog_.unlabelled(), unlabelled_, visitor);
    }

    /// @brief Visits one labelled patch on the host, including an empty local interval.
    template<class Visitor>
    void visit(mesh::PatchId id, Visitor&& visitor) const
    {
        mpi::require(id >= 0 && static_cast<std::size_t>(id) < conditions_.size(),
                     "Physical boundary: unknown patch ID");
        const auto p = static_cast<std::size_t>(id);
        dispatch(catalog_.patches()[p].range, conditions_[p], visitor, true);
    }
    template<class Visitor>
    void visit_unlabelled(Visitor&& visitor) const
    { dispatch(catalog_.unlabelled(), unlabelled_, visitor, true); }

    /// @brief Device callback(context, typed_condition); performs no automatic field access.
    template<class Function, class Transform = UnchangedContext>
    void for_each_face_async(Function function, Real time = 0, Transform transform = {}) const
    {
        mpi::require(std::isfinite(time), "Physical boundary: time must be finite");
        if constexpr (requires { transform.template validate<Exec>(static_cast<std::size_t>(n_own_)); })
            transform.template validate<Exec>(static_cast<std::size_t>(n_own_));
        for_each_patch([&](PatchRange range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/physical/for_each_face", Policy(exec_, range.begin, range.end),
                detail::ForEachFace<Space, Real, C, Function, Transform>{geometry_, condition, function, transform, time});
        });
    }

    /// @brief Evaluates states through a supplied closure; the device sink controls how to consume results.
    template<class Reader, class Closure, class Sink, class Transform = UnchangedContext>
    void evaluate_async(Reader reader, Closure closure, Sink sink, Real time = 0, Transform transform = {}) const
    {
        static_assert(std::is_same_v<typename Closure::real_type, Real>, "Closure and manager arithmetic must match");
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(reader(std::size_t{}))>, typename Closure::state_type>,
                      "Reader must return the closure's primitive state type");
        // Validate all host descriptors before submitting any patch work.
        if constexpr (requires { closure.validate(); }) closure.validate();
        require_complete();
        for (const auto& condition : conditions_) validate_closure_condition(closure, condition);
        if (!std::holds_alternative<std::monostate>(unlabelled_)) validate_closure_condition(closure, unlabelled_);
        for_each_face_async(detail::Evaluate<Reader, Closure, Sink>{reader, closure, sink}, time, transform);
    }

    /// @brief Fills extended face slots; owned cells, MPI ghosts and unused trailing storage remain untouched.
    template<class Fields, class Closure, class Transform = UnchangedContext>
    void apply_async(Fields fields, Closure closure, Real time = 0, Transform transform = {}) const
    {
        if constexpr (requires { Fields::constant_density; })
            static_assert(!Closure::compressible || !Fields::constant_density,
                          "Compressible output requires a writable density field");
        if constexpr (requires { fields.template validate<Exec>(required_size_); })
            fields.template validate<Exec>(required_size_);
        evaluate_async(fields, closure, detail::Store<Space, Fields>{geometry_, fields}, time, transform);
    }

    [[nodiscard]] GeometryView<Space> geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return catalog_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return required_size_; }
    void fence() const { exec_.fence("KRONA physical boundary: complete"); }

private:
    PatchCatalog catalog_;
    Exec exec_;
    GeometryView<Space> geometry_;
    std::vector<Condition> conditions_;
    Condition unlabelled_;
    std::size_t required_size_ = 0;
    LocalIndex n_own_ = 0;

    template<class A>
    static void require_array(const A& a, std::size_t count, bool exact = true)
    {
        mpi::require((exact ? a.extent(0) == count : a.extent(0) >= count) &&
                     (count == 0 || a.data() != nullptr), "Physical boundary: geometry extent/storage mismatch");
    }
    void require_complete() const
    {
        for (const auto& c : conditions_)
            mpi::require(!std::holds_alternative<std::monostate>(c), "Physical boundary: missing labelled condition");
        mpi::require(!catalog_.needs_unlabelled() || !std::holds_alternative<std::monostate>(unlabelled_),
                     "Physical boundary: unlabelled faces need an explicit condition");
    }
    template<class Visitor>
    static void dispatch(PatchRange range, const Condition& condition, Visitor& visitor, bool include_empty = false)
    {
        if (range.empty() && !include_empty) return;
        mpi::require(!std::holds_alternative<std::monostate>(condition), "Physical boundary: patch has no condition");
        std::visit([&](const auto& c) {
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(c)>, std::monostate>) visitor(range, c);
        }, condition);
    }
    template<class Closure>
    static void validate_closure_condition(const Closure& closure, const Condition& condition)
    {
        std::visit([&](const auto& typed) {
            using C = std::remove_cvref_t<decltype(typed)>;
            if constexpr (!std::is_same_v<C, std::monostate> && requires { closure.validate_condition(typed); })
                closure.validate_condition(typed);
        }, condition);
    }
    static void finite(Real value)
    { mpi::require(std::isfinite(value), "Physical boundary: nonfinite parameter"); }
    static void positive(Real value)
    { mpi::require(std::isfinite(value) && value > Real{0}, "Physical boundary: parameter must be positive"); }
    static void validate_vector(Vector3<Real> v)
    { finite(v.x); finite(v.y); finite(v.z); }
    static void validate_temperature(Temperature<Real> t)
    {
        mpi::require(t.kind == ThermalKind::Extrapolated || t.kind == ThermalKind::FixedTemperature,
                     "Physical boundary: unknown thermal condition");
        if (t.kind == ThermalKind::FixedTemperature) positive(t.value);
    }
    static void validate_pressure(Real p) { finite(p); }
    static void validate_condition(VelocityInlet<Real> c)
    { validate_vector(c.velocity); validate_temperature(c.temperature); }
    static void validate_condition(TotalInlet<Real> c)
    {
        positive(c.pressure); positive(c.temperature); validate_vector(c.direction);
        const Real norm = c.direction.dot(c.direction);
        mpi::require(std::isfinite(norm) && std::abs(norm - Real{1}) <= Real{256} * std::numeric_limits<Real>::epsilon(),
                     "Physical boundary: inlet direction must be a unit vector");
    }
    static void validate_condition(SupersonicInlet<Real> c)
    { validate_pressure(c.pressure); positive(c.temperature); validate_vector(c.velocity); }
    static void validate_condition(PressureOutlet<Real> c) { validate_pressure(c.pressure); }
    static void validate_condition(InletOutlet<Real> c)
    { validate_pressure(c.pressure); validate_vector(c.backflow_velocity); validate_temperature(c.backflow_temperature); }
    static void validate_condition(AdiabaticWall<Real> c) { validate_vector(c.velocity); }
    static void validate_condition(IsothermalWall<Real> c) { validate_vector(c.velocity); positive(c.temperature); }
    static void validate_condition(SlipWall<Real> c)
    { validate_vector(c.velocity); validate_temperature(c.temperature); }
    template<class C> static void validate_condition(const C&) {}

    static Real convert(double value)
    {
        mpi::require(std::isfinite(value) && std::abs(value) <= static_cast<double>(std::numeric_limits<Real>::max()),
                     "Physical boundary: parameter is not representable");
        return static_cast<Real>(value);
    }
    static Real required(const std::optional<double>& value)
    { mpi::require(value.has_value(), "Physical boundary: missing required configuration parameter"); return convert(*value); }
    static Condition from_config(const ConditionConfig& config)
    {
        const bool velocity = config.kind == Kind::VelocityInlet || config.kind == Kind::SupersonicInlet ||
            config.kind == Kind::InletOutlet || (config.kind == Kind::AdiabaticWall || config.kind == Kind::IsothermalWall) || config.kind == Kind::SlipWall;
        const bool pressure = config.kind == Kind::SupersonicInlet || config.kind == Kind::InletOutlet ||
            config.kind == Kind::PressureOutlet;
        const bool total = config.kind == Kind::TotalInlet;
        mpi::require((!config.pressure || pressure) && (!config.temperature || (velocity && config.kind != Kind::AdiabaticWall)) &&
                     (!config.total_pressure || total) && (!config.total_temperature || total),
                     "Physical boundary: inapplicable configuration parameter");
        for (const auto v : config.velocity)
            mpi::require(velocity || v == 0, "Physical boundary: velocity is inapplicable to this condition");
        for (const auto d : config.direction)
            mpi::require(total || d == 0, "Physical boundary: direction is inapplicable to this condition");
        const Vector3<Real> v{convert(config.velocity[0]), convert(config.velocity[1]), convert(config.velocity[2])};
        const Temperature<Real> t = config.temperature
            ? Temperature<Real>{ThermalKind::FixedTemperature, convert(*config.temperature)} : Temperature<Real>{};
        Condition result;
        switch (config.kind) {
            case Kind::VelocityInlet: result = VelocityInlet<Real>{v, t}; break;
            case Kind::TotalInlet:
                result = TotalInlet<Real>{required(config.total_pressure), required(config.total_temperature),
                    {convert(config.direction[0]), convert(config.direction[1]), convert(config.direction[2])}}; break;
            case Kind::SupersonicInlet:
                result = SupersonicInlet<Real>{required(config.pressure), required(config.temperature), v}; break;
            case Kind::PressureOutlet: result = PressureOutlet<Real>{required(config.pressure)}; break;
            case Kind::InletOutlet: result = InletOutlet<Real>{required(config.pressure), v, t}; break;
            case Kind::SupersonicOutlet: result = SupersonicOutlet{}; break;
            case Kind::AdiabaticWall: result = AdiabaticWall<Real>{v}; break;
            case Kind::IsothermalWall: result = IsothermalWall<Real>{v, required(config.temperature)}; break;
            case Kind::SlipWall: result = SlipWall<Real>{v, t}; break;
            case Kind::Symmetry: result = Symmetry{}; break;
            case Kind::Extrapolate: result = Extrapolate{}; break;
            default: mpi::fatal("Physical boundary: unknown condition kind");
        }
        std::visit([](const auto& c) { validate_condition(c); }, result);
        return result;
    }
};

} // namespace krona::boundary::physical
