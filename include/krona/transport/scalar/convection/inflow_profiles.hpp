#pragma once

#include <krona/boundary/patches.hpp>
#include <krona/transport/scalar/convection/types.hpp>
#include <krona/transport/scalar/detail/views.hpp>

#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace krona::transport::scalar::convection {

/// @brief Host-only patch bindings; profiles contain exactly one value per local patch face.
/// Owns only small host metadata; the catalog outlives the registry, each buffer outlives its kernels.
/// Creation allocates host metadata once; set/clear change descriptors without copying data or
/// synchronizing. Contributions select a device descriptor once per patch, outside device kernels.
template<class Exec = execution::ExecutionSpace, class Fallback = FixedValueInflow>
class InflowProfiles {
    using Profile = std::variant<std::monostate, PatchInflowValues<float>, PatchInflowValues<double>>;
public:
    using execution_space = Exec;
    using fallback_type = Fallback;

    explicit InflowProfiles(const boundary::PatchCatalog& catalog, Fallback fallback = {})
        : catalog_(&catalog), fallback_(std::move(fallback))
    { mpi::guarded([&] { profiles_.resize(catalog.patches().size()); }); }

    InflowProfiles(const InflowProfiles&) = delete;
    InflowProfiles& operator=(const InflowProfiles&) = delete;
    InflowProfiles(InflowProfiles&&) = default;
    InflowProfiles& operator=(InflowProfiles&&) = default;

    /// @brief Binds a float/double rank-one View without copying it, including const/strided views.
    /// View extent must equal range.end-range.begin, including zero on ranks without this patch.
    template<class View>
    void set(mesh::PatchId patch_id, View values)
    {
        const auto p = patch_index(patch_id);
        bind(profiles_[p], catalog_->patches()[p].range, values);
    }

    template<class View>
    void set_unlabelled(View values)
    {
        mpi::require(catalog_->needs_unlabelled(), "Scalar convection: no unlabelled boundary patch");
        bind(unlabelled_, catalog_->unlabelled(), values);
    }

    /// @brief Restores the fallback policy for this patch; does not touch or release profile data.
    void clear(mesh::PatchId patch_id) { profiles_[patch_index(patch_id)] = std::monostate{}; }
    void clear_unlabelled() { unlabelled_ = std::monostate{}; }

    /// @brief A binding is specific to the manager/catalog used at setup, including its local ranges.
    void require_catalog(const boundary::PatchCatalog& catalog) const
    { mpi::require(catalog_ == &catalog, "Scalar convection: profiles are bound to a different boundary catalog"); }

    /// @brief Host dispatch only; visitor receives a compact device descriptor or the fallback.
    template<class Visitor>
    void for_patch(boundary::PatchRange range, Visitor&& visitor) const
    {
        const auto& profile = range.patch_id == -1 ? unlabelled_ : profiles_[patch_index(range.patch_id)];
        std::visit([&](const auto& typed) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(typed)>, std::monostate>) visitor(fallback_);
            else visitor(typed);
        }, profile);
    }

private:
    const boundary::PatchCatalog* catalog_;
    Fallback fallback_;
    std::vector<Profile> profiles_;
    Profile unlabelled_;

    std::size_t patch_index(mesh::PatchId patch_id) const
    {
        mpi::require(patch_id >= 0 && static_cast<std::size_t>(patch_id) < profiles_.size(),
            "Scalar convection: unknown inflow patch ID");
        return static_cast<std::size_t>(patch_id);
    }

    template<class View>
    static void bind(Profile& profile, boundary::PatchRange range, View values)
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        using Value = typename View::non_const_value_type;
        static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, double>,
            "Inflow profiles require float or double values");
        const auto count = static_cast<std::size_t>(range.end-range.begin);
        mpi::require(values.extent(0) == count, "Scalar convection: profile extent must equal local patch size");
        scalar::detail::validate_view<Exec>(values, count);
        profile = PatchInflowValues<Value>{values.data(), static_cast<std::size_t>(values.stride(0)), range.begin};
    }
};

/// @brief Default: unbound patches use FixedValue. A custom device-callable fallback is optional.
template<class Boundary, class Fallback = FixedValueInflow>
auto make_inflow_profiles(const Boundary& boundary, Fallback fallback = {})
{
    using Exec = std::remove_cvref_t<decltype(boundary.execution_space())>;
    return InflowProfiles<Exec, Fallback>(boundary.catalog(), std::move(fallback));
}

namespace detail {
template<class Inflow, class Visitor>
void visit_inflow(const Inflow& inflow, boundary::PatchRange, Visitor&& visitor) { visitor(inflow); }
template<class Exec, class Fallback, class Visitor>
void visit_inflow(const InflowProfiles<Exec, Fallback>& inflow, boundary::PatchRange range, Visitor&& visitor)
{ inflow.for_patch(range, std::forward<Visitor>(visitor)); }
} // namespace detail

} // namespace krona::transport::scalar::convection
