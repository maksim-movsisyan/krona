#pragma once

#include <krona/boundary/profiles.hpp>
#include <krona/execution/utils.hpp>
#include <krona/io/boundary_config/profile_reader.hpp>
#include <krona/mesh/host/mesh_part.hpp>

#include <cmath>

namespace krona::boundary::detail {

/// @brief Owns the allocation borrowed by a patch provider, or stores a constant inline.
template<class Real, class Space>
struct TabulatedPatchValues {
    Kokkos::View<const Real*, Space> storage;
    profile::PatchValues<Real, Space> provider;
    Real constant{};
    bool sampled = false;

    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& context) const {
        return sampled ? provider(context) : constant;
    }

    template<class Exec>
    void validate(PatchRange range) const {
        if (sampled) provider.template validate<Exec>(range);
        else mpi::require(std::isfinite(constant), "Boundary profile: constant must be finite");
    }
};

template<class Real, class Exec, class Convert>
[[nodiscard]] auto load_patch_values(const std::filesystem::path& path, PatchRange range,
                                     const mesh::host::MeshPart& host, mpi::CommView comm,
                                     const Exec& exec, Convert convert) {
    using Space = typename Exec::memory_space;
    // Empty local patches still take part in collective profile loading.
    const auto table = io::boundary_config::read_tabulated_profile_csv(path, comm);
    Kokkos::View<Real*, Space> values;
    if (!range.empty()) {
        const std::size_t count = static_cast<std::size_t>(range.end - range.begin);
        const std::size_t end = static_cast<std::size_t>(range.end);
        mpi::require(end <= host.face_centroid_x.size() && end <= host.face_centroid_y.size() &&
                     end <= host.face_centroid_z.size(), "Boundary profile: missing host face centroids");
        values = Kokkos::View<Real*, Space>{Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                                                              "boundary/profile"), count};
        auto staging = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, values);
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t face = static_cast<std::size_t>(range.begin) + i;
            staging(i) = convert(table.evaluate(host.face_centroid_x[face], host.face_centroid_y[face],
                                                host.face_centroid_z[face]));
        }
        execution::complete(exec, [&] { Kokkos::deep_copy(exec, values, staging); });
    }
    return TabulatedPatchValues<Real, Space>{values, profile::patch_values(range, values), Real{}, true};
}

} // namespace krona::boundary::detail
