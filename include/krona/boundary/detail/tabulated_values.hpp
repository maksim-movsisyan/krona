#pragma once

#include <krona/boundary/profiles.hpp>
#include <krona/execution/utils.hpp>
#include <krona/io/boundary_config/profile_reader.hpp>
#include <krona/mesh/host/mesh_part.hpp>

#include <filesystem>

namespace krona::boundary::detail {

/// @brief Owns a patch's sampled profile values or stores one constant inline.
/// @tparam Real Floating-point type of the constant and sampled values.
/// @tparam Space Kokkos memory space holding the samples used by the device callbacks.
template<class Real, class Space>
struct TabulatedPatchValues {
    Kokkos::View<const Real*, Space> storage;   ///< Owning view keeping the sampled array alive.
    profile::PatchValues<Real, Space> provider; ///< Reads a sample by local face ID.
    Real constant{};                            ///< Inline value used in constant mode.
    bool sampled = false;                       ///< Selects the array in sampled mode.

    /// @brief Reads the selected parameter for this face.
    /// @tparam Context Face context exposing the local mesh face ID as face.
    /// @param context Face on the provider's patch when sampled.
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& context) const {
        return sampled ? provider(context) : constant;
    }

    /// @brief Checks the sampled array's patch metadata or the inline constant.
    /// @tparam Exec Execution space that will read the provider.
    /// @param range Local range of the patch receiving this parameter.
    template<class Exec>
    void validate(PatchRange range) const {
        if (sampled) provider.template validate<Exec>(range);
        else profile::Constant<Real>{constant}.template validate<Exec>(range);
    }
};

/// @brief Loads a CSV profile collectively, samples local face centroids and uploads the resulting array.
/// @tparam Real Floating-point type stored in the device array.
/// @tparam Exec Kokkos execution space used for the upload.
/// @tparam Convert Host callback that checks or converts an interpolated double value to Real.
/// @param path CSV file opened by rank zero of comm.
/// @param range Local face range where the profile will be used.
/// @param host Host mesh whose face numbering matches range.
/// @param comm Mesh communicator. All ranks participate, including those with an empty local range.
/// @param exec Execution instance on which the copy is submitted and completed before return.
/// @param convert Applies the condition's numeric rules to each sample.
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

        // Store the samples in patch order so provider(context) can use face ID minus range.begin.
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
