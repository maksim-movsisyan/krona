#pragma once

#include <krona/boundary/patch_range.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mesh/host/mesh_part.hpp>

#include <limits>
#include <span>
#include <string>
#include <vector>

namespace krona::boundary {

/// @brief Name and local face range for one labelled mesh patch.
struct PatchEntry {
    PatchRange range; ///< Local range, possibly empty on this rank.
    std::string name; ///< Exact mesh patch name used when binding a configuration.
};

/// @brief Owns the patch names and face ranges used to dispatch boundary work on this rank.
/// @details The constructor copies the names and builds the ranges from the host mesh's face labels.
class PatchCatalog {
public:
    /// @brief Builds the ranges and checks that each patch occupies one contiguous interval.
    /// @param m Host mesh with grouped faces and one patch ID per local face.
    explicit PatchCatalog(const mesh::host::MeshPart& m) {
        mpi::guarded([&] {
            mpi::require(m.faces_grouped && m.n_own >= 0 && m.n_cells >= m.n_own &&
                         m.n_internal_faces >= 0 && m.n_internal_faces <= m.n_inner_faces && m.n_inner_faces <= m.n_faces,
                         "PatchCatalog: requires valid counts and grouped faces");
            mpi::require(m.face_patch.size() == static_cast<std::size_t>(m.n_faces),
                         "PatchCatalog: face_patch extent mismatch");
            mpi::require(m.patches.size() <= static_cast<std::size_t>(std::numeric_limits<mesh::PatchId>::max()),
                         "PatchCatalog: too many patch labels");

            // Keep every label in patch-ID order so even a locally empty patch can be configured.
            patches_.reserve(m.patches.size());
            for (std::size_t p = 0; p < m.patches.size(); ++p)
                patches_.push_back({{static_cast<mesh::PatchId>(p), m.n_inner_faces, m.n_inner_faces},
                                      m.patches[p].name});
            unlabelled_ = {-1, m.n_inner_faces, m.n_inner_faces};
            std::vector<bool> seen(m.patches.size(), false);
            bool seen_unlabelled = false;

            // Walk only boundary faces and record each run of equal patch IDs.
            for (LocalIndex begin = m.n_inner_faces; begin < m.n_faces;) {
                const auto id = m.face_patch[static_cast<std::size_t>(begin)];
                mpi::require(id >= -1 && (id == -1 || static_cast<std::size_t>(id) < patches_.size()),
                             "PatchCatalog: invalid patch ID on boundary face");
                LocalIndex end = begin + 1;
                while (end < m.n_faces && m.face_patch[static_cast<std::size_t>(end)] == id) ++end;
                if (id == -1) {
                    mpi::require(!seen_unlabelled, "PatchCatalog: unlabelled faces must be contiguous");
                    seen_unlabelled = true;
                    unlabelled_ = {id, begin, end};
                } else {
                    const std::size_t p = static_cast<std::size_t>(id);
                    mpi::require(!seen[p], "PatchCatalog: each patch must be a contiguous face interval");
                    seen[p] = true;
                    patches_[p].range = {id, begin, end};
                }
                begin = end;
            }
            mpi::require(m.n_labeled_bfaces_g <= m.n_bfaces_g, "PatchCatalog: invalid global boundary counts");

            // A rank with an empty local range still requires the binding if the mesh has unlabelled faces.
            needs_unlabelled_ = seen_unlabelled || m.n_labeled_bfaces_g < m.n_bfaces_g;
        });
    }

    /// @brief Returns the labelled entries indexed by patch ID, including locally empty patches.
    [[nodiscard]] std::span<const PatchEntry> patches() const noexcept { return patches_; }

    /// @brief Returns the local range for unlabelled boundary faces, possibly empty.
    [[nodiscard]] PatchRange unlabelled() const noexcept { return unlabelled_; }

    /// @brief Reports whether the boundary manager must bind a condition for unlabelled faces.
    [[nodiscard]] bool needs_unlabelled() const noexcept { return needs_unlabelled_; }

private:
    std::vector<PatchEntry> patches_;
    PatchRange unlabelled_;
    bool needs_unlabelled_ = false;
};

} // namespace krona::boundary
