#pragma once

#include <krona/boundary/detail/support.hpp>
#include <krona/mesh/localmesh.hpp>

#include <limits>
#include <span>
#include <string>
#include <vector>

namespace krona::boundary {

/// @brief Absolute face interval [begin,end); patch_id == -1 denotes unlabelled faces.
struct PatchRange {
    mesh::PatchId patch_id = -1;
    LocalIndex begin = 0, end = 0;
    [[nodiscard]] bool empty() const noexcept { return begin == end; }
};

struct PatchEntry {
    PatchRange range;
    std::string name;
};

/// @brief Owns small host metadata; stores no per-face maps and borrows no MeshPart arrays.
class PatchCatalog {
public:
    explicit PatchCatalog(const mesh::MeshPart& m)
    {
        detail::guarded([&] {
            detail::require(m.faces_grouped && m.n_own >= 0 && m.n_cells >= m.n_own &&
                            m.n_internal_faces >= 0 && m.n_internal_faces <= m.n_inner_faces &&
                            m.n_inner_faces <= m.n_faces,
                            "Scalar boundary: requires valid counts and grouped faces");
            detail::require(m.face_patch.size() == static_cast<std::size_t>(m.n_faces),
                            "Scalar boundary: face_patch extent mismatch");
            detail::require(m.patches.size() <= static_cast<std::size_t>(std::numeric_limits<mesh::PatchId>::max()),
                            "Scalar boundary: too many patch labels");
            patches_.reserve(m.patches.size());
            for (std::size_t p = 0; p < m.patches.size(); ++p)
                patches_.push_back({{static_cast<mesh::PatchId>(p), m.n_inner_faces, m.n_inner_faces},
                                    m.patches[p].name});
            unlabelled_ = {-1, m.n_inner_faces, m.n_inner_faces};
            std::vector<bool> seen(m.patches.size(), false);
            bool seen_unlabelled = false;
            for (LocalIndex begin = m.n_inner_faces; begin < m.n_faces;) {
                const auto id = m.face_patch[static_cast<std::size_t>(begin)];
                detail::require(id >= -1 && (id == -1 || static_cast<std::size_t>(id) < patches_.size()),
                                "Scalar boundary: invalid patch ID on boundary face");
                LocalIndex end = begin + 1;
                while (end < m.n_faces && m.face_patch[static_cast<std::size_t>(end)] == id) ++end;
                if (id == -1) {
                    detail::require(!seen_unlabelled, "Scalar boundary: unlabelled faces must be contiguous");
                    seen_unlabelled = true;
                    unlabelled_ = {id, begin, end};
                } else {
                    const std::size_t p = static_cast<std::size_t>(id);
                    detail::require(!seen[p], "Scalar boundary: each patch must be a contiguous face interval");
                    seen[p] = true;
                    patches_[p].range = {id, begin, end};
                }
                begin = end;
            }
            detail::require(m.n_labeled_bfaces_g <= m.n_bfaces_g,
                            "Scalar boundary: invalid global boundary counts");
            needs_unlabelled_ = seen_unlabelled || m.n_labeled_bfaces_g < m.n_bfaces_g;
        });
    }

    [[nodiscard]] std::span<const PatchEntry> patches() const noexcept { return patches_; }
    [[nodiscard]] PatchRange unlabelled() const noexcept { return unlabelled_; }
    [[nodiscard]] bool needs_unlabelled() const noexcept { return needs_unlabelled_; }

private:
    std::vector<PatchEntry> patches_;
    PatchRange unlabelled_;
    bool needs_unlabelled_ = false;
};

} // namespace krona::boundary
