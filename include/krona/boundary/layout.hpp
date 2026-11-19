#pragma once

#include <krona/boundary/detail/kernels.hpp>
#include <krona/boundary/patches.hpp>
#include <krona/execution/utils.hpp>

#include <limits>
#include <type_traits>

namespace krona::boundary {

/// @brief Shares one patch catalog, geometry view and execution instance among boundary managers.
/// @tparam Exec Kokkos execution space used to check geometry and run the boundary kernels.
/// @tparam Real Floating-point type used for shared boundary geometry calculations.
/// @details The layout owns the catalog and borrows the mesh arrays. Keep the layout and the mesh
/// storage alive while managers and submitted kernels use them.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class BoundaryLayout {
    static_assert(std::is_floating_point_v<Real>);
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using real_type = Real;

    /// @brief Builds the catalog and waits for the local device geometry check before returning.
    /// @param host Grouped host mesh supplying the patch names and face ranges.
    /// @param view Device view of the same local mesh, with FaceCells and Geometry attached.
    /// @param exec Execution instance used for validation and later boundary submissions.
    BoundaryLayout(const mesh::host::MeshPart& host,
        mesh::device::MeshView<memory_space> view, Exec exec = {}) : catalog_(host), exec_(exec) {
        execution::require_kokkos();
        const auto& s = view.sizes;
        mpi::require(s.faces_grouped && s.n_own == host.n_own && s.n_cells == host.n_cells &&
            s.n_faces == host.n_faces && s.n_internal_faces == host.n_internal_faces &&
            s.n_inner_faces == host.n_inner_faces, "Boundary layout: host/device mesh counts differ");

        // An extended field stores all local cells first, then one entry per boundary face.
        const std::size_t cells = static_cast<std::size_t>(s.n_cells);
        const std::size_t boundary_faces = static_cast<std::size_t>(s.n_faces-s.n_inner_faces);
        mpi::require(cells <= std::numeric_limits<std::size_t>::max()-boundary_faces,
                     "Boundary layout: extended size overflows size_t");
        required_size_ = cells+boundary_faces;
        owned_ = static_cast<std::size_t>(s.n_own);
        mpi::guarded([&] {
            execution::complete(exec_, [&] { geometry_ = detail::attach_geometry<Real>(exec_, view); });
        });
    }

    // Managers borrow the layout at a fixed address.
    BoundaryLayout(const BoundaryLayout&) = delete;
    BoundaryLayout& operator=(const BoundaryLayout&) = delete;
    BoundaryLayout(BoundaryLayout&&) = delete;
    BoundaryLayout& operator=(BoundaryLayout&&) = delete;

    /// @brief Returns the patch names and local face ranges used for dispatch.
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return catalog_; }

    /// @brief Returns a copy of the borrowed mesh descriptor for passing into kernels.
    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return geometry_; }

    /// @brief Returns the execution instance shared by boundary submissions for this layout.
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }

    /// @brief Returns the minimum extent of an extended field on this rank, including ghost cells and boundary faces.
    [[nodiscard]] std::size_t required_field_size() const noexcept { return required_size_; }

    /// @brief Returns the number of owned cells for sizing owner-only inputs such as gradients.
    [[nodiscard]] std::size_t owned_cells() const noexcept { return owned_; }
private:
    PatchCatalog catalog_;
    Exec exec_;
    GeometryView<memory_space> geometry_;
    std::size_t required_size_ = 0, owned_ = 0;
};

} // namespace krona::boundary
