#pragma once

#include <krona/boundary/detail/kernels.hpp>
#include <krona/boundary/patches.hpp>
#include <krona/execution/utils.hpp>

#include <limits>
#include <type_traits>

namespace krona::boundary {

/// @brief Shared host patch catalog and borrowed geometry. Construction is local and blocking.
/// Keep the layout and mesh allocations alive and unchanged while managers/kernels use them.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class BoundaryLayout {
    static_assert(std::is_floating_point_v<Real>);
public:
    using execution_space = Exec;
    using real_type = Real;
    using memory_space = typename Exec::memory_space;

    BoundaryLayout(const mesh::host::MeshPart& host,
        mesh::device::MeshView<memory_space> view, Exec exec = {}) : catalog_(host), exec_(exec)
    {
        execution::require_kokkos();
        const auto& s = view.sizes;
        mpi::require(s.faces_grouped && s.n_own == host.n_own && s.n_cells == host.n_cells &&
            s.n_faces == host.n_faces && s.n_internal_faces == host.n_internal_faces &&
            s.n_inner_faces == host.n_inner_faces, "Boundary layout: host/device mesh counts differ");
        const auto cells = static_cast<std::size_t>(s.n_cells);
        const auto boundary_faces = static_cast<std::size_t>(s.n_faces-s.n_inner_faces);
        mpi::require(cells <= std::numeric_limits<std::size_t>::max()-boundary_faces,
                     "Boundary layout: extended size overflows size_t");
        required_size_ = cells+boundary_faces;
        owned_ = static_cast<std::size_t>(s.n_own);
        mpi::guarded([&] {
            execution::complete(exec_, [&] { geometry_ = detail::attach_geometry<Real>(exec_, view); });
        });
    }
    BoundaryLayout(const BoundaryLayout&) = delete;
    BoundaryLayout& operator=(const BoundaryLayout&) = delete;
    BoundaryLayout(BoundaryLayout&&) = delete;
    BoundaryLayout& operator=(BoundaryLayout&&) = delete;

    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return catalog_; }
    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return exec_; }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return required_size_; }
    [[nodiscard]] std::size_t owned_cells() const noexcept { return owned_; }
private:
    PatchCatalog catalog_;
    Exec exec_;
    GeometryView<memory_space> geometry_;
    std::size_t required_size_ = 0, owned_ = 0;
};

} // namespace krona::boundary
