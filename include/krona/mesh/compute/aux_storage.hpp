#pragma once

#include <krona/mesh/compute/aux_view.hpp>
#include <krona/mesh/compute/detail/array_transfer.hpp>

namespace krona::mesh::compute {

/// @brief Borrows selected ready host caches on CPU; owns their copies on GPU.
/// No table/metric construction and no MPI operations are performed here.
template<class Exec = execution::ExecutionSpace>
class AuxConnectivityStorage {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using view_type = AuxConnectivityView<memory_space>;
    static constexpr bool borrows_host = detail::ArrayTransfer<Exec>::borrows_host;

    /// @brief Attaches all groups marked ready in the source cache.
    explicit AuxConnectivityStorage(const MeshPart& mesh, const MeshAuxConnectivity& source, Exec exec = {})
        : AuxConnectivityStorage(mesh, source, source.active_mask, exec) {}

    /// @brief Attaches a requested subset; missing groups are rejected.
    AuxConnectivityStorage(const MeshPart& mesh, const MeshAuxConnectivity& source, AuxConnType requested, Exec exec = {})
    {
        validate(mesh, source, requested);
        view_.active_mask = requested;
        view_.n_faces = mesh.n_faces;
        view_.n_cells_own = mesh.n_own;
        view_.n_nodes_own = mesh.n_nodes_own;
        try {
            if (has(requested, AuxConnType::CellFaces)) {
                detail::ArrayTransfer<Exec>::attach(view_.cell_faces_offsets, buffers_.cell_faces_offsets, source.cell_faces_offsets,
                    "mesh/aux/cell_faces_offsets", exec);
                detail::ArrayTransfer<Exec>::attach(view_.cell_faces, buffers_.cell_faces, source.cell_faces,
                    "mesh/aux/cell_faces", exec);
            }
            if (has(requested, AuxConnType::CellCellsByFace)) {
                detail::ArrayTransfer<Exec>::attach(view_.cell_cells_face_offsets, buffers_.cell_cells_face_offsets, source.cell_cells_face_offsets,
                    "mesh/aux/cell_cells_face_offsets", exec);
                detail::ArrayTransfer<Exec>::attach(view_.cell_cells_face, buffers_.cell_cells_face, source.cell_cells_face,
                    "mesh/aux/cell_cells_face", exec);
            }
            if (has(requested, AuxConnType::NodeCells)) {
                detail::ArrayTransfer<Exec>::attach(view_.node_cells_offsets, buffers_.node_cells_offsets, source.node_cells_offsets,
                    "mesh/aux/node_cells_offsets", exec);
                detail::ArrayTransfer<Exec>::attach(view_.node_cells, buffers_.node_cells, source.node_cells,
                    "mesh/aux/node_cells", exec);
            }
            if (has(requested, AuxConnType::NodeFaces)) {
                detail::ArrayTransfer<Exec>::attach(view_.node_faces_offsets, buffers_.node_faces_offsets, source.node_faces_offsets,
                    "mesh/aux/node_faces_offsets", exec);
                detail::ArrayTransfer<Exec>::attach(view_.node_faces, buffers_.node_faces, source.node_faces,
                    "mesh/aux/node_faces", exec);
            }
            if (has(requested, AuxConnType::CellCellsByNode)) {
                detail::ArrayTransfer<Exec>::attach(view_.cell_cells_node_offsets, buffers_.cell_cells_node_offsets, source.cell_cells_node_offsets,
                    "mesh/aux/cell_cells_node_offsets", exec);
                detail::ArrayTransfer<Exec>::attach(view_.cell_cells_node, buffers_.cell_cells_node, source.cell_cells_node,
                    "mesh/aux/cell_cells_node", exec);
            }
            detail::ArrayTransfer<Exec>::finish(exec);
        } catch (...) {
            // Drain queued copies before owned buffers are released during unwinding.
            detail::ArrayTransfer<Exec>::finish(exec);
            throw;
        }
    }

    // Temporary host caches would leave CPU views dangling.
    AuxConnectivityStorage(const MeshPart&, MeshAuxConnectivity&&, Exec = {}) = delete;
    AuxConnectivityStorage(const MeshPart&, const MeshAuxConnectivity&&, Exec = {}) = delete;
    AuxConnectivityStorage(const MeshPart&, MeshAuxConnectivity&&, AuxConnType, Exec = {}) = delete;
    AuxConnectivityStorage(const MeshPart&, const MeshAuxConnectivity&&, AuxConnType, Exec = {}) = delete;
    AuxConnectivityStorage(const AuxConnectivityStorage&) = delete;
    AuxConnectivityStorage& operator=(const AuxConnectivityStorage&) = delete;
    AuxConnectivityStorage(AuxConnectivityStorage&&) = delete;
    AuxConnectivityStorage& operator=(AuxConnectivityStorage&&) = delete;

    [[nodiscard]] view_type view() const & noexcept { return view_; }
    view_type view() const && = delete;

private:
    // These managed descriptors remain empty on HostSpace.
    detail::AuxConnectivityArrays<memory_space, true> buffers_;
    view_type view_;

    [[nodiscard]] static bool has(AuxConnType mask, AuxConnType flags) noexcept
    {
        const auto bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(mask) & bits) == bits;
    }

    static void validate(const MeshPart& mesh, const MeshAuxConnectivity& source, AuxConnType requested)
    {
        detail::require_kokkos();
        mpi::require((static_cast<std::uint64_t>(requested) & ~static_cast<std::uint64_t>(AuxConnType::All)) == 0,
                        "AuxConnectivityStorage: unknown flags");
        mpi::require(has(source.active_mask, requested), "AuxConnectivityStorage: requested data is not ready");
        if (requested == AuxConnType::None) return;
        mpi::require(mesh.n_own >= 0 && mesh.n_cells >= mesh.n_own && mesh.n_nodes_own >= 0 &&
                        mesh.n_nodes >= mesh.n_nodes_own && mesh.n_faces >= 0, "AuxConnectivityStorage: invalid counts");
        mpi::require(source.n_faces == mesh.n_faces && source.n_cells_own == mesh.n_own &&
                        source.n_nodes_own == mesh.n_nodes_own, "AuxConnectivityStorage: counts mismatch");
        if (has(requested, AuxConnType::CellFaces))
            mpi::require_csr(source.cell_faces_offsets, source.cell_faces,
                                static_cast<std::size_t>(source.n_cells_own), "cell_faces_offsets");
        if (has(requested, AuxConnType::CellCellsByFace))
            mpi::require_csr(source.cell_cells_face_offsets, source.cell_cells_face,
                                static_cast<std::size_t>(source.n_cells_own), "cell_cells_face_offsets");
        if (has(requested, AuxConnType::NodeCells))
            mpi::require_csr(source.node_cells_offsets, source.node_cells,
                                static_cast<std::size_t>(source.n_nodes_own), "node_cells_offsets");
        if (has(requested, AuxConnType::NodeFaces))
            mpi::require_csr(source.node_faces_offsets, source.node_faces,
                                static_cast<std::size_t>(source.n_nodes_own), "node_faces_offsets");
        if (has(requested, AuxConnType::CellCellsByNode))
            mpi::require_csr(source.cell_cells_node_offsets, source.cell_cells_node,
                                static_cast<std::size_t>(source.n_cells_own), "cell_cells_node_offsets");
    }
};

/// @brief Borrows selected ready host caches on CPU; owns their copies on GPU.
/// No table/metric construction and no MPI operations are performed here.
template<class Exec = execution::ExecutionSpace>
class AuxGeometryStorage {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using view_type = AuxGeometryView<memory_space>;
    static constexpr bool borrows_host = detail::ArrayTransfer<Exec>::borrows_host;

    /// @brief Attaches all groups marked ready in the source cache.
    explicit AuxGeometryStorage(const MeshPart& mesh, const MeshAuxGeometry& source, Exec exec = {})
        : AuxGeometryStorage(mesh, source, source.active_mask, exec) {}

    /// @brief Attaches a requested subset; missing groups are rejected.
    AuxGeometryStorage(const MeshPart& mesh, const MeshAuxGeometry& source, AuxGeomType requested, Exec exec = {})
    {
        validate(mesh, source, requested);
        view_.active_mask = requested;
        view_.n_faces = mesh.n_faces;
        view_.n_cells = mesh.n_cells;
        try {
            if (has(requested, AuxGeomType::FaceCellDistance)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_cell_dist, buffers_.face_cell_dist, source.face_cell_dist,
                    "mesh/aux/face_cell_dist", exec);
            }
            if (has(requested, AuxGeomType::FaceCellDistanceVector)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_cell_dist_x, buffers_.face_cell_dist_x, source.face_cell_dist_x,
                    "mesh/aux/face_cell_dist_x", exec);
                detail::ArrayTransfer<Exec>::attach(view_.face_cell_dist_y, buffers_.face_cell_dist_y, source.face_cell_dist_y,
                    "mesh/aux/face_cell_dist_y", exec);
                detail::ArrayTransfer<Exec>::attach(view_.face_cell_dist_z, buffers_.face_cell_dist_z, source.face_cell_dist_z,
                    "mesh/aux/face_cell_dist_z", exec);
            }
            if (has(requested, AuxGeomType::FaceCellDistanceInv)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_cell_dist_inv, buffers_.face_cell_dist_inv, source.face_cell_dist_inv,
                    "mesh/aux/face_cell_dist_inv", exec);
            }
            if (has(requested, AuxGeomType::FaceInterpolationWeights)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_interp_weight, buffers_.face_interp_weight, source.face_interp_weight,
                    "mesh/aux/face_interp_weight", exec);
            }
            if (has(requested, AuxGeomType::FaceNonOrthogonalityVector)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_non_ortho_x, buffers_.face_non_ortho_x, source.face_non_ortho_x,
                    "mesh/aux/face_non_ortho_x", exec);
                detail::ArrayTransfer<Exec>::attach(view_.face_non_ortho_y, buffers_.face_non_ortho_y, source.face_non_ortho_y,
                    "mesh/aux/face_non_ortho_y", exec);
                detail::ArrayTransfer<Exec>::attach(view_.face_non_ortho_z, buffers_.face_non_ortho_z, source.face_non_ortho_z,
                    "mesh/aux/face_non_ortho_z", exec);
            }
            if (has(requested, AuxGeomType::FaceNormalDistance)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_normal_dist, buffers_.face_normal_dist, source.face_normal_dist,
                    "mesh/aux/face_normal_dist", exec);
            }
            if (has(requested, AuxGeomType::FaceNormalDistanceInv)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_normal_dist_inv, buffers_.face_normal_dist_inv, source.face_normal_dist_inv,
                    "mesh/aux/face_normal_dist_inv", exec);
            }
            if (has(requested, AuxGeomType::FaceAreaOverNormalDistance)) {
                detail::ArrayTransfer<Exec>::attach(view_.face_area_over_normal_dist, buffers_.face_area_over_normal_dist, source.face_area_over_normal_dist,
                    "mesh/aux/face_area_over_normal_dist", exec);
            }
            if (has(requested, AuxGeomType::WallDistance)) {
                detail::ArrayTransfer<Exec>::attach(view_.cell_wall_dist, buffers_.cell_wall_dist, source.cell_wall_dist,
                    "mesh/aux/cell_wall_dist", exec);
            }
            detail::ArrayTransfer<Exec>::finish(exec);
        } catch (...) {
            // Drain queued copies before owned buffers are released during unwinding.
            detail::ArrayTransfer<Exec>::finish(exec);
            throw;
        }
    }

    // Temporary host caches would leave CPU views dangling.
    AuxGeometryStorage(const MeshPart&, MeshAuxGeometry&&, Exec = {}) = delete;
    AuxGeometryStorage(const MeshPart&, const MeshAuxGeometry&&, Exec = {}) = delete;
    AuxGeometryStorage(const MeshPart&, MeshAuxGeometry&&, AuxGeomType, Exec = {}) = delete;
    AuxGeometryStorage(const MeshPart&, const MeshAuxGeometry&&, AuxGeomType, Exec = {}) = delete;
    AuxGeometryStorage(const AuxGeometryStorage&) = delete;
    AuxGeometryStorage& operator=(const AuxGeometryStorage&) = delete;
    AuxGeometryStorage(AuxGeometryStorage&&) = delete;
    AuxGeometryStorage& operator=(AuxGeometryStorage&&) = delete;
    
    [[nodiscard]] view_type view() const & noexcept { return view_; }
    view_type view() const && = delete;


private:
    // These managed descriptors remain empty on HostSpace.
    detail::AuxGeometryArrays<memory_space, true> buffers_;
    view_type view_;

    [[nodiscard]] static bool has(AuxGeomType mask, AuxGeomType flags) noexcept
    {
        const auto bits = static_cast<std::uint64_t>(flags);
        return (static_cast<std::uint64_t>(mask) & bits) == bits;
    }
 
    static void validate(const MeshPart& mesh, const MeshAuxGeometry& source, AuxGeomType requested)
    {
        detail::require_kokkos();
        mpi::require((static_cast<std::uint64_t>(requested) & ~static_cast<std::uint64_t>(AuxGeomType::All)) == 0,
                        "AuxGeometryStorage: unknown flags");
        mpi::require(has(source.active_mask, requested), "AuxGeometryStorage: requested data is not ready");
        if (requested == AuxGeomType::None) return;
        mpi::require(mesh.n_own >= 0 && mesh.n_cells >= mesh.n_own && mesh.n_nodes_own >= 0 &&
                        mesh.n_nodes >= mesh.n_nodes_own && mesh.n_faces >= 0, "AuxGeometryStorage: invalid counts");
        mpi::require(source.n_faces == mesh.n_faces, "AuxGeometryStorage: face count mismatch");
        const auto faces = static_cast<std::size_t>(mesh.n_faces);
        const auto cells = static_cast<std::size_t>(mesh.n_cells);
        if (has(requested, AuxGeomType::FaceCellDistance)) {
            mpi::require_size(source.face_cell_dist, faces, "face_cell_dist");
        }
        if (has(requested, AuxGeomType::FaceCellDistanceVector)) {
            mpi::require_size(source.face_cell_dist_x, faces, "face_cell_dist_x");
            mpi::require_size(source.face_cell_dist_y, faces, "face_cell_dist_y");
            mpi::require_size(source.face_cell_dist_z, faces, "face_cell_dist_z");
        }
        if (has(requested, AuxGeomType::FaceCellDistanceInv)) {
            mpi::require_size(source.face_cell_dist_inv, faces, "face_cell_dist_inv");
        }
        if (has(requested, AuxGeomType::FaceInterpolationWeights)) {
            mpi::require_size(source.face_interp_weight, faces, "face_interp_weight");
        }
        if (has(requested, AuxGeomType::FaceNonOrthogonalityVector)) {
            mpi::require_size(source.face_non_ortho_x, faces, "face_non_ortho_x");
            mpi::require_size(source.face_non_ortho_y, faces, "face_non_ortho_y");
            mpi::require_size(source.face_non_ortho_z, faces, "face_non_ortho_z");
        }
        if (has(requested, AuxGeomType::FaceNormalDistance)) {
            mpi::require_size(source.face_normal_dist, faces, "face_normal_dist");
        }
        if (has(requested, AuxGeomType::FaceNormalDistanceInv)) {
            mpi::require_size(source.face_normal_dist_inv, faces, "face_normal_dist_inv");
        }
        if (has(requested, AuxGeomType::FaceAreaOverNormalDistance)) {
            mpi::require_size(source.face_area_over_normal_dist, faces, "face_area_over_normal_dist");
        }
        if (has(requested, AuxGeomType::WallDistance)) {
            mpi::require_size(source.cell_wall_dist, cells, "cell_wall_dist");
        }
    }
};

} // namespace krona::mesh::compute
