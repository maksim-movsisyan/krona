#pragma once

#include <krona/mesh/compute/detail/array_transfer.hpp>
#include <krona/mesh/compute/halo_plan.hpp>
#include <krona/mesh/localmesh.hpp>

#include <vector>

namespace krona::mesh::compute {

/// @brief Default FVM preset; select a different mask for other methods.
struct MeshOptions {
    MeshData data = MeshData::FiniteVolume;
};

/// @brief Borrows MeshPart arrays on CPU; owns their selected copies on GPU.
/// MeshPart must be validated and contain the selected groups.
/// Finish all users before destroying storage or reallocating borrowed arrays.
template<class Exec = execution::ExecutionSpace>
class MeshStorage {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using view_type = MeshView<memory_space>;
    static constexpr bool borrows_host = detail::ArrayTransfer<Exec>::borrows_host;

    /// @brief Creates a ready-to-use view; all requested uploads finish before return.
    explicit MeshStorage(const MeshPart& mesh, MeshOptions options = {}, Exec exec = {})
    {
        const auto selected = options.data;
        validate(mesh, selected);
        view_.active_mask = selected;
        view_.sizes = {mesh.n_own, mesh.n_cells, mesh.n_nodes_own, mesh.n_nodes,
                       mesh.n_faces, mesh.n_internal_faces, mesh.n_inner_faces, mesh.faces_grouped};

        try {
            if (has(selected, MeshData::FaceCells)) {
                attach(view_.connectivity.face_owner, buffers_.connectivity.face_owner, mesh.face_owner, "mesh/face_owner", exec);
                attach(view_.connectivity.face_neigh, buffers_.connectivity.face_neigh, mesh.face_neigh, "mesh/face_neigh", exec);
            }
            if (has(selected, MeshData::CellGeometry)) {
                attach(view_.geometry.cell_centroid_x, buffers_.geometry.cell_centroid_x, mesh.cell_centroid_x, "mesh/cell_centroid_x", exec);
                attach(view_.geometry.cell_centroid_y, buffers_.geometry.cell_centroid_y, mesh.cell_centroid_y, "mesh/cell_centroid_y", exec);
                attach(view_.geometry.cell_centroid_z, buffers_.geometry.cell_centroid_z, mesh.cell_centroid_z, "mesh/cell_centroid_z", exec);
                attach(view_.geometry.cell_volume, buffers_.geometry.cell_volume, mesh.cell_volume, "mesh/cell_volume", exec);
            }
            if (has(selected, MeshData::FaceGeometry)) {
                attach(view_.geometry.face_centroid_x, buffers_.geometry.face_centroid_x, mesh.face_centroid_x, "mesh/face_centroid_x", exec);
                attach(view_.geometry.face_centroid_y, buffers_.geometry.face_centroid_y, mesh.face_centroid_y, "mesh/face_centroid_y", exec);
                attach(view_.geometry.face_centroid_z, buffers_.geometry.face_centroid_z, mesh.face_centroid_z, "mesh/face_centroid_z", exec);
                attach(view_.geometry.face_normal_x, buffers_.geometry.face_normal_x, mesh.face_normal_x, "mesh/face_normal_x", exec);
                attach(view_.geometry.face_normal_y, buffers_.geometry.face_normal_y, mesh.face_normal_y, "mesh/face_normal_y", exec);
                attach(view_.geometry.face_normal_z, buffers_.geometry.face_normal_z, mesh.face_normal_z, "mesh/face_normal_z", exec);
                attach(view_.geometry.face_area, buffers_.geometry.face_area, mesh.face_area, "mesh/face_area", exec);
            }
            if (has(selected, MeshData::Halo)) {
                attach_halo_plan(mesh);
                attach(view_.halo.send_owned_local, buffers_.halo.send_owned_local, mesh.send_owned_local, "mesh/send_owned_local", exec);
                attach(view_.halo.recv_ghost_local, buffers_.halo.recv_ghost_local, mesh.recv_ghost_local, "mesh/recv_ghost_local", exec);
            }
            if (has(selected, MeshData::CellNodes)) {
                attach(view_.connectivity.cell_type, buffers_.connectivity.cell_type, mesh.cell_type, "mesh/cell_type", exec);
                attach(view_.connectivity.cell_nodes_offsets, buffers_.connectivity.cell_nodes_offsets, mesh.cell_nodes_offsets, "mesh/cell_nodes_offsets", exec);
                attach(view_.connectivity.cell_nodes, buffers_.connectivity.cell_nodes, mesh.cell_nodes, "mesh/cell_nodes", exec);
            }
            if (has(selected, MeshData::NodeCoordinates)) {
                attach(view_.geometry.node_x, buffers_.geometry.node_x, mesh.node_x, "mesh/node_x", exec);
                attach(view_.geometry.node_y, buffers_.geometry.node_y, mesh.node_y, "mesh/node_y", exec);
                attach(view_.geometry.node_z, buffers_.geometry.node_z, mesh.node_z, "mesh/node_z", exec);
            }
            if (has(selected, MeshData::FaceNodes)) {
                attach(view_.connectivity.face_type, buffers_.connectivity.face_type, mesh.face_type, "mesh/face_type", exec);
                attach(view_.connectivity.face_local_index, buffers_.connectivity.face_local_index, mesh.face_local_index, "mesh/face_local_index", exec);
                attach(view_.connectivity.face_nodes_offsets, buffers_.connectivity.face_nodes_offsets, mesh.face_nodes_offsets, "mesh/face_nodes_offsets", exec);
                attach(view_.connectivity.face_nodes, buffers_.connectivity.face_nodes, mesh.face_nodes, "mesh/face_nodes", exec);
            }

            detail::ArrayTransfer<Exec>::finish(exec);
        } catch (...) {
            // Keep queued copies from outliving their buffers during unwinding.
            detail::ArrayTransfer<Exec>::finish(exec);
            throw;
        }
    }

    // A temporary MeshPart would leave CPU views dangling immediately.
    MeshStorage(MeshPart&&, MeshOptions = {}, Exec = {}) = delete;
    MeshStorage(const MeshPart&&, MeshOptions = {}, Exec = {}) = delete;
    MeshStorage(const MeshStorage&) = delete;
    MeshStorage& operator=(const MeshStorage&) = delete;
    MeshStorage(MeshStorage&&) = delete;
    MeshStorage& operator=(MeshStorage&&) = delete;

    /// @brief Copies descriptors only; their backing arrays must remain alive.
    [[nodiscard]] view_type view() const noexcept { return view_; }

    /// @brief Host MPI metadata, available when Halo was selected; no device pointers.
    [[nodiscard]] HaloPlanView halo_plan() const noexcept { return halo_plan_; }

private:
    // Managed views stay empty on CPU: there is no second host mesh allocation.
    detail::MeshArrays<memory_space, true> buffers_;
    view_type view_;
    // Empty on CPU. On GPU the small host plan is owned independently of MeshPart.
    std::vector<int> halo_peers_;
    std::vector<std::size_t> halo_send_offsets_, halo_recv_offsets_;
    HaloPlanView halo_plan_;

    void attach_halo_plan(const MeshPart& mesh)
    {
        if constexpr (borrows_host) {
            halo_plan_ = {mesh.nb_ranks, mesh.send_offsets, mesh.recv_offsets};
        } else {
            halo_peers_ = mesh.nb_ranks;
            halo_send_offsets_ = mesh.send_offsets;
            halo_recv_offsets_ = mesh.recv_offsets;
            halo_plan_ = {halo_peers_, halo_send_offsets_, halo_recv_offsets_};
        }
    }

    template<class T>
    static void attach(detail::MeshArray<T, memory_space, false>& view,
                       detail::MeshArray<T, memory_space, true>& buffer,
                       const std::vector<T>& source, const char* label, const Exec& exec)
    {
        detail::ArrayTransfer<Exec>::attach(view, buffer, source, label, exec);
    }

    [[nodiscard]] static bool has(MeshData mask, MeshData flags) noexcept
    {
        const auto bits = static_cast<std::uint32_t>(flags);
        return (static_cast<std::uint32_t>(mask) & bits) == bits;
    }

    // Cheap shape checks; full index/geometry validation belongs to the mesh module.
    static void validate(const MeshPart& m, MeshData selected)
    {
        detail::require_kokkos();
        detail::require((static_cast<std::uint32_t>(selected) & ~static_cast<std::uint32_t>(MeshData::All)) == 0,
                        "MeshStorage: unknown data flags");
        detail::require(m.n_own >= 0 && m.n_cells >= m.n_own && m.n_nodes_own >= 0 &&
                        m.n_nodes >= m.n_nodes_own && m.n_faces >= 0,
                        "MeshStorage: invalid cell/node/face counts");
        if (m.faces_grouped)
            detail::require(m.n_internal_faces >= 0 && m.n_inner_faces >= m.n_internal_faces &&
                            m.n_faces >= m.n_inner_faces, "MeshStorage: invalid grouped face ranges");
        const std::size_t cells = static_cast<std::size_t>(m.n_cells);
        const std::size_t faces = static_cast<std::size_t>(m.n_faces);
        const std::size_t nodes = static_cast<std::size_t>(m.n_nodes);
        if (has(selected, MeshData::FaceCells)) {
            detail::require_size(m.face_owner, faces, "face_owner");
            detail::require_size(m.face_neigh, faces, "face_neigh");
        }
        if (has(selected, MeshData::CellGeometry)) {
            detail::require_size(m.cell_centroid_x, cells, "cell_centroid_x");
            detail::require_size(m.cell_centroid_y, cells, "cell_centroid_y");
            detail::require_size(m.cell_centroid_z, cells, "cell_centroid_z");
            detail::require_size(m.cell_volume, cells, "cell_volume");
        }
        if (has(selected, MeshData::FaceGeometry)) {
            detail::require_size(m.face_centroid_x, faces, "face_centroid_x");
            detail::require_size(m.face_centroid_y, faces, "face_centroid_y");
            detail::require_size(m.face_centroid_z, faces, "face_centroid_z");
            detail::require_size(m.face_normal_x, faces, "face_normal_x");
            detail::require_size(m.face_normal_y, faces, "face_normal_y");
            detail::require_size(m.face_normal_z, faces, "face_normal_z");
            detail::require_size(m.face_area, faces, "face_area");
        }
        if (has(selected, MeshData::Halo)) {
            detail::require_size(m.send_offsets, m.nb_ranks.size() + 1, "send_offsets");
            detail::require_size(m.recv_offsets, m.nb_ranks.size() + 1, "recv_offsets");
            detail::require(m.send_offsets.front() == 0 && m.send_offsets.back() == m.send_owned_local.size() &&
                    m.recv_offsets.front() == 0 && m.recv_offsets.back() == m.recv_ghost_local.size(),
                    "MeshStorage: invalid halo CSR bounds");
            detail::require(m.nprocs > 0 && m.rank >= 0 && m.rank < m.nprocs,
                            "MeshStorage: invalid halo rank metadata");
            int previous = -1;
            for (std::size_t peer = 0; peer < m.nb_ranks.size(); ++peer) {
                const int rank = m.nb_ranks[peer];
                detail::require(rank > previous && rank < m.nprocs && rank != m.rank,
                                "MeshStorage: halo peers must be sorted, unique and valid");
                detail::require(m.send_offsets[peer] <= m.send_offsets[peer + 1] &&
                                m.recv_offsets[peer] <= m.recv_offsets[peer + 1],
                                "MeshStorage: halo offsets must be nondecreasing");
                previous = rank;
            }
        }
        if (has(selected, MeshData::CellNodes)) {
            detail::require_size(m.cell_type, cells, "cell_type");
            detail::require_csr(m.cell_nodes_offsets, m.cell_nodes, cells, "cell_nodes_offsets");
        }
        if (has(selected, MeshData::NodeCoordinates)) {
            detail::require_size(m.node_x, nodes, "node_x");
            detail::require_size(m.node_y, nodes, "node_y");
            detail::require_size(m.node_z, nodes, "node_z");
        }
        if (has(selected, MeshData::FaceNodes)) {
            detail::require_size(m.face_type, faces, "face_type");
            detail::require_size(m.face_local_index, faces, "face_local_index");
            detail::require_csr(m.face_nodes_offsets, m.face_nodes, faces, "face_nodes_offsets");
        }
    }
};

} // namespace krona::mesh::compute