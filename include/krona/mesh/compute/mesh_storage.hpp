#pragma once

#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/mesh/localmesh.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace krona::mesh::compute {

struct MeshOptions {
    // Vertex connectivity is unnecessary for a static cell-centred FVM step.
    bool with_connectivity = false;
};

/// @brief Borrows MeshPart arrays on CPU; owns their selected copies on GPU.
/// MeshPart must already be validated, grouped and have complete geometry.
/// Finish all users before destroying storage or reallocating borrowed arrays.
template<class Exec = execution::ExecutionSpace>
class MeshStorage {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using view_type = MeshView<memory_space>;
    static constexpr bool borrows_host = std::is_same_v<memory_space, Kokkos::HostSpace>;

    /// @brief Creates a ready-to-use view; all requested uploads finish before return.
    explicit MeshStorage(const MeshPart& mesh, MeshOptions options = {}, Exec exec = {})
    {
        validate(mesh, options);
        auto& t = view_.topology;
        t.n_own = mesh.n_own;
        t.n_cells = mesh.n_cells;
        t.n_nodes_own = mesh.n_nodes_own;
        t.n_nodes = mesh.n_nodes;
        t.n_faces = mesh.n_faces;
        t.n_internal_faces = mesh.n_internal_faces;
        t.n_inner_faces = mesh.n_inner_faces;

        try {
            attach(view_.topology.face_owner, buffers_.topology.face_owner, mesh.face_owner, "mesh/face_owner", exec);
            attach(view_.topology.face_neigh, buffers_.topology.face_neigh, mesh.face_neigh, "mesh/face_neigh", exec);
            attach(view_.topology.face_patch, buffers_.topology.face_patch, mesh.face_patch, "mesh/face_patch", exec);
            attach(view_.topology.patch_face_offsets, buffers_.topology.patch_face_offsets, mesh.patch_face_offsets, "mesh/patch_face_offsets", exec);
            attach(view_.topology.patch_faces, buffers_.topology.patch_faces, mesh.patch_faces, "mesh/patch_faces", exec);

            attach(view_.geometry.cell_centroid_x, buffers_.geometry.cell_centroid_x, mesh.cell_centroid_x, "mesh/cell_centroid_x", exec);
            attach(view_.geometry.cell_centroid_y, buffers_.geometry.cell_centroid_y, mesh.cell_centroid_y, "mesh/cell_centroid_y", exec);
            attach(view_.geometry.cell_centroid_z, buffers_.geometry.cell_centroid_z, mesh.cell_centroid_z, "mesh/cell_centroid_z", exec);
            attach(view_.geometry.cell_volume, buffers_.geometry.cell_volume, mesh.cell_volume, "mesh/cell_volume", exec);
            attach(view_.geometry.face_centroid_x, buffers_.geometry.face_centroid_x, mesh.face_centroid_x, "mesh/face_centroid_x", exec);
            attach(view_.geometry.face_centroid_y, buffers_.geometry.face_centroid_y, mesh.face_centroid_y, "mesh/face_centroid_y", exec);
            attach(view_.geometry.face_centroid_z, buffers_.geometry.face_centroid_z, mesh.face_centroid_z, "mesh/face_centroid_z", exec);
            attach(view_.geometry.face_normal_x, buffers_.geometry.face_normal_x, mesh.face_normal_x, "mesh/face_normal_x", exec);
            attach(view_.geometry.face_normal_y, buffers_.geometry.face_normal_y, mesh.face_normal_y, "mesh/face_normal_y", exec);
            attach(view_.geometry.face_normal_z, buffers_.geometry.face_normal_z, mesh.face_normal_z, "mesh/face_normal_z", exec);
            attach(view_.geometry.face_area, buffers_.geometry.face_area, mesh.face_area, "mesh/face_area", exec);

            attach(view_.halo.send_owned_local, buffers_.halo.send_owned_local, mesh.send_owned_local, "mesh/send_owned_local", exec);
            attach(view_.halo.recv_ghost_local, buffers_.halo.recv_ghost_local, mesh.recv_ghost_local, "mesh/recv_ghost_local", exec);

            if (options.with_connectivity) {
                attach(view_.connectivity.cell_type, buffers_.connectivity.cell_type, mesh.cell_type, "mesh/cell_type", exec);
                attach(view_.connectivity.cell_nodes_offsets, buffers_.connectivity.cell_nodes_offsets, mesh.cell_nodes_offsets, "mesh/cell_nodes_offsets", exec);
                attach(view_.connectivity.cell_nodes, buffers_.connectivity.cell_nodes, mesh.cell_nodes, "mesh/cell_nodes", exec);
                attach(view_.connectivity.node_x, buffers_.connectivity.node_x, mesh.node_x, "mesh/node_x", exec);
                attach(view_.connectivity.node_y, buffers_.connectivity.node_y, mesh.node_y, "mesh/node_y", exec);
                attach(view_.connectivity.node_z, buffers_.connectivity.node_z, mesh.node_z, "mesh/node_z", exec);
                attach(view_.connectivity.face_type, buffers_.connectivity.face_type, mesh.face_type, "mesh/face_type", exec);
                attach(view_.connectivity.face_local_index, buffers_.connectivity.face_local_index, mesh.face_local_index, "mesh/face_local_index", exec);
                attach(view_.connectivity.face_nodes_offsets, buffers_.connectivity.face_nodes_offsets, mesh.face_nodes_offsets, "mesh/face_nodes_offsets", exec);
                attach(view_.connectivity.face_nodes, buffers_.connectivity.face_nodes, mesh.face_nodes, "mesh/face_nodes", exec);
            }

            if constexpr (!borrows_host)
                exec.fence("KRONA mesh upload");
        } catch (...) {
            // Keep queued copies from outliving their buffers during unwinding.
            if constexpr (!borrows_host)
                exec.fence("KRONA failed mesh upload");
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

private:
    // Managed views stay empty on CPU: there is no second host mesh allocation.
    detail::MeshArrays<memory_space, true> buffers_;
    view_type view_;

    template<class T>
    static void attach(detail::MeshArray<T, memory_space, false>& view,
                       detail::MeshArray<T, memory_space, true>& buffer,
                       const std::vector<T>& source, const char* label, const Exec& exec)
    {
        if constexpr (borrows_host) {
            view = detail::MeshArray<T, memory_space, false>(source.data(), source.size());
        } else {
            if (source.empty()) return;
            // Initialization would be wasted: deep_copy overwrites every entry.
            buffer = detail::MeshArray<T, memory_space, true>(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), source.size());
            const detail::MeshArray<T, Kokkos::HostSpace, false> host(source.data(), source.size());
            Kokkos::deep_copy(exec, buffer, host);
            view = buffer;
        }
    }

    static void require(bool condition, const char* message)
    {
        if (!condition) throw std::invalid_argument(message);
    }

    template<class T>
    static void require_size(const std::vector<T>& a, std::size_t expected, const char* name)
    {
        if (a.size() != expected)
            throw std::invalid_argument(std::string("MeshStorage: invalid size of ") + name);
    }

    // Cheap shape checks; full index/geometry validation belongs to the mesh module.
    static void validate(const MeshPart& m, MeshOptions options)
    {
        require(Kokkos::is_initialized() && !Kokkos::is_finalized(), "MeshStorage requires active Kokkos");
        require(m.faces_grouped, "MeshStorage requires grouped faces");
        require(m.n_own >= 0 && m.n_cells >= m.n_own && m.n_nodes_own >= 0 &&
                m.n_nodes >= m.n_nodes_own,
                "MeshStorage: invalid cell/node counts");
        require(m.n_internal_faces >= 0 && m.n_inner_faces >= m.n_internal_faces &&
                m.n_faces >= m.n_inner_faces, "MeshStorage: invalid face ranges");
        const auto cells = static_cast<std::size_t>(m.n_cells);
        const auto faces = static_cast<std::size_t>(m.n_faces);
        const auto nodes = static_cast<std::size_t>(m.n_nodes);
        require_size(m.face_owner, faces, "face_owner");
        require_size(m.face_neigh, faces, "face_neigh");
        require_size(m.face_patch, faces, "face_patch");
        require_size(m.cell_centroid_x, cells, "cell_centroid_x");
        require_size(m.cell_centroid_y, cells, "cell_centroid_y");
        require_size(m.cell_centroid_z, cells, "cell_centroid_z");
        require_size(m.cell_volume, cells, "cell_volume");
        require_size(m.face_centroid_x, faces, "face_centroid_x");
        require_size(m.face_centroid_y, faces, "face_centroid_y");
        require_size(m.face_centroid_z, faces, "face_centroid_z");
        require_size(m.face_normal_x, faces, "face_normal_x");
        require_size(m.face_normal_y, faces, "face_normal_y");
        require_size(m.face_normal_z, faces, "face_normal_z");
        require_size(m.face_area, faces, "face_area");
        require_size(m.patch_face_offsets, m.patches.size() + 1, "patch_face_offsets");
        require(m.patch_face_offsets.front() == 0 && m.patch_face_offsets.back() == m.patch_faces.size(),
                "MeshStorage: invalid patch CSR bounds");
        require_size(m.send_offsets, m.nb_ranks.size() + 1, "send_offsets");
        require_size(m.recv_offsets, m.nb_ranks.size() + 1, "recv_offsets");
        require(m.send_offsets.front() == 0 && m.send_offsets.back() == m.send_owned_local.size() &&
                m.recv_offsets.front() == 0 && m.recv_offsets.back() == m.recv_ghost_local.size(),
                "MeshStorage: invalid halo CSR bounds");
        if (options.with_connectivity) {
            require_size(m.cell_type, cells, "cell_type");
            require_size(m.cell_nodes_offsets, cells + 1, "cell_nodes_offsets");
            require_size(m.node_x, nodes, "node_x");
            require_size(m.node_y, nodes, "node_y");
            require_size(m.node_z, nodes, "node_z");
            require_size(m.face_type, faces, "face_type");
            require_size(m.face_local_index, faces, "face_local_index");
            require_size(m.face_nodes_offsets, faces + 1, "face_nodes_offsets");
            require(m.cell_nodes_offsets.front() == 0 && m.cell_nodes_offsets.back() == m.cell_nodes.size() &&
                    m.face_nodes_offsets.front() == 0 && m.face_nodes_offsets.back() == m.face_nodes.size(),
                    "MeshStorage: invalid connectivity CSR bounds");
        }
    }
};

} // namespace krona::mesh::compute
