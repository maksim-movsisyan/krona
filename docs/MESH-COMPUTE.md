# KRONA compute mesh

`krona::mesh::compute` adapts a validated, static `MeshPart` for Kokkos kernels. It selects existing arrays and exposes read-only views; it does not construct topology, compute geometry, partition the mesh, or exchange halo values.

The host mesh and file-format contracts are described in [MESH-PARTITION.md](MESH-PARTITION.md). Solution storage is described in [FIELDS.md](FIELDS.md).

## 1. Objects and ownership

| Object | Responsibility |
| --- | --- |
| `mesh::MeshPart` | Host data used by preprocessing and I/O |
| `compute::MeshStorage<Exec>` | Attaches selected host arrays or owns their execution-space copies |
| `compute::MeshView<Space>` | Read-only, unmanaged descriptors passed by value to kernels |
| `compute::HaloPlanView` | Host-only peer ranks and communication offsets |

Kokkos must be active before storage construction. `Exec` defaults to `execution::ExecutionSpace`, selected by the generated KRONA configuration. Mesh indices use `LocalIndex`, CSR offsets use `std::size_t`, and coordinates/metrics use `GeometryReal`.

| Execution memory | Selected mesh arrays | Halo peer/offset tables |
| --- | --- | --- |
| `HostSpace` — Serial/OpenMP | Borrowed directly from `MeshPart`; no second payload allocation | Borrowed host spans |
| CUDA/HIP device memory | Owned copies; only selected arrays are uploaded | Small owned host copies |

`borrows_host` is a compile-time test for exact `HostSpace`. The constructor completes uploads with `exec.fence()` before returning; the host borrowing path performs no fence. GPU storage can then outlive its source mesh, but CPU storage requires the source arrays to remain alive and stable. Retain `MeshPart` when other modules still need its metadata.

Views and halo-plan spans do not extend storage lifetime. Complete all kernels, copies, and MPI requests before destroying their backing data. Borrowed arrays must not be resized, reallocated, or modified while readers use them. Storage destructors perform no implicit fence; storage objects are neither copyable nor movable.

## 2. Data selection

`MeshSizes` is always present. `MeshOptions::data` selects independent groups:

| `MeshData` flag | Available data |
| --- | --- |
| `FaceCells` | `connectivity.face_owner`, `face_neigh` |
| `CellNodes` | Cell types and cell-to-node CSR |
| `FaceNodes` | Face types, owner-local facet indices, and face-to-node CSR |
| `NodeCoordinates` | `geometry.node_x/y/z` |
| `CellGeometry` | Cell centroids and volumes, including ghosts |
| `FaceGeometry` | Face centroids, unit normals, and areas |
| `Halo` | Kernel gather/scatter indices and `storage.halo_plan()` |

The default `FiniteVolume` preset is `FaceCells | CellGeometry | FaceGeometry | Halo`. `Connectivity` combines the three connectivity groups; `Geometry` combines cell/face metrics and excludes node coordinates. `All` selects all seven groups; `None` selects only sizes.

Use `view.has(flag)` to establish availability. An available group can be empty on an empty rank. Unselected descriptors stay empty, and masks never build missing data.

```cpp
#include <krona/mesh/compute/mesh_storage.hpp>

// Kokkos is active; host_mesh is a validated MeshPart.
krona::execution::ExecutionSpace exec{};
using namespace krona::mesh::compute;
MeshStorage<> storage(host_mesh, {
    .data = MeshData::CellNodes | MeshData::NodeCoordinates
}, exec);
auto mesh = storage.view();
// Pass mesh or its required arrays by value to Kokkos kernels.
```

## 3. Local numbering and geometry

All indices are zero-based. Cells `[0, n_own)` are owned; `[n_own, n_cells)` are remote face-neighbour ghosts. Nodes `[0, n_nodes_own)` are used by owned cells; this prefix does not imply exclusive global node ownership.

Every face touches an owned cell. `face_owner` is owned; `face_neigh` is owned, ghost, or `-1` for an exterior boundary. Normals are unit vectors pointing outward from the owner, so the flux area vector is `face_area * face_normal`. Cut faces have a copy on each adjacent rank with that rank's outward orientation; there are no ghost-ghost faces.

The following intervals require `sizes.faces_grouped == true`:

| Face range | Meaning |
| --- | --- |
| `[0, n_internal_faces)` | Owned-owned |
| `[n_internal_faces, n_inner_faces)` | Owned-ghost |
| `[n_inner_faces, n_faces)` | Exterior boundary |

Cell/face node connectivity uses local node indices and CSR offsets measured in entries. Storage performs shape checks, not full topology or geometry validation; callers must supply a valid mesh.

Patch labels, patch-face lists, and global IDs remain outside `MeshView`. A boundary-condition component can prepare compact patch ranges/lists from `MeshPart`; the three face regions alone do not identify individual patches.

## 4. Halo and optional caches

With `Halo` selected, `view.halo` holds `send_owned_local` and `recv_ghost_local` in execution memory. The host-only `halo_plan()` supplies `nb_ranks`, `send_offsets`, and `recv_offsets`. Offsets count cell records, not components or bytes. Preserve the prepared per-peer index order, and use a communicator matching the mesh rank numbering. Do not capture the host plan in GPU kernels.

These are exchange maps only. Packing, MPI requests, staging, and unpacking belong to a separate exchange component; CUDA-aware MPI is not required by this module.

`AuxConnectivityStorage` and `AuxGeometryStorage` independently adapt ready host caches using the same borrowing/upload rules. Their views provide optional CSR adjacency and additional metrics; `IndexRow` accesses CSR rows inside kernels. Requested groups must be ready and match the mesh snapshot/numbering. A one-layer face halo does not guarantee complete vertex-neighbour or higher-order stencils.

## 5. Scope and updates

Contract failures use `mpi::fatal`; successful construction performs no MPI operations. Handle allocation/library exceptions at the application boundary and drain submitted work before releasing referenced buffers.

The current interface represents a static snapshot. Host edits do not refresh GPU copies. Reordering or topology changes require consistent field remapping and rebuilt dependent caches/storage. Moving meshes, nonconforming AMR, FEM DOF ownership, and coordinate exchange need explicit additional components.
