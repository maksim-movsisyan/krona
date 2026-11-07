# Device Mesh

`krona::mesh::device` exposes selected mesh arrays to Kokkos kernels. It works with CPU and GPU execution spaces. Host construction and auxiliary algorithms are described in [MESH-HOST](MESH-HOST.md).

The interface has two parts:

| Object | Purpose |
| --- | --- |
| `MeshStorage<Exec>` | Attaches selected arrays from `host::MeshPart` and exposes a base mesh view. |
| `AuxConnectivityStorage<Exec>` / `AuxGeometryStorage<Exec>` | Attach selected groups from already-built host auxiliary caches. No `MeshPart` argument is needed. |
| `MeshView<Space>` / `AuxConnectivityView<Space>` / `AuxGeometryView<Space>` | Read-only, non-owning descriptors copied into kernels. `Space` is a memory space; storage is parameterized by an execution space. |

Storage neither builds topology/metrics nor performs halo exchanges. Kokkos must be initialized before creating storage. The source mesh must already contain every requested group.

## Load, attach aux and use a kernel

This example loads a prepared HDF5 mesh, constructs host caches, attaches them and sums `A / h` over each owned cell's faces. The output is a separate writable Kokkos array.

```cpp
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/mesh/host/aux_geometry.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv};
    return krona::mpi::guarded([&] {
        Kokkos::ScopeGuard kokkos{argc, argv};
        namespace host = krona::mesh::host;
        namespace device = krona::mesh::device;
        using krona::mesh::AuxConnType;
        using krona::mesh::AuxGeomType;
        using Exec = krona::execution::ExecutionSpace;
        using Space = typename Exec::memory_space;
        using Policy = Kokkos::RangePolicy<Exec,
            Kokkos::IndexType<krona::LocalIndex>>;
        const auto comm = krona::mpi::CommView::world();
        Exec exec;

        auto host_mesh = krona::io::solver_mesh::import_mesh_hdf5("mesh.h5", comm);
        auto host_conn = host::build_aux_connectivity(
            host_mesh, AuxConnType::CellFaces);
        auto host_geom = host::build_aux_geometry(
            host_mesh, AuxGeomType::FaceAreaOverNormalDistance);

        device::MeshStorage<Exec> mesh_storage{
            host_mesh, {.data = device::MeshData::FiniteVolume}, exec};
        device::AuxConnectivityStorage<Exec> conn_storage{host_conn, exec};
        device::AuxGeometryStorage<Exec> geom_storage{host_geom, exec};

        const auto mesh = mesh_storage.view();
        const auto conn = conn_storage.view();
        const auto geom = geom_storage.view();
        const auto check = device::validation::validate<
            device::MeshData::FiniteVolume, AuxConnType::CellFaces,
            AuxGeomType::FaceAreaOverNormalDistance, true>(exec, mesh, conn, geom);
        krona::mpi::require(check.ok(), "Invalid mesh views");

        Kokkos::View<krona::GeometryReal*, Space> face_weight_sum{
            "face_weight_sum", static_cast<std::size_t>(mesh.sizes.n_own)};
        krona::execution::complete(exec, [&] {
            Kokkos::parallel_for("sum_face_weights",
                Policy(exec, 0, mesh.sizes.n_own),
                KOKKOS_LAMBDA(krona::LocalIndex cell) {
                    krona::GeometryReal sum = 0;
                    for (auto j = conn.cell_faces_offsets(cell);
                         j < conn.cell_faces_offsets(cell + 1); ++j) {
                        sum += geom.face_area_over_normal_dist(conn.cell_faces(j));
                    }
                    face_weight_sum(cell) = sum;
                });
        }); // Drain submitted work before releasing its backing arrays.
        return 0;
    });
}
```

Use local indices inside kernels. Array length is available through `.extent(0)`; empty local meshes are supported. A kernel can access only memory spaces accessible to its execution space.

## Select base mesh data

Header: `krona/mesh/device/mesh_storage.hpp`. Combine `MeshData` flags with `|` and pass them in `MeshOptions::data`. The default is `FiniteVolume`; sizes are always present.

| `MeshData` | Attached view arrays |
| --- | --- |
| `FaceCells` | `connectivity.face_owner`, `face_neigh`. |
| `CellNodes` | `connectivity.cell_type`, `cell_nodes_offsets`, `cell_nodes`. |
| `FaceNodes` | `connectivity.face_type`, `face_local_index`, `face_nodes_offsets`, `face_nodes`. |
| `NodeCoordinates` | `geometry.node_x`, `node_y`, `node_z`. |
| `CellGeometry` | `geometry.cell_centroid_x/y/z`, `cell_volume`. Covers owned and ghost cells. |
| `FaceGeometry` | `geometry.face_centroid_x/y/z`, `face_normal_x/y/z`, `face_area`. |
| `Halo` | `halo.send_owned_local`, `recv_ghost_local`, plus host metadata from `halo_plan()`. |
| `Connectivity` | `FaceCells \| CellNodes \| FaceNodes`. |
| `Geometry` | `CellGeometry \| FaceGeometry`. |
| `FiniteVolume` | `FaceCells \| Geometry \| Halo`. |
| `All` | Connectivity, node coordinates, geometry and halo. |
| `None` | Sizes only. |

**`FiniteVolume` does not include node connectivity or coordinates.** Select them explicitly for kernels needing cell/face node loops. Unselected arrays stay absent; there are no implicit group dependencies. For example, selecting `FaceNodes` does not also select `FaceCells`.

`mesh.has(flags)` checks whether all requested groups are present. Face intervals in `mesh.sizes` are meaningful only when `faces_grouped` is true; their meanings match `MeshPart`. Storage does not group faces, renumber entities or generate missing geometry. Patch strings, global IDs and donor ranks are not part of `MeshView`.

## Select auxiliary data

Header: `krona/mesh/device/aux_storage.hpp`.

The two constructor forms attach all ready groups or an explicit subset:

```cpp
device::AuxConnectivityStorage<Exec> all_ready{host_conn, exec};
device::AuxConnectivityStorage<Exec> selected{
    host_conn, AuxConnType::CellFaces, exec};
```

The same forms apply to `AuxGeometryStorage`, using `AuxGeomType`. The names `device`, `Exec`, `host_conn` and `exec` are those from the first example.

The requested mask must be a subset of the source cache's `active_mask`; requesting an unbuilt group aborts. Unknown flags and inconsistent selected array shapes are also rejected. Unselected ready groups are not attached. Auxiliary flags, array names, row ranges and geometric definitions are listed in [MESH-HOST](MESH-HOST.md).

Cache counts come from the cache itself. Storage cannot establish that a cache belongs to a particular mesh just from its constructor arguments; use caches from the same mesh snapshot and optionally validate the combined views. For wall distance, call the host cache's collective `add_wall_distance` **before** attaching `AuxGeomType::WallDistance`.

Auxiliary views are independent descriptors, outside `MeshView`. Pass only those a kernel needs; `has(flag)` and convenience predicates such as `has_cell_faces()` report attached groups.

## Ownership and lifetime

Storage chooses its behaviour from `Exec::memory_space`:

| Memory space | Behaviour |
| --- | --- |
| Exactly `Kokkos::HostSpace` | Borrows the source vectors through unmanaged read-only views. No second copy of the selected numeric arrays is allocated. |
| Any other memory space | Allocates selected buffers there and copies source data. Submitted copies finish before the constructor returns. |

`Storage::borrows_host` exposes this choice. On the copying path, completed storage owns its selected data independently of the host source. On the borrowing path, the source vectors must remain alive at stable addresses for every user of the view.

- Keep each storage object alive until all its kernels and halo users have completed. A copied view does not keep the buffers alive.
- Do not resize, clear, move from or reorder borrowed sources while views are in use. Host cache extension can allocate arrays: finish users and recreate storage when changing its source snapshot or selected groups.
- Rebuild storage after mesh changes. Device copies are snapshots and are not refreshed automatically; CPU aliases do not automatically refresh copied sizes, masks or other descriptors.
- Storage objects cannot be copied or moved. Constructors reject temporary host meshes/caches; `.view()` and `.halo_plan()` cannot be called on temporary storage.
- Use the same execution-space instance for dependent work, or synchronize producers before another instance accesses the data. Do not directly dereference GPU pointers on the host.

Kernel views expose const data. Store solver fields and temporary results in separate writable arrays, as in the example. Constructor failures pass through `mpi::guarded`; pending device copies are drained before aborting.

## Halo metadata

When `MeshData::Halo` is selected, `mesh_storage.halo_plan()` returns a `host::HaloPlanView` containing `nb_ranks`, `send_offsets` and `recv_offsets`. It stays on the CPU for MPI scheduling; do not capture it in GPU kernels.

The corresponding gather/scatter maps are `mesh.halo.send_owned_local` and `mesh.halo.recv_ghost_local` in the execution space's memory. Peer offsets count cells in those maps, not scalar entries or bytes. The CPU path borrows the host plan; the copying path owns a small host copy of the peer metadata.

Neither storage construction nor taking `halo_plan()` communicates field values. Use a halo exchange implementation with the matching maps, plan and communicator rank numbering when ghost fields need updating.

## The issue of asynchrony (!)

Storage constructors are synchronous with respect to data upload. The optional Exec exec = {} argument selects the execution-space instance used for allocations and copies. On the copying path, all selected transfers are completed by exec.fence() before the constructor returns. For Kokkos::HostSpace, storage borrows the host arrays and performs neither uploads nor a fence. Subsequent kernel launches may still be asynchronous; completing them before releasing their backing storage remains the caller’s responsibility.

## Validate selected views

Header: `krona/mesh/device/validation.hpp`.

`validation::validate<Data, Connectivity, Geometry, Compare>(exec, mesh, conn, geom)` returns local error counters; `result.ok()` means all counters are zero. Select exactly the groups you need to check. The default `Data` is `MeshData::All`, so a default call is unsuitable for a `FiniteVolume`-only view.

Shape/readiness errors prevent validation kernels from being launched. Further checks cover selected index ranges, CSR structure and numerical constraints. `Compare = true` additionally compares available incidence relations and selected metric relations. It does not require groups absent from the chosen masks. Validation completes its reductions before returning and performs no MPI collectives.

`valid_halo_plan(mesh, plan, rank, comm_size)` checks host peer metadata against map lengths. `duplicate_halo_receives(exec, mesh)` checks repeated receive destinations using temporary storage. Element-specific topology checks are also available as validation kernels; none of these establish global manifoldness or matching remote messages.

Counters are diagnostics, not automatic fatal errors. As in the example, explicitly pass `result.ok()` to `mpi::require` when invalid data must abort the job.
