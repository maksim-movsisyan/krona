# Mesh architecture

KRONA separates the host mesh, optional host caches, and Kokkos compute storage.
Topology and geometry are prepared on the host. Compute storage then exposes
selected arrays to kernels, borrowing host memory or owning copies in the
execution space's memory.

This document describes the current interfaces. Examples assume initialized MPI
and Kokkos, a valid `mpi::CommView comm`, and a mesh-file `path`. Code snippets
show setup only; keep their source objects and storage alive throughout the
computation.

## Objects and responsibilities

| Object | Role |
| --- | --- |
| `mesh::MeshPart` | Host mesh: local topology, basic geometry, halo maps, patch labels, and global metadata. |
| `mesh::MeshAuxConnectivity` | Optional host CSR tables, built and extended on demand. |
| `mesh::MeshAuxGeometry` | Optional host metrics, built and extended on demand; wall distance has a separate collective operation. |
| `compute::MeshStorage<Exec>` | Makes selected basic mesh arrays available in `Exec::memory_space`. |
| `compute::AuxConnectivityStorage<Exec>` | Attaches or uploads selected, already-built connectivity tables. |
| `compute::AuxGeometryStorage<Exec>` | Attaches or uploads selected, already-built geometry metrics. |
| `MeshView<Space>`, `AuxConnectivityView<Space>`, `AuxGeometryView<Space>` | Read-only, unmanaged Kokkos descriptors passed by value to kernels. |
| `compute::HaloPlanView` | Host spans over MPI peer ranks and send/receive offsets. |

Compute types live in `krona::mesh::compute`; host types live in `krona::mesh`.
Storage types expose `execution_space`, `memory_space`, and `view_type`. Views
expose `memory_space`; the execution-space instance is selected when work is
submitted.

## Local mesh contracts

`MeshPart` stores arrays in structure-of-arrays form. IDs are zero-based;
connectivity values use `LocalIndex`, global IDs use `GlobalIndex`, CSR offsets
use `std::size_t`, and coordinates and metrics use `GeometryReal`.

- Cells `[0, n_own)` are owned; `[n_own, n_cells)` are ghosts from a one-layer
  face-neighbour halo. Owned cells are ordered by Hilbert/SFC or RCM.
- Nodes `[0, n_nodes_own)` are used by at least one owned cell. The remaining
  nodes are used only by ghost cells. This prefix does not denote exclusive
  global node ownership.
- Every local face has an owned owner. Its neighbour is owned, ghost, or `-1`
  for an exterior boundary. Face nodes follow the oriented CGNS facet order
  relative to the owner.
- With `faces_grouped == true`, the face intervals are:

  | Interval | Faces |
  | --- | --- |
  | `[0, n_internal_faces)` | Owned–owned |
  | `[n_internal_faces, n_inner_faces)` | Owned–ghost |
  | `[n_inner_faces, n_faces)` | Boundary |

  These ranges describe the grouped/reordered mesh; migration alone leaves
  `faces_grouped == false`. Boundary faces from the decompositor are also grouped
  by patch. `patch_face_offsets` and `patch_faces` retain an explicit mapping for
  generality.
- Basic cell geometry covers owned and ghost cells. Face geometry contains
  centroids, areas, and unit normals. Compute storage transfers these arrays;
  it does not calculate them.

Patch labels describe the imported mesh. Physical boundary conditions are
assigned separately. Global IDs, patch metadata, and patch-face lists remain in
`MeshPart`; they are not part of the current `MeshView`.

## Load and select basic mesh data

```cpp
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

namespace mesh = krona::mesh;
namespace compute = krona::mesh::compute;
using Exec = krona::execution::ExecutionSpace;

auto host_mesh = krona::io::solver_mesh::import_mesh_hdf5(path, comm);
Exec exec{};

// The default selection is MeshData::FiniteVolume.
compute::MeshStorage<Exec> mesh_storage(host_mesh, {}, exec);

auto mesh_view = mesh_storage.view();
auto halo_plan = mesh_storage.halo_plan();
```

HDF5 import is collective on `comm` and restores the saved local snapshot. The
file requires its original MPI process count; a different count requires
repartitioning. The reader accepts equally wide or wider mesh index types and
requires matching geometry precision. It validates the imported mesh before
returning.

`MeshOptions::data` controls which arrays are attached or uploaded. Sizes and the
active mask are always present.

| `MeshData` flag | Selected data |
| --- | --- |
| `FaceCells` | Face owner and neighbour indices. |
| `CellNodes` | Cell types and cell-to-node CSR. |
| `FaceNodes` | Face types, owner-local facet indices, and face-to-node CSR. |
| `NodeCoordinates` | Node coordinates. |
| `CellGeometry` | Cell centroids and volumes, including ghosts. |
| `FaceGeometry` | Face centroids, unit normals, and areas. |
| `Halo` | Send-owned and receive-ghost index maps, plus the host halo plan. |
| `Connectivity` | `FaceCells \| CellNodes \| FaceNodes`. |
| `Geometry` | `CellGeometry \| FaceGeometry`. |
| `FiniteVolume` | `FaceCells \| Geometry \| Halo`. |
| `All` | All groups above. |

For example, use this selection instead of the default storage declaration when
cell vertices and node coordinates are also required:

```cpp
compute::MeshOptions options{
    .data = compute::MeshData::FiniteVolume |
            compute::MeshData::CellNodes |
            compute::MeshData::NodeCoordinates
};

compute::MeshStorage<Exec> mesh_storage(host_mesh, options, exec);
auto mesh_view = mesh_storage.view();
```

`MeshData::None` selects no arrays. `mesh_view.has(flags)` reports whether all
requested flag bits are active. Unselected arrays remain empty. Storage has no
API for adding groups after construction; choose a new selection when creating
a new storage object.

## Build and expose optional connectivity

Host caches build only missing requested tables and retain tables already built.
Connectivity construction is local and performs no MPI operations.

| `AuxConnType` flag | CSR relation |
| --- | --- |
| `CellFaces` | Owned cells → local faces. |
| `CellCellsByFace` | Owned cells → local face neighbours; one entry per connecting face. |
| `NodeCells` | Owned-used nodes → owned and ghost cells present locally. |
| `NodeFaces` | Owned-used nodes → local faces. |
| `CellCellsByNode` | Owned cells → unique local cells sharing a vertex. |

```cpp
#include <krona/mesh/aux_connectivity.hpp>
#include <krona/mesh/compute/aux_storage.hpp>

auto host_conn = mesh::build_aux_connectivity(
    host_mesh,
    mesh::AuxConnType::CellFaces | mesh::AuxConnType::CellCellsByFace);

// Extend the host cache without rebuilding existing tables.
host_conn.add_connectivity(host_mesh, mesh::AuxConnType::NodeCells);

// Expose only the subset needed by the computation.
compute::AuxConnectivityStorage<Exec> conn_storage(
    host_mesh, host_conn, mesh::AuxConnType::CellCellsByFace, exec);

auto conn_view = conn_storage.view();
```

The overload `(host_mesh, host_conn, exec)` attaches every group in
`host_conn.active_mask`. An explicit requested mask must be a subset of the
ready host groups. `AuxConnType::All` selects all connectivity groups.

## Build and expose optional geometry

Face metrics are calculated locally on the host, with double intermediates,
and stored as `GeometryReal`. Only requested arrays are allocated.

Available face flags cover cell-distance magnitude/vector/inverse,
normal-projected distance/inverse, area divided by normal distance,
interpolation weights, and the non-orthogonality correction vector.
`AuxGeomType::AllFaces` selects all face metrics.

For interior faces, the distance vector is `C_neighbour - C_owner`; for boundary
faces it is `F - C_owner`. Interpolation uses
`phi_face = w * phi_owner + (1 - w) * phi_other`, with boundary weight zero.

```cpp
#include <krona/mesh/aux_geometry.hpp>

const auto face_metrics =
    mesh::AuxGeomType::FaceCellDistanceVector |
    mesh::AuxGeomType::FaceInterpolationWeights;

auto host_geom = mesh::build_aux_geometry(host_mesh, face_metrics);

host_geom.add_geometry(
    host_mesh, mesh::AuxGeomType::FaceAreaOverNormalDistance);

compute::AuxGeometryStorage<Exec> geom_storage(
    host_mesh, host_geom, face_metrics, exec);

auto geom_view = geom_storage.view();
```

The overload `(host_mesh, host_geom, exec)` attaches every ready group. Storage
does not build missing metrics. Face-metric construction requires valid topology,
grouped faces, and complete, synchronized owned/ghost basic geometry.

### Wall distance

`WallDistance` is computed through a separate collective call. Neither
`add_geometry()` nor `build_aux_geometry()` computes it, even if the requested
mask includes `WallDistance` or `All`.

Compute it before creating the geometry storage that selects it; the following
storage declaration replaces the earlier geometry-storage example.

```cpp
// wall_patch is a replicated std::vector<std::uint8_t>, one entry per patch.
// Nonzero entries identify physical wall patches.
const auto stats = host_geom.add_wall_distance(
    host_mesh, host_conn, comm,
    std::span<const std::uint8_t>{wall_patch});

compute::AuxGeometryStorage<Exec> geom_storage(
    host_mesh, host_geom,
    face_metrics | mesh::AuxGeomType::WallDistance, exec);
```

All ranks participate, including empty or disconnected ranks, with identical
options and wall masks. `WallDistanceOptions` selects face or node neighbours
and controls the sweep limit and convergence tolerance. The operation refreshes
final ghost distances and returns replicated convergence statistics.

## Ownership and execution

The shared `detail::ArrayTransfer<Exec>` mechanism uses two paths:

| Target memory | Behaviour | Source lifetime |
| --- | --- | --- |
| Exactly `Kokkos::HostSpace` | Unmanaged views borrow the source vectors; managed buffers remain empty. | Sources must outlive every use of the views. |
| Other `Exec::memory_space` | Selected arrays are allocated and copied into managed buffers; unmanaged views reference those buffers. | Uploads complete before construction returns. |

On the copy path, `MeshStorage` also owns copies of the small host halo-plan
arrays. Once construction finishes, it does not borrow the source `MeshPart`.

All storage constructors currently finish uploads before returning, using the
supplied `exec`. Copies are submitted through `deep_copy(exec, ...)`, followed
by an instance fence. Exception handlers also drain queued copies before owned
buffers are released. Construction therefore has a synchronous, ready-to-use
contract.

Views contain read-only unmanaged descriptors. Copying a view copies descriptors,
not array elements, and does not extend the backing memory's lifetime. Keep the
storage or borrowed host sources alive until all kernels and other users finish.
Do not modify data concurrently with readers or reallocate borrowed vectors.

Storage classes are non-copyable and non-movable. Constructors reject temporary
objects supplying the source arrays, including on the copy path. `view()` and
`halo_plan()` cannot be called on temporary storage objects.

`HaloPlanView` belongs to host MPI code and must not be captured in GPU kernels.
Its offsets count cells, not field components or bytes; peer `p` selects
`[offsets[p], offsets[p + 1])` in the corresponding halo index map. The plan is
available only when `MeshData::Halo` was selected. It does not execute exchanges.

## Optional diagnostics

`compute::validation` provides local checks for selected view shapes, indices,
CSR tables, geometry, and optional incidence comparisons. It performs no MPI
collectives. The regular validator creates no explicit temporary arrays;
`duplicate_halo_receives()` uses a temporary counter array.

```cpp
#include <krona/mesh/compute/validation.hpp>

// Match the selection used by the default MeshStorage.
const auto result = compute::validation::validate<
    compute::MeshData::FiniteVolume>(exec, mesh_storage.view());

const bool valid = result.ok();
```

The validator defaults to `MeshData::All`, whereas `MeshStorage` defaults to
`FiniteVolume`; explicitly select groups available in the storage. Counters count
bad entities or CSR rows, not individual failed predicates. The caller decides
how to handle a failed result.

To check the auxiliary subsets from the examples above, including available
metric identities and incidence comparisons:

```cpp
const auto detailed = compute::validation::validate<
    compute::MeshData::FiniteVolume,
    mesh::AuxConnType::CellCellsByFace,
    mesh::AuxGeomType::FaceCellDistanceVector |
        mesh::AuxGeomType::FaceInterpolationWeights,
    true>(exec, mesh_storage.view(), conn_storage.view(), geom_storage.view());
```

`Compare == true` checks only relations supported by the selected arrays. Shape
errors prevent every kernel launch. Incidence gathers run only after the earlier
checks have succeeded, so malformed CSR/index data cannot reach those gathers.

Halo-plan checks and receive-destination uniqueness are separate operations:

```cpp
const bool plan_valid = compute::validation::valid_halo_plan(
    mesh_storage.view(), mesh_storage.halo_plan(), comm.rank(), comm.size());

const auto duplicates = compute::validation::duplicate_halo_receives(
    exec, mesh_storage.view());
```

### When to validate

One-time checks during setup or in constructors of mesh-dependent objects fit
the current API. The checks require initialized Kokkos, an execution space able
to access the selected arrays, and backing memory that stays alive and unchanged.
Their reductions return host counters and finish before returning. On a GPU,
such a constructor waits for the relevant queued work; it cannot make an
immediate host decision from `result.ok()` while leaving those checks pending.

Work submitted earlier to the same `exec` is ordered before its checks.
Producers on another execution-space instance must be synchronized first.
Validation is not a general synchronization operation: shape failures and empty
checks can return without launching a reduction. Use it for diagnostics or
setup decisions, rather than repeatedly in an asynchronous compute loop.

The regular validator does not establish manifoldness, MPI halo-message
agreement, patch correctness, or element-specific topology. Type-aware checks
are separate `ValidateCellTypes` / `ValidateFaceTypes` functors and require
caller-provided device-callable rules. A local success is not a global mesh or
communication-consistency proof.

### Reduction results and asynchronous use

Kokkos can store a scalar reduction result in a writable rank-zero View in the
chosen memory space. This permits potentially asynchronous submission. The
current `validation::validate()` uses host scalars instead and does not expose
an asynchronous result object.

Existing validation functors can be submitted directly once their required
counts, availability, and shapes have been checked. For example, using the
existing cell-geometry checker:

```cpp
using Space = typename Exec::memory_space;
using Counter = Kokkos::View<std::size_t, Kokkos::Device<Exec, Space>>;
using CellCheck = compute::validation::ValidateCells<
    compute::MeshData::CellGeometry, mesh::AuxGeomType::None, Space>;

Counter device_bad(Kokkos::view_alloc(
    exec, Kokkos::WithoutInitializing, "mesh/cell_validation_result"));

Kokkos::parallel_reduce(
    "mesh/check_cell_geometry",
    Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>(
        exec, 0, static_cast<std::size_t>(mesh_view.sizes.n_cells)),
    CellCheck{mesh_view, {}}, device_bad);
```

`Counter` has no `*`: it represents one value. Keep the result and input storage
alive until completion. A later operation on the same `exec` can consume the
result in order; reading it on the host requires completion and, for device-only
memory, a copy to host storage. Allocation itself need not be asynchronous.
See the [Kokkos reduction interface](https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html)
for result and submission semantics. This example uses an existing checker; it
does not replace the regular validator's guarded sequence of checks.

## Known limitations and future considerations

- A face-neighbour halo does not contain every remote vertex-only neighbour.
  `NodeCells`, `CellCellsByNode`, and the node wall-distance stencil describe
  only cells present in the local `MeshPart`.
- Wall distance is a graph-path approximation seeded by owner-to-wall-face
  centroid distances. It is not an exact nearest-surface distance. Unreachable
  cells retain positive infinity; inspect `stats.unreachable_owned` even when
  relaxation converges.
- Storage has no incremental upload, refresh, or reallocation interface. Host
  cache extensions do not update an existing storage selection or its copied
  metadata. GPU copies do not track later host changes.
- Coordinate-only motion leaves connectivity caches valid but invalidates
  geometry caches. Topology or numbering changes require clearing/rebuilding
  both caches, even when counts remain unchanged. Reordering must update
  connectivity, halo indices, patch indices, and basic geometry together.
  Both host cache types provide `clear()` to release their stored arrays.
- For future moving/adaptive meshes, finish outstanding users before changing
  backing arrays, rebuild affected host caches, and recreate the relevant
  storage. Automatic invalidation is not implemented.
- If upload/computation overlap becomes necessary, revisit the constructor's
  completion boundary and the lifetime of source and destination arrays.
  Preserve the exception-cleanup guarantee when changing that contract.
- An asynchronous validation interface would need deferred result consumption
  and the same shape/index gates before incidence gathers. Returning an
  immediately usable host `Result` requires a completion boundary.
- Aux geometry and storage not necessarily need host mesh object, delete it in future
architecture refactoring to minimize dependencies among modules.

