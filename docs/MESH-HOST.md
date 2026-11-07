# Host Mesh

`krona::mesh::host` owns mesh arrays in ordinary host memory. It provides topology construction, migration, geometry, local reordering and optional auxiliary caches. It does not require Kokkos.

Use `RawMesh` while importing and distributing source connectivity. Use `MeshPart` in a solver or when preparing kernel views. The conversion pipeline is described in [MESH-PARTITIONER](MESH-PARTITIONER.md).

## Load a prepared mesh and add auxiliary data

```cpp
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/mesh/host/aux_geometry.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <cstdint>
#include <vector>

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv};
    return krona::mpi::guarded([&] {
        namespace host = krona::mesh::host;
        using krona::mesh::AuxConnType;
        using krona::mesh::AuxGeomType;
        const auto comm = krona::mpi::CommView::world();

        auto mesh = krona::io::solver_mesh::import_mesh_hdf5("mesh.h5", comm);
        auto connectivity = host::build_aux_connectivity(
            mesh, AuxConnType::CellFaces | AuxConnType::CellCellsByFace);
        auto geometry = host::build_aux_geometry(
            mesh, AuxGeomType::FaceInterpolationWeights |
                  AuxGeomType::FaceAreaOverNormalDistance);

        // Extend caches without rebuilding groups that are already ready.
        connectivity.add_connectivity(mesh, AuxConnType::NodeCells);
        geometry.add_geometry(mesh, AuxGeomType::FaceNonOrthogonalityVector);

        // Row 0 contains local face indices for owned cell 0.
        if (mesh.n_own > 0) {
            const auto begin = connectivity.cell_faces_offsets[0];
            const auto end = connectivity.cell_faces_offsets[1];
            for (auto j = begin; j < end; ++j) {
                const auto face = connectivity.cell_faces[j];
                const auto area = mesh.face_area[face];
                (void)area;
            }
        }
        return 0;
    });
}
```

The reader returns a validated, grouped, geometric `MeshPart`; rebuilding its faces or metrics is unnecessary. Loading is collective and requires the saved MPI size/rank numbering, matching geometry precision and equally wide or wider index types. The HDF5 snapshot stores the base mesh; construct auxiliary caches separately after loading.

Keep the application workflow inside `mpi::guarded`: validation failures use `mpi::fatal`, and the guard converts escaping exceptions, including allocation failures, into a job-wide fatal error.

## Data layout

### `RawMesh`: before migration

Header: `krona/mesh/host/raw_mesh.hpp`.

| Member | Contents |
| --- | --- |
| `cells` | Owned cell types and CSR connectivity in **global node IDs**. |
| `coordinates` | Only this rank's owned coordinate slice, in `x`, `y`, `z` arrays. A local cell can reference coordinates owned by another rank. |
| `boundary` | Imported boundary elements, source IDs and patch IDs. |
| `distribution` | Balanced block ownership of cells, nodes and dense boundary ordinals, plus the source rank. |
| `patches` | Replicated `Patch{name, family, source_type}` metadata. Source labels do not assign solver physics. |

`ElementTable::offsets` has `types.size() + 1` entries; `nodes(i)` borrows one element's oriented connectivity. All IDs are zero-based. `global_cell_id(i)` and `global_node_id(i)` derive IDs from the import distribution and are valid only while that ownership remains unchanged.

### `MeshPart`: after migration

Header: `krona/mesh/host/mesh_part.hpp`. Connectivity values are **local indices**; `cell_gid` and `node_gid` preserve global identities. Local indices use `LocalIndex`, global IDs use `GlobalIndex`, and CSR offsets use `size_t`. Coordinate and metric arrays use `GeometryReal`.

| Range / field | Meaning |
| --- | --- |
| Cells `[0, n_own)` | Cells owned by this rank. |
| Cells `[n_own, n_cells)` | One-layer face-neighbour ghosts. `cell_donor` identifies the owner rank; owned entries have donor `-1`. |
| Nodes `[0, n_nodes_own)` | Nodes used by at least one owned cell. **This is not exclusive ownership across MPI ranks.** |
| Nodes `[n_nodes_own, n_nodes)` | Additional nodes used only by local ghost cells. |
| `cell_nodes_offsets`, `cell_nodes` | Cell-to-node CSR, covering owned and ghost cells. |
| `face_nodes_offsets`, `face_nodes` | Face-to-node CSR with oriented node loops. |
| `face_owner[f]` | Always an owned local cell. |
| `face_neigh[f]` | Another owned cell, a ghost cell, or `invalid_local_index` (`-1`) on an external boundary. |
| `face_patch[f]` | Patch ID or `invalid_patch_id` (`-1`). A negative patch ID does not remove the face. |

Face intervals are valid only when `faces_grouped == true`:

| Face range | Category |
| --- | --- |
| `[0, n_internal_faces)` | Owned–owned. |
| `[n_internal_faces, n_inner_faces)` | Owned–ghost; partition interfaces. |
| `[n_inner_faces, n_faces)` | External boundary. |

An owned–owned face appears once locally. A partition interface appears on both adjacent ranks, with each rank's owned cell as owner. `n_faces_g` counts unique global faces; `n_ghost_copies_g` counts ghost copies, not unique cells.

### Patches and halo maps

Patch `p` contains `patch_faces[patch_face_offsets[p] ... patch_face_offsets[p+1])`. Unlabelled external faces remain in the boundary range but are absent from patch lists. No fallback patch is created by host face construction. Assign physical boundary conditions separately.

When faces_grouped == true, faces belonging to each patch occupy a contiguous range in the boundary face array. patch_face_offsets and patch_faces provide an explicit patch-to-face mapping and remain required by the current API.

Halo peers in `nb_ranks` are sorted and unique. For peer position `p`:

- `send_owned_local[send_offsets[p] ... send_offsets[p+1])` lists owned cells to send.
- `recv_ghost_local[recv_offsets[p] ... recv_offsets[p+1])` lists ghost cells to fill.

Offsets count **cells**, not scalar components or bytes. These maps define exchange order; storing a mesh or a `HaloPlanView` does not exchange field values. See [MESH-DEVICE](MESH-DEVICE.md) for the split between host MPI metadata and kernel maps.

## Geometry, reordering and validation

Headers: `geometry.hpp`, `reorder.hpp`, `validate.hpp` under `krona/mesh/host/`.

| Operation | Contract and result |
| --- | --- |
| `compute_mesh_geometry(mesh)` | Recomputes centroids and volumes for owned and ghost cells, and metrics for all local faces. Returns local GeometryStats; cell-volume statistics include owned cells only. Requires valid topology and consistent owned/ghost coordinates. No collective communication. |
| `compute_mesh_geometry(mesh, comm)` | Also reduces global statistics and refreshes global bounds. Every rank participates on the MPI main thread. |
| `reorder_local_mesh(mesh, method)` | Reorders owned cells, groups/sorts faces and remaps connectivity, halo indices, patches and existing metrics. Ghost order is preserved. No MPI exchanges. `NONE` still groups faces; `HILBERT_SFC` needs cell centroids; `RCM` uses owned-cell face adjacency. |
| `reorder_local_mesh(mesh, permutation)` | Uses an owned new-to-old permutation: `new[i] = old[permutation[i]]`. It must contain each owned index once and must not alias mesh storage. Face grouping still follows. |
| `validate_meshpart(mesh)` | Checks local grouped topology, geometry, patch and halo layout invariants. Invalid input aborts. It does not verify remote message agreement. |
| `collect_mesh_statistics(mesh, comm)` | Collective global counts, load extrema, volume and patch counts/flux areas. Call after local validation, with matching replicated patch metadata. |

Metrics include centroids and volumes for owned **and ghost** cells, plus face centroids, outward unit normals and areas. A face normal points out of its owner. For a quad, geometry uses a vertex-mean triangle fan; `face_area` is the magnitude of the summed area vector. Positive finite measures must fit `GeometryReal`.

Keep duplicated node coordinates consistent across ranks before recomputing metrics. Build caches and device storage **after** the final reordering.

## Auxiliary connectivity

Header: `krona/mesh/host/aux_connectivity.hpp`. Flags are `krona::mesh::AuxConnType`.

Every table uses CSR: row `i` spans `[offsets[i], offsets[i+1])`; values are local indices. `build_aux_connectivity(mesh, mask)` creates a cache; `cache.add_connectivity(mesh, mask)` adds only missing groups.

| Flag | Arrays | Rows → values |
| --- | --- | --- |
| `CellFaces` | `cell_faces_offsets`, `cell_faces` | Owned cells → all faces of each cell. An owned–owned face occurs in both cell rows. |
| `CellCellsByFace` | `cell_cells_face_offsets`, `cell_cells_face` | Owned cells → face neighbours, possibly ghosts. Sorted; one entry per connecting face, so repeated neighbour IDs are possible. |
| `NodeCells` | `node_cells_offsets`, `node_cells` | Owned-used nodes → all locally stored cells that use each node, including ghost cells. |
| `NodeFaces` | `node_faces_offsets`, `node_faces` | Owned-used nodes → all faces that use each node. |
| `CellCellsByNode` | `cell_cells_node_offsets`, `cell_cells_node` | Owned cells → sorted unique local indices of cells sharing at least one node, including ghost cells present in the face-neighbour halo; excludes the cell itself. |
| `None` / `All` | — | No tables / all five tables. |

Construction is local. Node-based relations only see cells present in `MeshPart`: a face-neighbour halo does not supply every remote vertex-only neighbour. Expand the halo separately if a complete distributed vertex stencil is required. Requesting `CellCellsByNode` alone may use a temporary node-to-cell table without marking `NodeCells` ready.

## Auxiliary geometry

Header: `krona/mesh/host/aux_geometry.hpp`. Flags are `krona::mesh::AuxGeomType`.

Face metrics require grouped faces and complete, synchronized base geometry. Let `P` be the owner centre, `N` the neighbour centre, `F` the face centre, `n` the outward unit normal and `A` the face area. Define `d = N - P` on interior/cut faces and `d = F - P` on boundaries; `h = dot(d, n)`.

| Flag | Output | Meaning |
| --- | --- | --- |
| `FaceCellDistance` | `face_cell_dist` | `length(d)`. |
| `FaceCellDistanceVector` | `face_cell_dist_x/y/z` | Components of `d`. |
| `FaceCellDistanceInv` | `face_cell_dist_inv` | `1 / length(d)`. |
| `FaceNormalDistance` | `face_normal_dist` | `h`. |
| `FaceNormalDistanceInv` | `face_normal_dist_inv` | `1 / h`. |
| `FaceAreaOverNormalDistance` | `face_area_over_normal_dist` | `A / h`. |
| `FaceInterpolationWeights` | `face_interp_weight` | Owner weight `w` in `phi_f = w*phi_P + (1-w)*phi_N`, using normal-projected distances. Boundary weight is zero; the other value is at the face. |
| `FaceNonOrthogonalityVector` | `face_non_ortho_x/y/z` | Over-relaxed correction `T = A * (n - dot(n,n)/h * d)`. |
| `WallDistance` | `cell_wall_dist` | Approximate graph distance for owned and ghost cells; computed by a separate collective call below. |
| `AllFaces` / `All` / `None` | — | All eight face groups / face groups plus wall-distance flag / no groups. |

`build_aux_geometry(mesh, mask)` and `cache.add_geometry(mesh, mask)` compute only missing **face** groups, in one pass. They allocate only requested outputs; requesting an inverse does not also store its direct distance. `WallDistance`, even within `All`, is not computed or marked ready by these functions.

Requested distances must be finite and positive; normal-based metrics require `h > 0`. Interpolation additionally requires interior cell centres on opposite sides of the face plane. Invalid geometry aborts instead of producing coefficients with an undefined stencil.

### Add wall distance

Insert this after constructing `mesh`, `connectivity` and `geometry` in the first example. Replace `"wall"` with a patch name from your mesh. The byte mask is replicated and must be identical on all ranks.

```cpp
std::vector<std::uint8_t> wall_patch(mesh.patches.size(), 0);
bool found_wall = false;
for (std::size_t p = 0; p < mesh.patches.size(); ++p) {
    if (mesh.patches[p].name == "wall") {
        wall_patch[p] = 1;
        found_wall = true;
    }
}
krona::mpi::require(found_wall, "Expected patch 'wall' was not found");

const auto wall_stats = geometry.add_wall_distance(
    mesh, connectivity, comm, wall_patch,
    {.stencil = host::WallDistStencil::FaceNeighbors,
     .max_sweeps = 100, .relative_tolerance = 1.0e-4});

// This example requires a converged field reaching every owned cell.
krona::mpi::require(wall_stats.converged, "Wall distance did not converge");
krona::mpi::require(wall_stats.unreachable_owned == 0,
                    "Some cells are unreachable from the selected walls");
```

The algorithm seeds wall-adjacent cells with owner-centre-to-face-centroid distance, then relaxes shortest paths along cell-centre edges with halo exchanges. It adds the required connectivity group automatically and refreshes ghost values before returning. This is a graph-path approximation, not exact nearest-surface distance.

All ranks participate, including empty ranks. `max_sweeps` must be positive; `relative_tolerance` must be finite and in `[0, 1)`. `FaceNeighbors` is the default; `NodeNeighbors` uses the locally available node stencil, with the halo limitation described above.

`WallDistanceStats` reports `sweeps`, global maximum `relative_change`, `converged` and global `unreachable_owned`. Unreachable cells retain infinity; convergence alone does not imply reachability. The result is marked ready even if the sweep limit was reached, so inspect the statistics before using it.

## Cache lifetime and mesh changes

Auxiliary caches are independent snapshots. `active_mask`, `has(flag)` and convenience methods such as `has_cell_faces()` describe ready groups. Repeating an add call does not refresh an existing group. `clear()` releases all groups and retained capacities.

| Mesh change | Required action |
| --- | --- |
| Coordinate motion only | Synchronize coordinates, recompute base geometry, clear/rebuild auxiliary geometry. Connectivity remains valid. |
| Reordering, migration or topology change | Clear/rebuild both auxiliary caches after the mesh is ready, even if entity counts stayed equal. |
| Any change affecting arrays already attached to device storage | Finish all kernel/exchange users and rebuild the affected storage. Borrowed CPU views may otherwise dangle; copied device arrays may be stale. |

Builders fill snapshot counts automatically for ready data. Manually populated connectivity caches must set `n_faces`, `n_cells_own` and `n_nodes_own`; geometry caches must set `n_faces` and `n_cells` (including ghosts), the selected arrays and `active_mask` consistently.

For source-independent migration, `MigrationSource` accepts cell connectivity, coordinates, patch metadata and explicit `P + 1` cell/node ownership endpoints. IDs must remain globally dense and contiguous by source rank; renumber/remap adapted inputs first. Migration consumes payloads, accepts one destination rank per source cell and currently requires packed MPI counts/displacements to fit `int`.
