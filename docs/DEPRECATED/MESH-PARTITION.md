# KRONA distributed solver mesh

This document describes `krona::mesh::MeshPart`, the distributed preprocessing pipeline, and the version 2 HDF5 solver-mesh format. The mesh is a snapshot of a conforming three-dimensional unstructured grid, split into rank-local subdomains with owned cells and one layer of face-neighbour ghost cells.

The mesh payload remains distributed throughout preprocessing and I/O. Rank 0 handles command-line parsing, diagnostics, and small shared outputs; it never assembles the complete mesh.

## 1. Representation and supported elements

`MeshPart` uses structure-of-arrays storage for entity attributes and CSR storage for variable-length connectivity. It holds topology, node coordinates, geometric metrics, source patch labels, and communication indices. Solution fields and physical boundary-condition parameters are managed separately.

All IDs and local indices are zero-based. `LocalIndex` and `GlobalIndex` must be signed types capable of representing the `-1` sentinel. Their configured widths may differ. Coordinates and metrics use `GeometryReal`; in-memory CSR offsets use `std::size_t`.

| Element | Stored `CellType` value | Dimension | Nodes | Facets | Role in the solver mesh |
| --- | ---: | ---: | ---: | ---: | --- |
| `TET` | 0 | 3 | 4 | 4 | Volume cell |
| `PYRA` | 1 | 3 | 5 | 5 | Volume cell |
| `PRISM` | 2 | 3 | 6 | 5 | Volume cell |
| `HEXA` | 3 | 3 | 8 | 6 | Volume cell |
| `TRI` | 4 | 2 | 3 | 3 | Triangular face |
| `QUAD` | 5 | 2 | 4 | 4 | Quadrilateral face |
| `BAR` | 6 | 1 | 2 | 2 | Source boundary edge for 2D input |
| `NODE` | 7 | 0 | 1 | 0 | Element utility support |

These are KRONA enum values, not CGNS or VTK numeric element codes. Volume connectivity follows canonical CGNS SIDS node ordering. The facet tables specify outward loops for positively oriented reference cells. The solver pipeline supports linear `TET/PYRA/PRISM/HEXA` cells and `TRI/QUAD` faces; the broader enum does not imply additional solver-mesh dimensions or higher-order elements.

A `FaceKey` contains sorted global node IDs with unused slots set to `-1`. It identifies a face independently of winding. It must never replace oriented face connectivity. The current matching scheme requires conforming interfaces and rejects more than two volume cells sharing one complete face.

## 2. Local storage contracts

### Cells and nodes

| Local range | Meaning |
| --- | --- |
| Cells `[0, n_own)` | Cells owned and advanced by this rank |
| Cells `[n_own, n_cells)` | Complete topology of remote face-neighbour cells |
| Nodes `[0, n_nodes_own)` | Nodes referenced by at least one owned cell |
| Nodes `[n_nodes_own, n_nodes)` | Additional nodes referenced only by ghost cells |

`n_nodes_own` describes local usage, not exclusive global node ownership. The same node GID can occur on several ranks, including in their owned-cell node prefixes. A ghost can also have copies on several ranks.

Every cell attribute array has `n_cells` entries, including ghosts. `cell_gid` identifies the global cell. `cell_donor` is `-1` for owned cells and the current owning MPI rank for ghosts. Ghosts are initially arranged by donor rank and GID; the local owned-cell reorder preserves ghost indices.

`cell_nodes_offsets` has `n_cells + 1` entries. Cell `c` uses:

```cpp
cell_nodes[cell_nodes_offsets[c] : cell_nodes_offsets[c + 1]]
```

This notation denotes a half-open slice. Entries in `cell_nodes` are local node indices, not global IDs. Owned cells reference only nodes below `n_nodes_own`; ghosts may reference either node range. Each row has the fixed arity of its cell type.

Every node attribute array has `n_nodes` entries. `node_gid` provides global identity; `node_x/y/z` hold coordinates. Within the current dense snapshot namespace, cell GIDs lie in `[0, n_cells_g)` and node GIDs in `[0, n_nodes_g)`. These are snapshot identities; adaptation needs an explicit policy for identity across snapshots.

### Faces and orientation

Every local face touches at least one owned cell. `face_owner` always lies in `[0, n_own)`. `face_neigh` identifies an owned cell, a ghost cell, or `invalid_local_index == -1` for an exterior boundary.

An owned-owned face is stored once locally. A partition-cut face has one copy on each adjacent rank, with that rank's owned cell as its owner. A physical boundary face has one copy globally. The final representation contains no ghost-ghost faces and no persistent face GID array.

`face_local_index` identifies the owner's canonical facet. `face_nodes_offsets` has `n_faces + 1` entries; `face_nodes` stores the corresponding owner-relative oriented loop using local node indices. Every face attribute and metric array has `n_faces` entries.

The normal points outward from the physical owner cell. Opposite copies of a cut face have opposite outward normals. Reordering changes numeric indices without switching physical ownership: `face_owner < face_neigh` is not a required invariant.

### Three face regions

The following ranges are valid only when `faces_grouped == true`:

| Face range | Neighbour | Ordering | Typical solver use |
| --- | --- | --- | --- |
| `[0, n_internal_faces)` | Owned | Nondecreasing `(owner, neighbour)` | Local interior fluxes |
| `[n_internal_faces, n_inner_faces)` | Ghost | Nondecreasing `(owner, neighbour)` | Fluxes using halo values |
| `[n_inner_faces, n_faces)` | `-1` | Patch, then owner; unlabelled faces last | Boundary conditions |

`n_inner_faces` is the total number of local interior faces, including cut faces. Thus the three lengths are `n_internal_faces`, `n_inner_faces - n_internal_faces`, and `n_faces - n_inner_faces`.

Migration leaves grouping unfinished. Even `ReorderMethod::NONE` must run the grouping stage before final validation or export. Empty ranks follow the same stage sequence.

This arrangement allows halo communication to overlap owned-owned face work. It does not guarantee MPI progress or complete inputs for higher-order reconstruction or viscous stencils. Those requirements belong to the solver.

### Halo communication maps

`nb_ranks` lists distinct peer ranks in increasing order. Both `send_offsets` and `recv_offsets` have `nb_ranks.size() + 1` entries.

| Array slice for peer `i` | Meaning |
| --- | --- |
| `send_owned_local[send_offsets[i] : send_offsets[i+1]]` | Owned cell indices whose field values are sent to `nb_ranks[i]` |
| `recv_ghost_local[recv_offsets[i] : recv_offsets[i+1]]` | Ghost cell slots filled from `nb_ranks[i]` |

Each ghost occurs exactly once in the receive list and belongs to the corresponding donor group. Send entries reference owned cells. A cell may be sent to several peers.

Migration's request/reply protocol establishes matching remote send and receive order. Reordering remaps `send_owned_local` values while preserving their sequence within each peer group. Independently sorting those entries would invalidate the protocol. The maps describe cell-field exchange; they do not establish unique shared-node ownership or implement coordinate synchronization.

### Patches and boundary conditions

`patches` is small replicated source metadata. Each patch stores `name`, `family`, and `source_type`. Its index is the patch ID; `source_type` is an import label, not an executable solver boundary condition.

Interior faces have `face_patch == invalid_patch_id == -1`. Exterior faces can have a valid patch ID or remain unlabelled with `-1`. Consequently, a boundary must be identified from topology, not from the presence of a patch label.

`patch_face_offsets` has `patches.size() + 1` entries. `patch_faces` lists each labelled local boundary face exactly once, grouped by patch. Empty patches have empty rows; unlabelled boundary faces are excluded. The optional TOML template assigns suggested physical BC types separately and requires application-specific parameters.

### Geometry

Cell centroids and volumes are stored for all owned and ghost cells. Face centroids, unit normals, and areas are stored for every local face. Migration leaves these arrays empty; final solver-mesh export requires populated geometry.

The pipeline's geometry convention integrates quadrilateral facets as four triangles connecting consecutive vertices to the vertex mean. For a face with oriented area vector `S`, it stores:

```text
face_area = norm(S)
face_normal = S / norm(S)
```

For a warped quadrilateral, this effective flux area may differ from the sum of triangle areas. Cell volume and first-moment integration use the same facet convention. Face centres use triangle-area weights.

Volumes and flux areas must be positive and finite; coordinates, centroids, and normals must be finite. The geometry stage checks owner/neighbour sides and does not silently repair inverted cells. These checks assume valid CFD cells and do not constitute a general self-intersection or concave-cell proof.

Shared and ghost coordinate copies must be synchronized before geometry is recomputed. The geometry function performs no coordinate exchange. Global volume sums include owned cells only, and global bounds use nodes referenced by owned cells.

### CSR and global counts

Every local CSR offset array starts at zero, is nondecreasing, has one terminal entry, and ends at its payload length. Offsets measure element positions, not bytes.

Global metadata is replicated on all ranks:

| Field | Meaning |
| --- | --- |
| `n_cells_g` | Global owned-cell count |
| `n_nodes_g` | Source/current snapshot global node namespace size; not the sum of local node copies |
| `n_faces_g` | Unique physical faces, including interior and boundary faces |
| `n_bfaces_g` | All exterior faces, including unlabelled faces |
| `n_labeled_bfaces_g` | Exterior faces assigned to patches |
| `n_cut_faces_g` | Unique faces between partitions |
| `n_ghost_copies_g` | Sum of local ghost counts, not distinct ghost GIDs |
| `bbox_lo`, `bbox_hi` | Global bounds of nodes referenced by owned cells |

The collective statistics stage checks these conservation identities:

```text
sum(n_own)                                      = n_cells_g
sum(n_cells - n_own)                            = n_ghost_copies_g
sum(n_inner_faces - n_internal_faces)           = 2 * n_cut_faces_g
sum(n_faces - n_inner_faces)                    = n_bfaces_g
sum(n_internal_faces) + n_cut_faces_g
    + n_bfaces_g                               = n_faces_g
sum(patch_faces.size())                         = n_labeled_bfaces_g
```

Therefore, the sum of local face counts equals `n_faces_g + n_cut_faces_g`. Summing local node counts does not recover `n_nodes_g`; nodes are duplicated and unused source nodes may be absent locally.

## 3. Local ordering

Normal production preprocessing uses either **Reverse Cuthill–McKee (RCM)** or the **Hilbert space-filling curve**. The executable accepts `--reorder RCM` or `--reorder SFC`; `SFC` selects `ReorderMethod::HILBERT_SFC`. The supplied CLI still defaults to `NONE`, so production runs should explicitly select the desired method.

| Method | Ordering basis | Required input |
| --- | --- | --- |
| RCM | Adjacency formed from owned-owned faces; component-wise reversed Cuthill–McKee traversal | Local topology |
| Hilbert SFC | 63-bit Hilbert keys of owned-cell centroids, using 21 bits per axis and a common spatial scale; GID breaks equal-key ties | Finite cell centroids |
| NONE | Preserve cell order; still group and sort faces | Local topology |

RCM is a bandwidth/profile heuristic. Hilbert ordering is a spatial-locality heuristic. Neither guarantees optimal solver performance or changes the partition assignment.

The stage permutes owned cell arrays and populated metrics, remaps face references and halo send indices, permutes all face arrays consistently, and rebuilds patch-face CSR. Ghost indices and node numbering stay unchanged. Global IDs survive the local reorder.

Any external solution field, matrix, lookup, or cache indexed by cells or faces must receive the same permutation or be rebuilt. Complete outstanding communication before changing indices or referenced storage. The saved file preserves the resulting order but does not record which algorithm produced it.

## 4. HDF5 solver snapshot

`export_mesh_hdf5(mesh, path, comm, part2rank)` writes one shared versioned HDF5 container. `import_mesh_hdf5(path, comm)` restores this rank's partition. This is a mesh snapshot, not a complete solver checkpoint or an archive of the original CGNS file.

The I/O contract requires:

- Active MPI and a parallel HDF5 build with the metadata APIs used by the implementation, documented by the pipeline as HDF5 1.10 or newer.
- All ranks in `comm`, including empty ranks, calling the same operations on the MPI main thread with identical paths and replicated metadata.
- A valid grouped mesh with complete metrics, validated before export. The application performs local validation and collective statistics before calling the writer.
- A replicated `part2rank` permutation, or an empty mapping to select rank order. Mapping values and patch metadata must agree across ranks; the writer does not perform a cross-rank equality check.
- An existing output parent directory. Export truncates and replaces an existing file; it is not an atomic checkpoint transaction.

Each rank writes a non-overlapping slice of every local dataset. A small directory `Allgather` collects lengths, not mesh payloads. Rank 0 writes shared metadata while other ranks contribute empty selections. All transfers request `H5FD_MPIO_COLLECTIVE`; datatype conversion can affect the underlying MPI-IO path. A globally empty dataset is created, and its payload call is skipped consistently by all ranks.

The reader requires the saved MPI size and preserves the meaning of saved rank numbers. The job can run on different machines, but assigning saved subdomains to different rank numbers requires donor/peer remapping and migration support beyond this reader.

For valid writer-produced files, signed local/global indices can widen from 32 to 64 bits. Index narrowing is rejected by the header checks even if particular values would fit. `GeometryReal` precision must match the file; the reader does not recompute geometry. A precision change needs a separate conversion/recomputation workflow.

### Partition prefixes and local offsets

Let `P` be the saved MPI size and `D[p,c]` be directory column `c` for partition `p`. For every payload category:

```text
count(p,c) = D[p,c]
start(p,c) = sum(D[q,c] for q < p)
total(c)   = sum(D[q,c] for all q)
```

Rank `r` reads partition `p = inverse(part2rank)[r]`. Prefixes follow **partition order**, which may differ from MPI rank order. Each category has its own prefix; cell counts cannot be reused for halo or patch arrays.

Full local CSR arrays are concatenated, including each partition's initial zero and terminal offset. Their values remain relative to that partition's local connectivity payload and are not shifted by its disk position. Thus `/cells/offsets`, for example, is a sequence of independent local CSR blocks, not one global CSR array.

### Expandable file layout

All datasets below are **one-dimensional**. The partition directory is logically `P × 19` but is physically stored as a flat array of length `19P`. Dataset paths are exact and case-sensitive. HDF5 manages physical byte locations; this is the logical schema.

Type notation: `U8` and `U64` are unsigned integers; `LI` and `GI` are the saved signed local/global index types; `R` is the saved IEEE FP32/FP64 geometry type; `RI` is the signed type selected for C++ `int` (normally `I32`); `PI` is the type selected for `PatchId` (`I32` in the current mesh definition). The writer uses explicit little-endian file types. All CSR and string offsets are always `U64`, independent of host `size_t`.

Let `Σc` denote the sum of directory column `c` over all partitions, `K` the patch count, and `B` the total patch-string byte count.

<details>
<summary><strong>KRONA HDF5 v2 — complete groups, datasets, and metadata indices</strong></summary>

#### Root groups

| Group | Contents |
| --- | --- |
| `/meta` | Format/build header and global bounds |
| `/partition` | Partition-to-rank mapping and payload directory |
| `/cells` | Owned/ghost cell topology and metrics |
| `/nodes` | Required node GIDs and coordinates |
| `/faces` | Owner-relative face topology, labels, and metrics |
| `/halo` | Peer ranks and ordered cell-field exchange indices |
| `/patches` | Shared patch strings and local labelled-face lists |

#### Shared metadata

| Dataset | Type | Length | Meaning |
| --- | --- | --- | --- |
| `/meta/header` | `U64` | 16 | Header fields listed below |
| `/meta/bbox_lo` | `R` | 3 | Global lower coordinate bounds |
| `/meta/bbox_hi` | `R` | 3 | Global upper coordinate bounds |
| `/partition/part2rank` | `RI` | `P` | `part2rank[p]` is the MPI rank for partition `p` |
| `/partition/directory` | `U64` | `19P` | Flat partition rows in partition order |
| `/patches/string_offsets` | `U64` | `3K + 1` | Offsets into shared string bytes |
| `/patches/string_bytes` | `U8` | `B` | Concatenated patch strings without added terminators |

#### `/meta/header` fields

| Index | Value |
| ---: | --- |
| 0 | Magic: `0x4b524f4e414d4553` |
| 1 | Schema version: `2` |
| 2 | Saved MPI size `P` |
| 3 | Local index width in bits: 32 or 64 |
| 4 | Global index width in bits: 32 or 64 |
| 5 | Geometry width in bits: 32 or 64 |
| 6 | Number of patches `K` |
| 7 | `n_cells_g` |
| 8 | `n_nodes_g` |
| 9 | `n_faces_g` |
| 10 | `n_bfaces_g` |
| 11 | `n_labeled_bfaces_g` |
| 12 | `n_cut_faces_g` |
| 13 | `n_ghost_copies_g` |
| 14 | Patch-string byte count `B` |
| 15 | Offset width in bits: `64` |

#### `/partition/directory` columns

Row `p` begins at flat position `19p`.

| Column | Category / field | Stored local value |
| ---: | --- | --- |
| 0 | `cells` | `n_cells` |
| 1 | `cell_indices` | `cell_nodes.size()` |
| 2 | `nodes` | `n_nodes` |
| 3 | `faces` | `n_faces` |
| 4 | `face_indices` | `face_nodes.size()` |
| 5 | `peers` | `nb_ranks.size()` |
| 6 | `send_indices` | `send_owned_local.size()` |
| 7 | `recv_indices` | `recv_ghost_local.size()` |
| 8 | `patch_indices` | `patch_faces.size()` |
| 9 | `cell_offsets` | `n_cells + 1` |
| 10 | `face_offsets` | `n_faces + 1` |
| 11 | `peer_offsets` | `nb_ranks.size() + 1` |
| 12 | `patch_offsets` | `K + 1` |
| 13 | `owned` | `n_own` |
| 14 | `owned_nodes` | `n_nodes_own` |
| 15 | `internal_faces` | `n_internal_faces` |
| 16 | `inner_faces` | `n_inner_faces` |
| 17 | `grouped` | `1`, corresponding to `faces_grouped == true` |
| 18 | `flipped` | `n_flipped`, preserved diagnostic counter |

Only columns 0–12 define payload lengths/prefixes. Columns 13–18 contain local scalar state. Both send and receive offset datasets use column 11, but their payload lists use distinct columns 6 and 7.

#### `/cells` datasets

| Dataset | Type | Length | `MeshPart` member |
| --- | --- | --- | --- |
| `/cells/type` | `U8` | `Σcells` | `cell_type` |
| `/cells/gid` | `GI` | `Σcells` | `cell_gid` |
| `/cells/donor` | `RI` | `Σcells` | `cell_donor` |
| `/cells/offsets` | `U64` | `Σcell_offsets` | `cell_nodes_offsets` |
| `/cells/nodes` | `LI` | `Σcell_indices` | `cell_nodes` |
| `/cells/centroid_x` | `R` | `Σcells` | `cell_centroid_x` |
| `/cells/centroid_y` | `R` | `Σcells` | `cell_centroid_y` |
| `/cells/centroid_z` | `R` | `Σcells` | `cell_centroid_z` |
| `/cells/volume` | `R` | `Σcells` | `cell_volume` |

#### `/nodes` datasets

| Dataset | Type | Length | `MeshPart` member |
| --- | --- | --- | --- |
| `/nodes/gid` | `GI` | `Σnodes` | `node_gid` |
| `/nodes/x` | `R` | `Σnodes` | `node_x` |
| `/nodes/y` | `R` | `Σnodes` | `node_y` |
| `/nodes/z` | `R` | `Σnodes` | `node_z` |

#### `/faces` datasets

| Dataset | Type | Length | `MeshPart` member |
| --- | --- | --- | --- |
| `/faces/owner` | `LI` | `Σfaces` | `face_owner` |
| `/faces/neighbour` | `LI` | `Σfaces` | `face_neigh` |
| `/faces/type` | `U8` | `Σfaces` | `face_type` |
| `/faces/patch` | `PI` | `Σfaces` | `face_patch` |
| `/faces/local_index` | `U8` | `Σfaces` | `face_local_index` |
| `/faces/offsets` | `U64` | `Σface_offsets` | `face_nodes_offsets` |
| `/faces/nodes` | `LI` | `Σface_indices` | `face_nodes` |
| `/faces/centroid_x` | `R` | `Σfaces` | `face_centroid_x` |
| `/faces/centroid_y` | `R` | `Σfaces` | `face_centroid_y` |
| `/faces/centroid_z` | `R` | `Σfaces` | `face_centroid_z` |
| `/faces/normal_x` | `R` | `Σfaces` | `face_normal_x` |
| `/faces/normal_y` | `R` | `Σfaces` | `face_normal_y` |
| `/faces/normal_z` | `R` | `Σfaces` | `face_normal_z` |
| `/faces/area` | `R` | `Σfaces` | `face_area` |

#### `/halo` datasets

| Dataset | Type | Length | `MeshPart` member |
| --- | --- | --- | --- |
| `/halo/ranks` | `RI` | `Σpeers` | `nb_ranks` |
| `/halo/send_offsets` | `U64` | `Σpeer_offsets` | `send_offsets` |
| `/halo/recv_offsets` | `U64` | `Σpeer_offsets` | `recv_offsets` |
| `/halo/send_owned` | `LI` | `Σsend_indices` | `send_owned_local` |
| `/halo/recv_ghost` | `LI` | `Σrecv_indices` | `recv_ghost_local` |

#### `/patches` local datasets and string ordering

| Dataset | Type | Length | `MeshPart` member |
| --- | --- | --- | --- |
| `/patches/offsets` | `U64` | `Σpatch_offsets = P(K+1)` | `patch_face_offsets` |
| `/patches/faces` | `LI` | `Σpatch_indices` | `patch_faces` |

The shared strings use this exact order for each patch `p`:

| String index | Content |
| --- | --- |
| `3p` | `patches[p].name` |
| `3p + 1` | `patches[p].family` |
| `3p + 2` | `patches[p].source_type` |

String `s` occupies bytes `[string_offsets[s], string_offsets[s+1])`. Equal adjacent offsets represent an empty string. The writer preserves string bytes without truncation or encoding conversion; UTF-8 is the pipeline's metadata convention. Shared strings are stored once, while patch-face CSR is stored separately for every partition.

There are **41 datasets in total**: 7 shared datasets and 34 partitioned arrays. There are no per-rank groups, stored inverse permutation, stored reorder-method flag, or serialized C++ structs.

</details>

## 5. Preprocessing pipeline

The following summarizes actual calls and their documented contracts in `mesh_partitioner.cpp`.

| Stage | Operation | Result |
| --- | --- | --- |
| 1. Import | `io::cgns::read_parallel` | Distributed `RawMesh` with block-owned cells, coordinates, boundary records, and replicated patch labels. Planar 2D input is automatically extruded to one PRISM/HEXA layer. |
| 2. Topology | `mesh::build_faces` | Globally unique face records and a symmetric cell dual graph. Batched hash rendezvous matches sorted node keys; exterior faces can be unlabelled. |
| 3. Partition | `mesh_partition::partition_cells` | dKaMinPar destination ranks and inverse partition/rank mappings. Unit weights balance cell counts; this stage does not move mesh data. |
| 4. Migration | `mesh::migrate_local_mesh` | Redistributed cells, cut-face copies, one face-neighbour halo, local node numbering, coordinates, patches, and field-exchange maps. Source mesh payloads and face records are consumed. |
| 5. Geometry | `mesh::compute_mesh_geometry` | Metrics for owned/ghost cells and local faces; refreshed global bounds and geometric statistics. |
| 6. Local order | `mesh::reorder_local_mesh` | RCM or Hilbert owned-cell order and three face regions, with topology, metrics, halo indices, and patch lists updated together. |
| 7. Verify | `mesh::validate_meshpart`, then `mesh::collect_mesh_statistics` | Local structural validation and collective count/metric conservation checks. Optional BC template is written from small metadata. |
| 8. Export | `io::solver_mesh::export_mesh_hdf5` | Shared mesh snapshot using `pr.part2rank` for disk order. Optional `io::vtk::write_vtu` writes visualization pieces and PVTU indices. |

The source dual graph is released before migration; target-rank storage is released after migration. Shared-memory grouping of destination ranks is a placement heuristic, not a guarantee of minimal network traffic. Local Hilbert ordering and partition-to-rank placement are distinct operations.

Example production invocation:

```bash
mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5 --reorder=RCM -v
mpirun -np 4 krona_mesh_partition mesh.cgns mesh.h5 --reorder=SFC --bc=mesh.toml --vtu=vtk -v
```

All `--reorder`, `--bc` and `--vtu` are optional. Visualization is an output view of the mesh and cannot replace the solver snapshot.

## 6. Reuse for moving and adaptive meshes

The reusable units are topology construction, graph partitioning, migration, geometry, local ordering, validation, and snapshot I/O. Their input ownership and connectivity contracts must be established explicitly; a final `MeshPart` is not directly interchangeable with the source `RawMesh` required by the import-time pipeline.

| Component / data | Coordinate motion with fixed connectivity and ownership | Conforming topology adaptation / remeshing |
| --- | --- | --- |
| CGNS import and 2D extrusion | Keep the initialized mesh; no repeated import is needed | Usually replace with generated/adapted source data; reimport only when the workflow uses a new file |
| Faces and dual graph | Preserve existing topology | Rebuild after assigning consistent IDs, valid source ownership, and boundary connectivity |
| dKaMinPar partitioning | Not required solely because coordinates move | Reuse on a rebuilt distributed graph when rebalancing is needed; current unit weights do not model heterogeneous cell cost |
| Migration and halo maps | Preserve while ownership and adjacency remain fixed | Reuse after constructing a valid migration source; rebuild halo, local connectivity, and patch lists |
| Coordinates | Synchronize every shared/ghost copy after updates | Establish unique source coordinate ownership and distribute required copies |
| Geometry and bounds | Recompute after coordinate synchronization | Recompute after rebuilding topology and halo |
| RCM / Hilbert and face grouping | Existing grouping remains valid. RCM topology is unchanged; Hilbert locality can be refreshed if useful | Rerun grouping and normally choose a fresh RCM or Hilbert order |
| Validation/statistics | Reuse after updates | Reuse after reconstruction |
| HDF5 writer/reader | Save/restore the updated validated snapshot with the same rank mapping | Save/restore a new supported conforming snapshot; refinement history is not retained |

### Fixed-topology motion

A minimal update sequence is: update coordinates, synchronize shared/ghost node copies, recompute geometry and bounds, and validate the resulting mesh. Connectivity, patches, halo cell maps, and face regions remain valid if their identities and ownership are unchanged. Local reordering is optional and requires corresponding updates to solver fields and caches.

This is geometric reuse, not a complete ALE or moving-interface implementation. The current structure does not store mesh velocity, swept-face volumes, previous-time geometry, time integration state, or sliding/overset interface maps. An ALE solver must add the required motion data and geometric conservation treatment. Coordinate motion that changes interface connectivity belongs to the topology-change path.

### Conforming adaptation

Adaptation must supply new cells, nodes, boundary labels, and valid positive orientations. The topology builder documented here requires block-owned source cells; the partitioner and migration modules document overloads for unequal contiguous source ranges. Newly generated dense IDs and all connectivity/graph references must be remapped consistently to satisfy the selected entry point. Migrated local ordering cannot simply be treated as the original block distribution.

After rebuilding faces and adjacency, a rebalance can reuse graph partitioning and migration. The present partitioner performs a fresh partition, not incremental repartitioning with migration-cost minimization. Migration reconstructs mesh data and halo maps; transferring solution fields, conservative remapping, and prolongation/restriction require separate operations.

Nonconforming AMR is outside the current contract. Sorted full-face node keys cannot match one coarse face to several fine faces. Supporting such interfaces requires subface topology and coarse/fine relations, extended flux coupling and communication maps, and any constraints required by the numerical method. Refinement trees, parent/child IDs, hanging-node constraints, and adaptation history are absent from the current file schema.

### Preconditions and hardening gaps

These points do not change the matching layout, but limit what the implementation currently guarantees:

1. **Export assumes prior validation.** It checks communicator identity, grouping, mapping validity, and array lengths, but does not call `validate_meshpart` or collective conservation checks. The supplied application performs both beforehand. Other callers must do the same or establish equivalent invariants.
2. **Replicated inputs are a caller contract.** Mapping validity is checked locally; equality of mappings, patch metadata, paths, and global counts across ranks is not checked. Inconsistent inputs can invalidate collective I/O rather than produce a usable snapshot.
3. **Import validation is local.** The reader invokes `validate_meshpart` after loading, but not `collect_mesh_statistics`. It does not perform a remote halo-order/GID handshake or independently prove global uniqueness and count conservation. The application's documented validator does not repeat the geometry stage's outward-side check.
4. **32-bit host offset conversion needs a guard.** In `detail::read`, the datatype-size test for offsets uses an upper bound of 8 bytes even when the destination is a 32-bit `size_t`. Thus it does not reject potentially lossy U64-to-`size_t` conversion before HDF5 reads it. A 32-bit port should enforce host representability using the relevant counts/terminal values or reject unsupported offset conversions explicitly. This is not an issue for U64 offsets read into a 64-bit `size_t`.
5. **Dataset widths are not checked against the full schema.** `detail::read` checks rank, extent, datatype class, integer signedness, and a maximum width, but accepts narrower primitive types. For example, a file declaring FP64 geometry can contain an FP32 geometry dataset that is widened by HDF5. Valid writer output uses consistent widths; strict malformed-file detection should also verify each dataset's width against the header/schema.
6. **Build assumptions are not completely enforced by export.** The reader accepts only 32/64-bit index and geometry widths, while the writer records `sizeof` values without an equivalent explicit supported-type check. Configuration should enforce signed 32/64-bit indices and FP32/FP64 geometry. The definition of `PatchId` is outside this corrected attachment set; this document uses the current I32 patch-ID contract and identifies it separately as `PI` in the schema.

The file schema therefore provides complete mesh-state coverage under its stated contracts. Broader restart, cross-rank integrity checking, precision conversion, and dynamic-mesh state require additional code and, where appropriate, a new schema version.
