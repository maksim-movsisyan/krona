# Mesh Partitioner

The mesh partitioner converts a CGNS mesh into a distributed KRONA solver mesh: one partition per MPI rank, local connectivity, a one-layer face-neighbour halo and geometric metrics. The result is saved in KRONA HDF5 format.

The application is `krona_mesh_partitioner`. Its workflow uses `io::cgns`, `mesh::host`, `mesh::partition` and `io::solver_mesh`; the graph partitioner itself only needs CSR connectivity and vertex ownership.

## Processing steps

| Step / API | Input → output | How it works |
| --- | --- | --- |
| 1. `io::cgns::read_parallel` | CGNS file → `ReadResult`, containing `host::RawMesh` | Reads one unstructured zone. Cells, coordinates and boundary elements are distributed in contiguous blocks; small metadata is replicated. Planar 2D input is extruded inside the reader, so the returned volume mesh is 3D. |
| 2. `host::build_faces` | `RawMesh` → unique `FaceRecord` objects, dual CSR graph and `FaceStats` | Generates cell facets, sorts node IDs into matching keys and sends matching keys to the same rank. Matches cell neighbours and boundary labels, then builds adjacency during the same distributed pass. Each graph vertex is a cell; an edge connects cells sharing a face. |
| 3. `partition::partition_graph` | CSR graph + current vertex ownership → `PartitionResult` | Imports the graph into dKaMinPar, partitions it into `comm.size()` unweighted blocks, maps blocks to destination ranks and counts planned loads. This step computes destinations without moving mesh data. |
| 4. `host::migrate_local_mesh` | Consumed source mesh and faces + destination ranks → `MeshPart` | Exchanges cells and faces, obtains remote neighbours and coordinates, converts connectivity to local indices and builds halo send/receive maps. Geometry arrays are still absent; faces are not yet grouped. |
| 5. `host::compute_mesh_geometry` | `MeshPart` → populated metrics + `GeometryStats` | Computes cell volumes/centroids and face areas/centroids/outward normals with double intermediates. The communicator overload also reduces statistics and refreshes global bounds. |
| 6. `host::reorder_local_mesh` | `MeshPart` + ordering method → updated `MeshPart` | Reorders owned cells if requested, groups faces into owned–owned / owned–ghost / boundary ranges and updates connectivity, halo indices, patches and existing geometry together. |
| 7. Validation and output | Prepared `MeshPart` → statistics and HDF5 file | `validate_meshpart` checks local invariants; `collect_mesh_statistics` checks global conservation. The writer saves each partition, geometry, halo maps and patch metadata. BC templates and VTK output are optional. |

During face construction, a unique face record is held by the source owner of its smaller cell ID. Node sorting is used only for matching; oriented connectivity is reconstructed from the cell and its local facet index. `BuildFacesOptions::batch_bytes` defaults to 64 MiB per rank per round as an average payload target, not a hard receive-memory limit.

Steps involving MPI must be called by every rank, including ranks with no cells, in the same order. Geometry computation with a communicator is collective; reordering and local validation do not exchange mesh data.

## Input and output contracts

- Volume cells are linear TET, PYRA, PRISM or HEXA; boundary facets are TRI or QUAD. Connectivity follows canonical CGNS node ordering.
- The CGNS reader accepts HDF5 unstructured zones with supported uniform or MIXED sections. Higher-order and polyhedral elements are unsupported. Legacy variable-arity sections needing implicit conversion must be converted to a suitable CGNS format first.
- Reader defaults select base 1 and zone 1. All ranks use the same file and options. CGNS calls must not overlap because CGNS configuration is process-global.
- In 3D, supported BC point sets identify boundary elements: `PointList` / `PointRange` at `FaceCenter`, or legacy element lists/ranges. Unsupported BC sets are rejected by default; `UnsupportedBoundary::ignore` records their names in `ReadResult::source.ignored_boundary_conditions`.
- External faces without a label remain boundary faces with `invalid_patch_id`. `build_faces` does not create an automatic patch. Duplicate boundary records, boundary labels on interior faces and faces shared by more than two cells are rejected. Hanging interfaces require a conforming mesh first.
- The HDF5 result is a partitioned mesh snapshot. Loading requires the saved MPI size and communicator rank numbering, matching `GeometryReal` precision and equally wide or wider local/global index types. Loading with a different MPI size requires repartitioning.

## Prepare and save a mesh in C++

This is the application's processing order, with explicit ownership transfer. No Kokkos initialization is needed.

```cpp
#include <krona/io/cgns/reader.hpp>
#include <krona/io/solver_mesh/hdf5_writer.hpp>
#include <krona/mesh/host/faces.hpp>
#include <krona/mesh/host/migrate.hpp>
#include <krona/mesh/host/geometry.hpp>
#include <krona/mesh/host/reorder.hpp>
#include <krona/mesh/host/validate.hpp>
#include <krona/mesh/partition/partition.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <utility>
#include <vector>

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv};
    return krona::mpi::guarded([&] {
        namespace host = krona::mesh::host;
        namespace partition = krona::mesh::partition;
        const auto comm = krona::mpi::CommView::world();

        auto imported = krona::io::cgns::read_parallel("mesh.cgns", comm);
        auto topology = host::build_faces(imported.mesh, comm);
        auto plan = partition::partition_graph(
            topology.graph.view(), imported.mesh.distribution.cells, comm);

        topology.graph = {}; // Graph storage is no longer needed.
        auto mesh = host::migrate_local_mesh(
            std::move(imported.mesh), std::move(topology.faces),
            plan.vertex_target_rank, comm);
        std::vector<int>{}.swap(plan.vertex_target_rank);

        (void)host::compute_mesh_geometry(mesh, comm);
        host::reorder_local_mesh(mesh, host::ReorderMethod::NONE);
        host::validate_meshpart(mesh);
        const auto statistics = host::collect_mesh_statistics(mesh, comm);
        (void)statistics;

        krona::io::solver_mesh::export_mesh_hdf5(
            mesh, "mesh.h5", comm, plan.part2rank);
        return 0;
    });
}
```

`std::move` marks the imported payload and face records as consumed. The destination-rank span is borrowed only for the migration call. Release the dual graph before migration to reduce peak memory.

## Partition an arbitrary graph

Header: `krona/mesh/partition/partition.hpp`. Namespace: `krona::mesh::partition`.

`partition_graph` accepts `graph::CsrGraphView`, which borrows two arrays:

| Array | Meaning |
| --- | --- |
| `offsets` (`size_t`) | `n_local + 1` nondecreasing offsets. Starts at zero and ends at `adjacency.size()`. |
| `adjacency` (`GlobalIndex`) | Zero-based **global vertex IDs** of neighbours. |
| `vertex_distribution` (`GlobalIndex`, separate argument) | Replicated `P + 1` ownership endpoints. Rank `r` owns IDs in `[distribution[r], distribution[r+1])`. Starts at zero. |

Row `i` on rank `r` represents vertex `distribution[r] + i`. The graph must be symmetric, without self-loops or duplicate neighbours. Each undirected edge appears in both rows. An empty rank supplies `offsets = {0}` and empty adjacency. Full graph correctness is a caller contract; release builds do not scan every edge for validity.

Use `graph::CsrGraph` when ownership of the arrays is useful; call `.view()` to borrow them. For arrays owned elsewhere, construct `CsrGraphView{offsets, adjacency}` directly. Keep both arrays alive and unchanged until partitioning returns.

The second overload accepts `distribution::BlockDistribution<GlobalIndex>` instead of endpoints. Use it only when the current vertex IDs actually follow that balanced distribution. After adaptation or migration, assign contiguous per-rank IDs and remap adjacency before using either overload.

### Options

| `DKaMinParOptions` field | Default | Meaning |
| --- | --- | --- |
| `imbalance_tolerance` | `0.03` | Relative load imbalance parameter passed to dKaMinPar: `0.03` requests 3% imbalance. Must be finite and strictly positive. All vertices have equal weight. |
| `threads_per_rank` | `1` | Backend worker count; at least 1. Values above 1 require `MPI_THREAD_FUNNELED` or stronger. |
| `seed` | `0` | Seed passed to dKaMinPar. It does not guarantee identical results across different process/thread configurations. |
| `quiet` | `true` | Suppresses backend progress output. |
| `group_by_shared_memory` | `true` | Places consecutive blocks on ranks grouped by shared-memory domain. This is a placement heuristic, not CPU binding or measured inter-node cut optimization. |

All ranks use identical options and ownership metadata. Calls run on the MPI main thread; do not overlap dKaMinPar calls within a process. Input dimensions must fit the backend's configured index and weight widths. A single rank or globally empty graph bypasses the backend solve.

### Result

| `PartitionResult` field | Meaning |
| --- | --- |
| `vertex_target_rank[i]` | Destination rank for local input row `i`. |
| `part2rank[b]` | Destination rank assigned to backend block `b`. |
| `rank2part[r]` | Inverse mapping: the block assigned to rank `r`. |
| `vertices_per_rank[r]` | Replicated planned vertex count on rank `r` after migration. |
| `global_edge_cut` | Global count of undirected edges crossing blocks. For a mesh dual graph, these correspond to cut cell-neighbour pairs. |
| `shared_memory_groups` | Discovered domain count; zero if grouping was disabled. |

Partitioning leaves input arrays and current ownership unchanged. The wrapper may convert index arrays and dKaMinPar copies its imported graph; accepting a view does not make the backend import zero-copy.

## Application choices

The current application fixes partition settings to the defaults above; they are not CLI options. It accepts an input CGNS path and an output HDF5 path, plus:

| Option | Effect |
| --- | --- |
| `--reorder NONE` | Default. Preserves owned-cell order but still groups/sorts faces. |
| `--reorder SFC` | Orders owned centroids along a local 3D Hilbert curve. C++ enum: `HILBERT_SFC`. |
| `--reorder RCM` | Uses Reverse Cuthill–McKee on owned-cell face adjacency. |
| `--bc <path>` | Writes an editable TOML BC template on rank 0. Source labels provide suggestions; physical conditions still need review. |
| `--vtu <directory>` | Writes owned volume cells and boundary faces as `part.pvtu` and `part_bnd.pvtu`, with per-rank pieces. |
| `-v` / `--verbose`, `-vv` | Summary / detailed application diagnostics. Backend output remains quiet. |

For mesh layout, auxiliary caches and solver-side loading, see [MESH-HOST](MESH-HOST.md). For Kokkos access, see [MESH-DEVICE](MESH-DEVICE.md).
