# Linear algebra

`krona::linalg` provides distributed square systems `A*x = b` with reusable CSR/BSR sparsity, native Tpetra storage, Belos solvers, and Ifpack2/MueLu preconditioners. It does not discretize equations or apply physical boundary conditions.

## Responsibilities and dependencies

| Part | Responsibility |
| --- | --- |
| `Distribution<Exec>` | Defines which global unknowns each rank owns, preserving the supplied local row order. Wraps a one-to-one Tpetra map. |
| `CsrInput` | Owns host CSR offsets and global column IDs supplied by your application. No coefficients. |
| `CsrPattern<Exec>` | Builds/finalizes the distributed graph and optionally maps original input positions to native coefficient positions. |
| `CsrMatrix<Scalar, Exec>` | Owns scalar coefficients on a shared immutable pattern. |
| `BsrPattern<Exec>` / `BsrMatrix<Scalar, Exec>` | Fixed-size dense blocks on a block-row graph, with an expanded scalar-DOF distribution. |
| `Vector<Scalar, Exec>` | Owns values for locally owned unknowns only. |
| `Solver<Matrix, PrecScalar>` | Configures Belos, prepares preconditioning when needed, solves, and verifies the true residual. |
| `Preconditioner<Matrix, PrecScalar>` | Manages Ifpack2, MueLu, or a custom factory; refreshes after coefficient changes. |
| `mesh_adapter.hpp` / `FaceAssemblyPlan` | Optional host-mesh numbering, face-neighbour sparsity, and face-to-coefficient positions. |
| `field_transfer.hpp` | Optional copying between KRONA field components and owned algebraic DOFs. |

The core uses MPI, Kokkos, and Trilinos. The project requests Trilinos 17.2.1 with Tpetra, Belos, Ifpack2, and MueLu. Use a compatible MPI/compiler/Kokkos build with Tpetra explicit instantiations for the selected execution space and scalar precisions. Native ordinals are always `LocalOrdinal = int` and `GlobalOrdinal = long long`, independently of mesh index widths. Algebraic scalar types are `float` or `double`.

Native coefficient/vector memory follows the selected Tpetra node configuration. Use the returned view types; shared/unified native memory need not be the same type as `Exec::memory_space`.

The mesh adapters additionally use `mesh::host::MeshPart`; field transfers use field views. Distribution, sparsity, matrices, and solvers can be used without a KRONA mesh or fields. Tpetra imports remote vector entries for its matrix operations: algebraic vectors do not need a KRONA halo exchanger or user-maintained ghost tail.

## Ownership, allocation, and copying

Owning `Teuchos::RCP` handles and managed Kokkos views retain shared allocations. Copying such a handle is not a deep copy of its data. A reference, `std::span`, unmanaged view, or `mpi::CommView` does not independently own its backing resource. Native APIs can also return non-owning `RCP`s: BSR `getGraph()` refers to the matrix's internal graph descriptor and must not outlive that native matrix.

| Object or operation | Storage behaviour |
| --- | --- |
| `Distribution` construction | Builds native map/indexing storage from the supplied GIDs. Does not retain the caller's GID array. Copies of `Distribution` share the native map through `RCP`. The Teuchos communicator wrapper borrows the MPI communicator handle; it does not duplicate it. |
| `CsrPattern` construction | Copies inserted CSR indices into a new finalized Tpetra graph. By default allocates the input-to-native permutation in device memory and retains its host counterpart; on HostSpace the two can alias. Caller CSR arrays may be released after construction. Pattern copies share graph/maps/permutation. |
| `CsrMatrix` construction | Copies the pattern handles, shares its native graph, and allocates independent coefficient storage. Two matrices on one pattern share sparsity, not numerical values. |
| `BsrPattern` construction | Shares the block graph and permutation. Expands the owned map to scalar DOFs; block size one shares the original map. Copies of this pattern share the prepared data. |
| `BsrMatrix` construction | Allocates independent block coefficients. Native `BlockCrsMatrix` copies graph/map descriptors, additionally copies CSR row offsets and column indices into private host arrays, and creates a point-column map/importer. Pattern reuse does not eliminate these native per-matrix allocations. |
| `Vector` construction | Shares the distribution map and allocates zero-initialized native values; it does not borrow a field allocation. |
| `FaceAssemblyPlan` | Owns/shared-owns its pattern and three managed entry-map arrays. Uses temporary host numbering/CSR arrays during construction. Does not retain the mesh. Copies share its prepared allocations. |
| Matrix/vector access views | Alias existing native values; they are not snapshots. Native views may retain allocations, but still obey the owner's access/commit contract. BSR block adapters are explicitly unmanaged. |
| Field bindings / `FieldPack` | Copy descriptors only. Transfer calls submit actual value-copy/conversion kernels into separately allocated destination storage. |
| `Solver` / `Preconditioner` | Borrow the KRONA matrix object by `const Matrix&`; the matrix must outlive them. Own backend objects and cached work allocations through `RCP`. Copy options/parameter lists. |

Original distribution/pattern wrapper objects need not outlive matrices or vectors constructed from them: those objects retain shared handles. This differs from the borrowed matrix wrapper used by solvers and preconditioners.

Solver residual storage is allocated lazily on the first nonempty solve and reused. Belos allocates its Krylov workspace. Its linear problem can retain native RHS/solution `RCP`s; solving does not copy those vectors into replacement solution/RHS arrays.

With equal matrix/preconditioner scalar types, preconditioning uses the original native matrix. With different types, it allocates a second coefficient array on the same logical graph and converts coefficients when refreshed. The precision bridge also caches input/output multivectors and converts values on each preconditioner application; an additional solver-precision buffer is needed for the nonzero-`beta` path. This is not zero-copy mixed precision.

Keep native-backed algebraic objects within the active MPI/Kokkos/Tpetra lifetimes; plain host inputs and options can be created before initialization. A borrowed MPI communicator must remain valid while any map/backend object using it remains alive. Matrix/vector wrappers are noncopyable; distribution and pattern copies intentionally share data.

Tpetra manages host/device coherence for native values; these buffers may be separate or share memory depending on the backend. A native `getLocalViewHost(ReadOnly)` aliases native host storage and may trigger a device-to-host copy. The example below instead uses `create_mirror()` to allocate an independent host snapshot, including on HostSpace.

## Distribution and sparsity contracts

Construct `Distribution(owned_gids, comm)` collectively. GIDs must be nonnegative, below `GlobalOrdinal::max()`, and globally unique. Gaps and arbitrary local ordering are allowed. All ranks participate, including empty ranks.

`CsrInput` describes rows in that exact local order:

- `row_offsets` has `local_rows + 1` entries, starts at zero, is nondecreasing, and ends at `column_gids.size()`;
- each column is a global unknown owned somewhere in the distribution;
- columns within a row may be unsorted, but duplicates are rejected;
- an empty rank supplies offsets `{0}` and no columns.

`CsrPattern` finalizes the graph, builds a native column map, and converts columns to local column-map indices. These indices are not global IDs and are not necessarily local owned-row indices. Your original column order can change.

By default, `pattern.input_to_entry()(k)` gives the native flat coefficient position corresponding to original CSR input position `k`. `host_input_to_entry()` exposes the same mapping as a borrowed host span. The device mapping is a managed const view.

Passing `keep_input_mapping = false` skips both mappings. You must then assemble in native entry order; original-order `set_values()` and `set_values_async()` are unavailable. `FaceAssemblyPlan` deliberately uses this mode.

Matrix application and solving require vectors sharing the **same native distribution map object**, not independently constructed equivalent maps. Create vectors from `A.distribution()` or copies of that distribution. For BSR, prepare one `BsrPattern` and reuse its expanded distribution.

## Complete pipeline

This mesh-independent example creates four rows per rank for a tridiagonal matrix with diagonal `4` and off-diagonals `-1`. It fills coefficients in original CSR order, fills the RHS in a Kokkos kernel, solves, and reads the solution on the host. The manufactured solution is `x = 1`.

```cpp
#include <krona/linalg/solver.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/communicator.hpp>
#include <Kokkos_Core.hpp>
#include <Tpetra_Core.hpp>
#include <cstdio>
#include <vector>

namespace la = krona::linalg;
using Exec = krona::execution::ExecutionSpace;
using Scalar = krona::SolverReal;
using GO = la::GlobalOrdinal;

int run(krona::mpi::CommView comm)
{
    return krona::mpi::guarded([&] {
        Exec exec{};
        constexpr std::size_t rows = 4;
        const GO first = static_cast<GO>(comm.rank()) * static_cast<GO>(rows);
        const GO total = static_cast<GO>(comm.size()) * static_cast<GO>(rows);
        std::vector<GO> gids(rows);
        for (std::size_t i = 0; i < rows; ++i)
            gids[i] = first + static_cast<GO>(i);
        la::Distribution<Exec> distribution{gids, comm};

        la::CsrInput input;
        std::vector<Scalar> coefficients;
        input.row_offsets.push_back(0);
        for (const GO gid : gids) {
            // Deliberately diagonal-first: native ordering may differ.
            input.column_gids.push_back(gid);
            coefficients.push_back(Scalar{4});
            if (gid > 0) {
                input.column_gids.push_back(gid - 1);
                coefficients.push_back(Scalar{-1});
            }
            if (gid + 1 < total) {
                input.column_gids.push_back(gid + 1);
                coefficients.push_back(Scalar{-1});
            }
            input.row_offsets.push_back(input.column_gids.size());
        }

        la::CsrPattern<Exec> pattern{distribution, input, exec};
        la::CsrMatrix<Scalar, Exec> A{pattern};
        A.set_values(coefficients); // Blocking copy in original input order.
        la::Vector<Scalar, Exec> b{A.distribution()}, x{A.distribution()};

        {
            const auto rhs = b.overwrite(exec);
            using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
            Kokkos::parallel_for("fill_rhs", Policy(exec, 0, rows),
                KOKKOS_LAMBDA(std::size_t row) {
                    const GO gid = first + static_cast<GO>(row);
                    rhs(row, 0) = Scalar{4}
                        - (gid > 0 ? Scalar{1} : Scalar{0})
                        - (gid + 1 < total ? Scalar{1} : Scalar{0});
                });
        }
        b.commit();
        x.fill(Scalar{0}, exec); // Blocking initial-guess fill.

        la::SolverOptions options;
        options.method = "GMRES";
        options.maximum_iterations = 200;
        options.restart_length = 30;
        options.preconditioner.type = "RELAXATION"; // Jacobi by default.
        la::Solver<decltype(A), Scalar> solver{A, options};
        const auto result = solver.solve(b, x); // Collective and blocking.
        krona::mpi::require(result.converged, "Example linear solve did not converge");

        {
            const auto values = x.read(exec); // Rank-two native vector view.
            auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, values);
            Kokkos::deep_copy(exec, host, values);
            x.commit(); // Also required for read-only batches.
            // host(row, 0) is now available to CPU code and should be close to 1.
            if (comm.rank() == 0)
                std::printf("iterations=%d residual=%e x[0]=%g\n",
                    result.iterations, result.residual_norm,
                    static_cast<double>(host(0, 0)));
        }
        return 0;
    });
}

int main(int argc, char** argv)
{
    krona::mpi::Runtime mpi{argc, argv, krona::mpi::ThreadLevel::funneled};
    return krona::mpi::guarded([&] {
        Kokkos::ScopeGuard kokkos{argc, argv};
        Tpetra::ScopeGuard tpetra{&argc, &argv, krona::mpi::CommView::world().native()};
        return run(krona::mpi::CommView::world());
    }); // Algebraic objects -> Tpetra -> Kokkos -> MPI destruction order.
}
```

The inner guarded computation reports failures before unwinding across runtime guards. If the application already owns MPI/Kokkos initialization, omit the corresponding initialization guards shown here; Tpetra uses those initialized runtimes. Construct all algebraic objects after Tpetra initialization and release them before its guard exits. Complete custom work before releasing its buffers.

## Obtain data for assembly

Choose the assembly path matching your coefficient layout:

| Path | API and ordering |
| --- | --- |
| Host coefficients in original CSR order | `A.set_values(std::span<const Scalar>)`. Copies values; caller storage is not retained. |
| Accessible device coefficients in original CSR order | `A.set_values_async(input, exec)`, then `A.commit()`. CSR input is rank one with exactly `pattern.local_nonzeros()` entries. Conversion to matrix scalar is performed. Keep input valid until completion. |
| Direct native-order assembly | `A.edit_values(exec)` or `A.overwrite_values(exec)` returns a rank-one coefficient view. Original position `k` maps through `pattern.input_to_entry()(k)`. |
| Mesh-face assembly | Use `FaceAssemblyPlan::entry_map()` to address native off-diagonal and diagonal positions directly. |

For direct CSR row traversal, obtain the immutable native graph and local map descriptors. The following reproduces the tridiagonal coefficients above; it does not depend on original input ordering:

```cpp
{
    const auto graph = A.pattern().native()->getLocalGraphDevice();
    const auto row_ids = A.distribution().native()->getLocalMap();
    const auto column_ids = A.pattern().native()->getColMap()->getLocalMap();
    const auto values = A.overwrite_values(exec);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("fill_native_csr",
        Policy(exec, 0, A.distribution().local_size()), KOKKOS_LAMBDA(std::size_t row) {
            const GO row_gid = row_ids.getGlobalElement(static_cast<la::LocalOrdinal>(row));
            for (auto k = graph.row_map(row); k < graph.row_map(row + 1); ++k) {
                const GO column_gid = column_ids.getGlobalElement(graph.entries(k));
                values(k) = column_gid == row_gid ? Scalar{4} : Scalar{-1};
            }
        });
}
A.commit();
```

`edit_values()` preserves old values, synchronizing native host/device storage if necessary. CSR/vector `overwrite*()` declares a complete overwrite and avoids copying old host contents to device. It does not zero the array: assign every entry or explicitly clear before additive assembly. BSR overwrite still uses native ReadWrite access in Trilinos 17.2.1 and may synchronize old values.

Changing coefficients does not require `resumeFill()` or a new graph. Each matrix edit/overwrite request advances `revision()`, including `set_values*()`; reads do not. Reuse the solver after coefficient updates. The next preconditioner preparation detects changes collectively, even if only one rank edited its matrix.

`native()` exposes an `RCP` for advanced interoperability, bypassing wrapper checks. Synchronize native edits yourself, then call `A.mark_modified()` so revision-aware preconditioning sees them. Do not change native graph/maps behind an existing pattern or retain stale local views across native storage changes.

Distribution/pattern/matrix/vector `native()` accessors return a reference to their retained handle. Copy that `RCP` by value to retain the native object independently; retaining a reference to the wrapper's handle does not extend its lifetime. This does not relax the solver/preconditioner's separate requirement that the KRONA matrix wrapper outlive them.

## Access batches and synchronization

`Vector::edit/read/overwrite` return rank-two native views accessed as `values(local_row, 0)`. Matrix `edit_values/read_values/overwrite_values` return flat coefficient views. Every such call opens or joins a pending access batch on the supplied `Exec`, including read-only access.

1. Acquire access on an execution-space instance.
2. Submit all related kernels/copies on that instance.
3. Keep access views scoped to that batch rather than retaining them for later submissions.
4. Call the accessed object's `commit()` before native matrix operations, solving, host assembly, or switching instances.

`commit()` fences the recorded instance and clears the pending flag. If no access is pending, it is a no-op. **An external fence does not clear the flag.** `require_committed()` checks it; it does not wait automatically. Commit every accessed object, including read sources. Do not launch another batch through an old access view after committing; acquire access again so tracking and matrix revision remain correct.

Access acquisition can itself trigger native host/device synchronization. The contract is that the subsequent custom kernels are asynchronous until committed, not that obtaining a view is guaranteed nonblocking.

## Fixed-size block systems

`BsrPattern(block_graph, B)` expands block GID `g` to point GIDs `g*B + c`, for `c = 0..B-1`. Local vector order is block/cell first, then component. All ranks must use the same positive block size; expansion validates local/global bounds. `block_distribution()` indexes blocks; `distribution()` indexes scalar DOFs.

Starting from the original-order `pattern` and `coefficients` in the complete example, this creates two uncoupled copies of that scalar operator using 2-by-2 blocks:

```cpp
la::BsrPattern<Exec> block_pattern{pattern, 2};
la::BsrMatrix<Scalar, Exec> block_A{block_pattern};
la::Vector<Scalar, Exec> block_b{block_A.distribution()}, block_x{block_A.distribution()};
std::vector<Scalar> blocks(coefficients.size()*4, Scalar{0});
for (std::size_t k = 0; k < coefficients.size(); ++k) {
    blocks[4*k] = coefficients[k];     // Equation 0, variable 0.
    blocks[4*k + 3] = coefficients[k]; // Equation 1, variable 1.
}
block_A.set_values(blocks);
```

Host input and rank-one `set_values_async()` input use original graph-entry order followed by row-major blocks: `(entry*B + equation)*B + variable`. Device input may instead be rank three with shape `(block_nonzeros, B, B)`.

For direct assembly, `edit_blocks()`, `overwrite_blocks()`, and `read_blocks()` expose `(native_entry, equation, variable)` views. They adapt native dense-block layout; do not assume the flat coefficient view is row-major. An entry-map position indexes a whole block, not its first scalar coefficient. Commit the matrix after device access. Solving uses the same `Solver` API with the expanded-distribution vectors.

## Mesh numbering and face assembly

`build_cell_numbering(mesh, comm)` collectively creates rank-contiguous algebraic cell GIDs in existing owned-cell order. It exchanges these IDs to every mesh ghost on a private MPI communicator. It does not use original mesh GIDs, geometry, or device storage. The mesh/communicator and halo plan must agree, including complete unique receive coverage of ghost cells.

The returned `CellNumbering` owns `cell_to_global` for owned plus ghost cells; `owned_gids()` borrows its owned prefix. `expanded_gids(B)` returns a new host vector for all local cells and components. `make_face_csr()` locally builds owned rows containing the diagonal and sorted unique face neighbours. Physical boundary slots are excluded; multiple connecting faces become one graph entry.

`FaceAssemblyPlan` combines this preparation with native entry lookup. It requires grouped face intervals, performs collective/blocking setup, and does not retain the host mesh. Include `<krona/linalg/face_assembly.hpp>` and use it instead of manually building distribution/CSR when assembling a face-neighbour operator:

```cpp
la::FaceAssemblyPlan<Exec> assembly{host_mesh, comm, exec};
la::CsrMatrix<Scalar, Exec> face_A{assembly.pattern()};
const auto entries = assembly.entry_map();
// entries.diagonal(cell): owned-cell diagonal entry.
// entries.owner_entries(face): owner row -> neighbour column, for face < n_inner_faces.
// entries.neighbour_entries(face): neighbour row -> owner column,
//                                  for face < n_internal_faces (owned/owned faces).
```

The entry map contains managed const views. The same plan serves CSR and BSR matrices; BSR positions identify blocks. Its pattern has no original-input permutation, so assemble with native access/entry maps rather than `set_values*()`.

External boundary faces have no off-diagonal map entry; their discretization usually contributes to the owned diagonal and RHS. The plan does not calculate those coefficients or boundary terms. If parallel face kernels share matrix/RHS destinations, including multiple faces connecting the same pair of cells, use atomic accumulation or a row-owned assembly strategy. Clear coefficients/RHS before additive assembly. Rebuild the plan after topology, halo, or local numbering changes.

## Copy fields to and from vectors

These helpers submit copy/conversion kernels; fields and algebraic vectors remain separate allocations. They respect component strides and write only owned prefixes, leaving field ghost/boundary tails unchanged.

| API | Mapping |
| --- | --- |
| `vector.copy_from_component(view, exec)` / `copy_to_component(view, exec)` | One rank-one component, indexed in vector-local row order. |
| `copy_from_field(vector, field, exec)` / `copy_to_field(vector, field, exec)` | Entire `FieldView`: local DOF `cell*N + component`. |
| `copy_from_fields(vector, field_pack(...), exec)` / `copy_to_fields(...)` | Concatenates supplied fields/components within each owned cell. Accepts `FieldView`s and individual rank-one component views. |

Include `<krona/linalg/field_transfer.hpp>`. For density, three momentum components, and energy, a five-DOF vector must use `[rho, mx, my, mz, E]` within each cell:

```cpp
// rho, momentum and energy are FieldViews; vector uses a five-DOF-per-cell distribution.
auto pack = la::field_pack(rho, momentum, energy);
la::copy_from_fields(vector, pack, exec);
vector.commit();
// ... matrix operations / solve ...
la::copy_to_fields(vector, pack, exec);
vector.commit(); // Completes the read batch and destination-field writes.
```

The alternative call form is `copy_from_fields(vector, exec, rho, momentum, energy)`. Total component width and actual cell order must match the intended algebraic DOFs; divisibility checks cannot establish that semantic agreement. Fields must be long enough, accessible, and writable when unpacking. Conversions use the destination scalar type. Sources/destinations must not alias the native vector; unpack destinations must not overlap each other. Complete other producers before transfer, and keep field buffers valid until commit.

## Solver and preconditioner controls

| `SolverOptions` member | Default / meaning |
| --- | --- |
| `method` | `"GMRES"`; native Belos method name, normalized to uppercase. `"FGMRES"` aliases `"FLEXIBLE GMRES"`. |
| `relative_tolerance` | Unset: `1e-5` for float, `1e-10` for double, using the actual matrix/solver scalar. |
| `absolute_tolerance` | `0`. At least one tolerance must be positive; relative tolerance must be below one. |
| `maximum_iterations` | `500`; zero checks the initial residual without starting iteration. |
| `restart_length` | `50`; supplies `Num Blocks` for applicable managers unless explicitly provided. |
| `side` | `Automatic`: left for CG-family names ending in `CG` and for `MINRES`, right otherwise. Enabled BICGSTAB/flexible GMRES preconditioning requires right side. |
| `preconditioner.type` | `"AMG"`; empty/`"NONE"` disables it, `"AMG"`/`"MUELU"` selects MueLu, otherwise an Ifpack2 name. |
| `solver_parameters`, `preconditioner.parameters` | Native Teuchos parameter lists. Common stopping controls are enforced by KRONA during solve. Explicit `Num Blocks`/`Maximum Restarts` override their automatic defaults. |

The application must choose an algorithm/preconditioner appropriate to the matrix. Selecting CG/MINRES declares symmetry to Belos; it does not verify matrix symmetry or definiteness.

`solve(b, x)` uses the current `x` as its initial guess. Both vectors and the matrix must be committed, use the matrix's native map object, and have distinct native RHS/solution allocations. The call is collective and blocking.

Convergence is determined by an independently recomputed true residual:

$$
\|b-Ax\|_2\le\max(\mathrm{absolute\_tolerance},
                         \mathrm{relative\_tolerance}\,R),
$$

where `R = ||b||_2` for nonzero RHS and `R = ||b-A*x_initial||_2` otherwise. `relative_residual` uses that same reference. `SolveResult::converged` is the authoritative result; `krylov_converged` reports Belos's own status. The result also contains iteration count, whether iteration started, RHS norm, initial/final residual norms, and threshold. Nonconvergence is returned for the caller to handle, not automatically fatal. An accepted initial guess or globally empty system skips iteration; the preconditioner may remain unprepared.

`solver.prepare()` optionally warms up preconditioning. `solve()` also prepares when iteration is needed. Any-rank coefficient revision changes trigger refresh on all ranks. Ifpack2 initializes its structure once and recomputes numerical data after updates; `RELAXATION` defaults to Jacobi. MueLu defaults to rebuilding (`reuse: type = "none"`); supply a supported reuse policy to reuse a hierarchy. `preconditioner().invalidate()` requests another refresh; `generation()` counts completed enabled refreshes.

For BSR on Trilinos 17.2.1, `SCHWARZ` requires zero overlap and defaults its inner solver to `RBILUK`. Default block AMG uses an unsmoothed, block-preserving hierarchy, Jacobi relaxation, and `use kokkos refactor = false`; `number of equations` must match the block size. CSR AMG defaults to smoothed aggregation and a Chebyshev smoother. Native parameter choices and available coarse solvers depend on the installed Trilinos configuration.

A custom preconditioner factory receives `(RCP<const native_matrix_in_PrecScalar>, previous_operator)` and returns a nonnull compatible native operator. It is called initially and after collective refresh decisions; it may update/reuse the previous object. Domain/range maps must match the matrix distribution; local subdomain solvers need a compatible wrapper such as SCHWARZ. Solver and preconditioner objects cache mutable workspace and are not reentrant.

## Completion and lifetime rules

| Operation | Completion behaviour |
| --- | --- |
| Distribution/expanded distribution, patterns, mesh numbering, face-assembly preparation | Collective setup. Pattern preparation and face-map uploads finish before return. |
| CSR/BSR matrix construction | Native setup/allocation and a process-wide Kokkos fence before return. |
| Vector construction | Native allocation and zero initialization; no explicit KRONA fence or caller-selected `Exec` parameter. |
| Host `set_values()` | Completes host assembly before return. Native device storage may synchronize later when accessed. Requires no pending access. |
| Access acquisition, `set_values_async()`, `fill_async()`, field/component transfer helpers | Open/join an access batch. Custom copies/kernels have no final fence. Commit the accessed matrix/vector. Some helper names have no `_async` suffix. |
| Matrix/vector `commit()` | Local fence on the pending execution instance and state reset; no MPI barrier. Destructors also commit pending accesses. |
| Vector `fill()` | Calls `fill_async()` followed by `commit()`. |
| Matrix `apply()` | Collective, blocking native multiplication/imports and a process-wide Kokkos fence. Requires all operands committed; in-place `x == y` is rejected. |
| `Preconditioner::prepare()` / `Solver::prepare()` | Treat as collective blocking preparation; enabled refreshes fence before return. Clean/disabled paths can return without new device work. |
| `Solver::solve()` | Collective blocking solve with host-visible residuals. Iterative path ends with a process-wide Kokkos fence; early exits rely on completed native norm/residual operations. |

Process-wide Kokkos fences can wait beyond one stream; instance `commit()` waits only for its recorded instance. A fence does not synchronize other MPI ranks. Queue dependent custom work on the same instance, or explicitly synchronize producers on other instances. Native access through `native()` does not open a tracked batch: the caller must enforce completion and coefficient revision tracking. Explicitly commit before releasing external transfer buffers; a later matrix/vector destructor cannot protect a buffer already destroyed.

Backend ownership details above were checked against the Trilinos 17.2.1 implementations of [Map](https://github.com/trilinos/Trilinos/blob/trilinos-release-17-2-1/packages/tpetra/core/src/Tpetra_Map_def.hpp), [CrsGraph](https://github.com/trilinos/Trilinos/blob/trilinos-release-17-2-1/packages/tpetra/core/src/Tpetra_CrsGraph_decl.hpp), and [BlockCrsMatrix](https://github.com/trilinos/Trilinos/blob/trilinos-release-17-2-1/packages/tpetra/core/src/Tpetra_BlockCrsMatrix_def.hpp).
