# Distributed linear algebra

`krona::linalg` owns Tpetra distributions, fixed CSR graphs, matrices and vectors, and provides restarted Belos GMRES with optional Ifpack2 right preconditioning. The core accepts algebraic GIDs and CSR input, independently of a mesh or equation.

## Types and initialization

- `LocalOrdinal=int`, `GlobalOrdinal=long long` match the installed Tpetra explicit instantiations. They are independent of `LocalIndex` and `GlobalIndex`. Local row/column counts must fit `int`; CSR offsets and entry counts use `size_t`. GIDs are nonnegative and smaller than the Tpetra invalid ordinal sentinel.
- Scalars are `float` or `double`; the default is `SolverReal`. `Node` is explicitly tied to `Exec`, never to Tpetra's default backend. The installed library must instantiate the chosen scalar/node/ordinal combination.
- Initialize MPI, Kokkos and Tpetra before constructing objects. Destroy all algebra objects before Tpetra/Kokkos/MPI finalization. `Tpetra::ScopeGuard` can follow existing MPI and Kokkos guards; it does not take ownership of their already initialized runtimes.
- Distribution construction, CSR preparation, matrix construction, preconditioner preparation, matrix apply and solve are collective over the supplied communicator. Empty ranks participate. Numbering exchanges are collective on that communicator as well.
- Communicators are borrowed and must outlive their distributions and all derived objects. Only intracommunicators are supported. Host API calls follow the MPI FUNNELED/main-thread contract; objects are not concurrently mutable.
- All ranks use the same solver options and invoke collective operations in the same order. Local coefficient edits may differ between ranks: one dirty-state reduction makes numeric preconditioner refresh consistent when any rank changes its matrix.
- Local/library failures use `mpi::fatal`; ordinary iteration-limit nonconvergence returns a result. A fatal error follows the project's MPI_COMM_WORLD abort policy.

## Numbering and unknowns

`Distribution` preserves the local order of its owned GID list. Owned GIDs must be globally unique; the same distribution object or a copy must be used for associated matrices and vectors. The first version supports square systems with identical row, domain and range maps.

`build_cell_numbering` assigns `owned_begin+local_owned_cell` GIDs using `MPI_Exscan`. It exchanges GIDs using the existing host halo plan, then scatters received values through `recv_ghost_local`. Consequently `cell_to_global[local_cell]` remains correct even when receive indices are permuted. It uses no replicated rank-offset table, preserves no dependency on original file GIDs, and uses a temporary duplicated communicator for its private exchange. Peer counts/order must match the mesh halo contract; every ghost must have exactly one receive entry.

The mesh rank, process count and halo peer ranks must refer to the supplied communicator, not to a different parent communicator.

`expanded_gids(N)` produces `N*cell_gid+component` in cell-major local order, including ghosts. Pass only the owned prefix to `Distribution`. These GIDs describe the chosen algebraic numbering, not physical mesh IDs.

Physical boundary entries in `CellExtnd` and MPI ghost copies are **not** additional owned unknowns. Boundary-condition contributions must be assembled into coefficients/RHS by the discretization. Tpetra builds its own column map and imports remote vector entries; KRONA halo indices are not Tpetra local column IDs.

`make_face_csr` is an optional scalar graph builder: each owned diagonal and all real face neighbours are included. Repeated connections are merged, physical boundary slots are excluded. This does not implement a PDE operator or BC elimination. Other stencils can supply their own `CsrInput`.

## CSR ownership and assembly

- `CsrInput` is host setup data: `rows+1` monotone offsets, first zero, final equal to the number of column GIDs. Columns within a row may be unsorted but must not repeat. Every column must belong to the domain distribution.
- `CsrPattern` finalizes the native graph once. Its copies and all matrices derived from it share graph storage. Each matrix owns independent coefficients, initialized to zero.
- By default a permutation preserves the original CSR input coefficient order despite native column renumbering/sorting. `set_values(span)` is blocking host assembly. `set_values_async(view,exec)` queues device assembly in the same input order and requires `commit()` afterward.
- `edit_values(exec)` returns the native coefficient array and begins a write batch. Native row offsets/column indices are available from `pattern.native()->getLocalGraphDevice()`. Use `input_to_entry()` to address original input slots when that mapping is enabled. Parallel assembly collisions require atomics or a race-free organization chosen by the caller.
- Pass `keep_input_mapping=false` to the pattern constructor when assembling directly in native order. This omits permutation storage and disables input-order setters. Pattern copies are shallow. The permutation otherwise uses one `size_t` per entry in each retained memory space; on HostSpace the host/device views alias.
- Application-owned CSR input can be discarded after pattern preparation. Tpetra manages its own graph, host/device views and communication metadata; this wrapper does not guarantee absence of native host mirrors.
- Rebuild distributions/graphs after ownership, numbering or sparsity changes. With fixed sparsity, update only coefficients. Every wrapper edit increments the matrix revision, so the solver recomputes numeric preconditioning as necessary while retaining symbolic preparation.

## Fields, ordering and precision

`Vector` stores only locally owned unknowns. `copy_from_component` / `copy_to_component` respect rank-one strides and use the owned prefix. Full-field helpers pack/unpack cell-major DOFs, independently of LayoutLeft/Right. Field tails remain unchanged. They require the vector's local row order to match the field's owned-cell/component order; arbitrary GID lists do not establish this correspondence automatically.

Field storage and solver scalar types may differ; transfers convert explicitly. Ifpack2 currently uses the **same scalar as the matrix/solver**. `PreconditionerReal` does not select a separate lower-precision preconditioner in this version.

## Execution and lifetime

Kokkos edits and field transfers accept an `Exec` instance and enqueue work without a final fence. After a vector/matrix write batch, `commit()` fences that instance before native Trilinos operations. Starting another edit while the batch is pending is an error. Reading pending vectors or entering apply/solve with pending inputs is an error.

Native Tpetra operations may use their own default execution instance. Wrapper `apply`, native `fill`, `prepare` and `solve` are blocking boundaries; `solve` inserts no wrapper fence per Krylov iteration. Constructors finish native preparation before returning. Do not assume that arbitrary custom streams and Tpetra's streams are interchangeable.

Outgoing asynchronous copies require the caller to synchronize before modifying/destroying the vector or accessing the destination from another instance. All referenced buffers must outlive queued work. Destructors do not fence. Device inputs to coefficient assembly must remain alive until commit. Host inputs to `set_values` may be released on return.

Native access is an advanced escape hatch and bypasses state/revision tracking. Finish native coefficient edits and call `mark_modified()` before using a cached solver. Do not change a finalized graph through native access.

## Solver result and scope

`Solver` implements restarted GMRES with a supplied initial guess. Default right preconditioning is Ifpack2 RELAXATION/Jacobi; an empty preconditioner name disables it. Other Ifpack2 names/parameters are forwarded, subject to their installed backend capabilities. The matrix must outlive the solver; options remain fixed for its lifetime.

The default relative tolerance follows the configured `SolverReal`: `1e-5` for float and `1e-8` for double. Set it explicitly when selecting a different solver scalar or application accuracy requirement.

GMRES basis memory grows with `local_DOF_count * restart_length`; the default restart length is 50. This storage is native Belos workspace, not an additional copy of the field manager.

`SolveResult.converged` checks the independently recomputed, unpreconditioned residual:

`||b-A*x||_2 <= max(absolute_tolerance, relative_tolerance*||b||_2)`.

`krylov_converged` separately records Belos' stopping status; its reported stopping criterion alone is not accepted as proof. Results include iterations, RHS norm and true residual. With zero RHS, use a positive absolute tolerance unless exact zero is required; relative residual is zero for an exact zero residual and infinity otherwise. An already acceptable initial guess returns zero iterations. Globally empty systems are valid.

Belos handles the initial-residual check. The wrapper adds one final matrix apply and residual norm reduction to verify its result, and obtains the RHS norm for the requested stopping criterion. These checks can be material for very short solves; there is no fixed-percentage overhead guarantee.

This first version does not wrap BSR, custom LDU operators, MueLu hierarchies, rectangular systems, mixed-precision preconditioners or multiple RHS. These can extend the same maps/operator interface without modifying mesh and fields. There is no measured overhead guarantee relative to raw Tpetra or hand-written kernels.

Tests check independent manufactured matrix products/solutions, arbitrary GIDs, permuted mesh ghost IDs, remote columns, separate communicators, empty ranks/systems, float/double, field layouts/strides/tails, native and input-order assembly, shared graphs, local/global coefficient updates, zero RHS, true residual and nonconvergence. Expected failures require both a nonzero exit and the intended diagnostic; timeouts do not pass.
