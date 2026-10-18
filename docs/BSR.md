# CSR, BSR and coupled fields

`CsrMatrix` stores scalar coefficients. `BsrMatrix` stores fixed-size, dense,
square blocks using `Tpetra::BlockCrsMatrix`. Both use the same solver and
preconditioner implementation. All ranks must use the same block size.

## Graphs and distributions

| Object | Local order / meaning |
| --- | --- |
| `CsrPattern` used as a block graph | One row per owned cell; one entry per coupled cell pair |
| `BsrPattern::block_distribution()` | Owned block/cell GIDs |
| `BsrPattern::distribution()` | Owned scalar DOFs, ordered by cell and then component |
| `BsrPattern::local_nonzeros()` | Number of stored dense blocks |
| `BsrPattern::local_values()` | Number of scalar coefficients: blocks × block size² |
| `Vector` | Scalar DOFs only; no mesh ghosts or physical boundary slots |

`Distribution::expanded(B)` preserves local cell order and produces
`point_gid = cell_gid*B + component`. It checks local/global index limits and
requires the same positive `B` on every rank. `B == 1` shares the original map.

```cpp
#include <krona/linalg/solver.hpp>
#include <krona/linalg/field_transfer.hpp>

using namespace krona;
using namespace krona::linalg;
using Exec = execution::ExecutionSpace;

// owned_cell_gids and connectivity describe owned cells in the same local order
// as the mesh fields. connectivity contains cell GIDs, not expanded DOF GIDs.
Distribution<Exec> cells(owned_cell_gids, comm);
CsrPattern<Exec> graph(cells, connectivity);
BsrPattern<Exec> pattern(graph, 5);
BsrMatrix<SolverReal, Exec> matrix(pattern);
Vector<SolverReal, Exec> rhs(pattern.distribution());
Vector<SolverReal, Exec> solution(pattern.distribution());
```

Copies of `BsrPattern` share the graph, input permutation and point map.
Construct several matrices from one pattern when they should use the same
vectors. `BsrMatrix(graph, B)` is a convenience constructor that creates its
own block pattern. As with CSR, vectors supplied to a matrix/solver must share
that matrix's `Distribution` object, including its local row order.

Graph/pattern construction is collective and blocking. The communicator must
outlive its users. The block graph need not be a face-neighbour graph; the
existing mesh adapter is one possible graph source.

## Coefficient assembly

`set_values(host_span)` takes blocks in the original graph input order. Each
block is flattened in row-major order: `(entry*B + equation)*B + variable`.
The operation is blocking. `set_values_async(input, exec)` accepts either that
flat rank-one format or a rank-three Kokkos View `(entry, equation, variable)`;
the input View's layout is respected.

For direct device assembly:

```cpp
Exec exec;
auto blocks = matrix.overwrite_blocks(exec);
auto input_to_entry = pattern.input_to_entry();

// Assembly on exec writes blocks(native_entry, equation, variable).
// input_to_entry(original_entry) translates original graph slots to native slots.
// Accumulating assembly must initialize every coefficient, including zeros.

matrix.commit();
```

`edit_blocks`, `overwrite_blocks` and `read_blocks` return unmanaged rank-three
Views in native graph entry order. Their strides follow Tpetra's dense-block
layout. Prefer these accessors over assuming a flat native block layout.
`edit_values`, `overwrite_values` and `read_values` expose flat native storage.
Direct native coefficient edits require `mark_modified()` after completing
the native work.

Reads and writes are tracked on one execution instance. Repeated access on
that instance is allowed; switching instances or entering a native operation
requires `commit()`. A write advances the coefficient revision; a read does not.
Keep asynchronous inputs and returned unmanaged views' owners alive until
their work completes. Matrix destruction commits its tracked accesses.

Trilinos 17.2.1 exposes only `ReadWrite` access for native block coefficients.
Consequently, a BSR overwrite may synchronize previous host coefficients even
when the caller will overwrite every value. CSR retains `OverwriteAll` access.

## Several fields to one AoS vector

`field_pack` holds descriptors only. It accepts `FieldView`s and rank-one
Kokkos Views, including strided component subviews. Components are concatenated
in argument order within each cell, independently of the source layouts and
scalar types. Transfers cast to the destination scalar type.

Five separate scalar residual fields:

```cpp
// res1, ..., res5 are FieldViews or rank-one component Views.
auto residuals = field_pack(res1, res2, res3, res4, res5);
copy_from_fields(rhs, residuals, exec);
rhs.commit();
// rhs: res1(cell0), ..., res5(cell0), res1(cell1), ..., res5(cell1), ...
```

A scalar, a three-component field and another scalar have the same layout:

```cpp
auto residuals = field_pack(density_residual, momentum_residual, energy_residual);
copy_from_fields(rhs, residuals, exec);

// Explicit-execution convenience form; no intermediate field_pack variable:
copy_from_fields(rhs, exec, density_residual,
    momentum_residual.component(0), momentum_residual.component(1),
    momentum_residual.component(2), energy_residual);
rhs.commit();
```

Each transfer submits one kernel and allocates no intermediate buffer. It uses
only the owned prefix; source ghost/boundary slots are not packed. The inverse
operation leaves all destination ghost/boundary tails untouched:

```cpp
auto increments = field_pack(density_increment, momentum_increment, energy_increment);
copy_to_fields(solution, increments, exec);
solution.commit();  // completes the tracked read from the vector
```

Keep all fields alive until completion. Destinations must be writable and must
not overlap one another or the vector. Packing sources must not alias the
destination vector. The total component count must match the intended block
size, and field order must match equation/variable order in the matrix. Transfers
check local divisibility and prefix extents; a scalar vector cannot infer or
validate that physical ordering from its GIDs.

The existing single-field and single-component transfer APIs remain available.

## Solvers and preconditioners

`Solver` and `Preconditioner` now take the matrix type followed by the optional
preconditioner scalar type. The matrix determines the solver scalar and execution
space. Update old explicit template arguments accordingly:

```cpp
SolverOptions options;
options.method = "FGMRES";
options.preconditioner.type = "RELAXATION"; // native block Jacobi for BSR

Solver solver(matrix, options); // uses PreconditionerReal
const auto result = solver.solve(rhs, solution);

// Explicit precision, also valid for a CsrMatrix:
Solver<decltype(matrix), float> mixed(matrix, options);
Preconditioner<decltype(matrix), float> standalone(matrix, options.preconditioner);
```

The matrix must outlive the solver/preconditioner. Native solve and
preconditioner preparation remain collective and blocking. The supplied initial
guess is retained; independent true-residual verification remains authoritative.
GMRES, FGMRES, BiCGSTAB, CG and MINRES are exercised with BSR. The selected
algorithm's mathematical requirements still apply.

Equal precisions share the original native matrix and do not create a precision
adapter. Different precisions allocate another BSR coefficient array, share the
device graph, convert coefficients during preparation and reuse the existing
precision adapter during application. Prefer FGMRES when a low-precision
preconditioner behaves as an inexact operator.

For block ILU, use a Schwarz wrapper around `RBILUK`:

```cpp
options.preconditioner.type = "SCHWARZ";
options.preconditioner.parameters.set("schwarz: overlap level", 0);
options.preconditioner.parameters.set("inner preconditioner name", "RBILUK");
options.preconditioner.parameters.sublist("inner preconditioner parameters")
    .set("fact: iluk level-of-fill", 1);
```

The custom factory receives the native matrix in its actual storage format and
preconditioner precision. Its returned operator must have the scalar-DOF domain
and range maps. Other native preconditioners are usable only when they support
the selected matrix format and map contract.

For BSR, the `SCHWARZ` defaults are zero overlap and an `RBILUK` inner solver.
Positive overlap is rejected: the overlapping-row path in Trilinos 17.2.1
requires a scalar `CrsMatrix`. Direct `RBILUK` exposes local/block maps and
therefore does not satisfy the distributed scalar-DOF operator contract;
use the Schwarz wrapper, including on one rank.

### Block AMG in Trilinos 17.2.1

BSR `AMG`/`MUELU` uses the native block hierarchy with these defaults:

| Parameter | BSR default |
| --- | --- |
| `multigrid algorithm` | `unsmoothed` |
| `number of equations` | Matrix block size |
| `use kokkos refactor` | `false` |
| `smoother: type` | `RELAXATION`, Jacobi |
| `coarse: type` | `RELAXATION`, Jacobi, 8 sweeps |
| `reuse: type` | `none` |

The equation count must match the block size. `use kokkos refactor=true` is
rejected for BSR because that native path requests a point-CSR local matrix.
Native parameters remain configurable within the selected backend's capabilities.
CSR keeps its existing smoothed-aggregation/Kokkos/Chebyshev defaults.

The resulting hierarchy matrices retain block storage, including coarse levels.
However, Xpetra's block matrix-matrix products in this release temporarily
convert operands to scalar CSR during hierarchy setup and convert the result
back to BSR. Trilinos prints `Using inefficient BlockCrs Multiply Placeholder`
for that path. This affects setup time and peak memory; persistent BSR storage
does not guarantee a smaller peak during AMG construction. KRONA does not
maintain an expanded CSR companion matrix for BSR.

`reuse: type=full` retains the native hierarchy/operator according to MueLu's
reuse semantics. It is an explicit performance/convergence choice; unchanged
coefficients already avoid preparation regardless of that parameter.

## Native-library limits and future work

- Blocks are uniformly sized, dense and square. Variable block sizes and
  heterogeneous field-split/Schur operators need a separate representation.
- BSR reduces graph storage; dense block coefficients still occupy `B²` scalars.
- Tpetra shares the device graph but creates separate host index buffers inside
  each native `BlockCrsMatrix`; that allocation is native-library behaviour.
- Native `BlockCrsMatrix::apply` in this Trilinos release does not expose transpose
  application (`hasTransposeApply()` is false).
- Native block coefficient access lacks `OverwriteAll`; avoiding that limitation
  needs a native API change or a separately designed storage backend.
- Native block AMG setup still needs an efficient block matrix-matrix path in
  Xpetra to eliminate temporary CSR expansion. Measure setup peak memory and
  solve time separately when choosing a preconditioner.
- GPU/HIP execution and performance require backend-specific verification; a
  passing Serial/MPI run does not establish GPU performance.

## Verification

The suite registers 50 checks, including complete positive-suite runs on two
and three MPI ranks. It covers independent BSR products versus a dense-block
reference and expanded CSR, coefficient order/layouts, five Krylov methods,
block Jacobi, Schwarz/RBILUK, multilevel block AMG and reuse, both precision
directions, factory sharing/conversion, empty distributions, field packing and
unpacking, and contract rejections.

Validation used the actual Trilinos 17.2.1/Kokkos 5.2.1 Serial libraries and
MPICH 5.0.2, with the supplied `common/test_support.hpp`. A standalone fixture
provided the missing KRONA configuration and MPI/runtime wrappers. This was
not a build of the full KRONA repository. CUDA, HIP and OpenMP were not run.

