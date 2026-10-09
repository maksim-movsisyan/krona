# Finite-volume gradients

`krona::numerics::fv::gradient::GradientOperator` evaluates cell-centred gradients on a static 3-D mesh. It is independent of equations, boundary-condition types and MPI scheduling.

## Data contract

- Faces are grouped: owned-owned, owned-ghost, physical boundary. Normals are unit vectors pointing outward from the owner; volumes and areas are positive.
- The input has at least `n_cells + n_faces - n_inner_faces` entries. `[0,n_own)` contains owned values, `[n_own,n_cells)` MPI ghosts, and entry `n_cells + f - n_inner_faces` the value for boundary face `f`.
- Input ghosts and physical boundary values must be ready before use. The operator performs no exchange and imposes no physical BC.
- Only output `[0,n_own)` is written. Optional `CellRange{begin,end}` restricts it further. Output ghosts, boundary entries and cells outside the requested range remain untouched; they can remain uninitialized until halo/BC processing.
- Input and output storage must not overlap. Output components must not overlap each other. All participating views must be accessible to `Exec` and have the same scalar value type.
- A field with `N` input components has `3*N` output components: `3*c+0`, `3*c+1`, `3*c+2` are the x, y, z derivatives of component `c`. Thus a vector gradient uses tensor order `xx,xy,xz,yx,yy,yz,zx,zy,zz`, with the first index denoting the input component.
- `FieldView::size()` and rank-one component extents may exceed the minimum. LayoutLeft, LayoutRight and strided component subviews are supported.

## Boundary location and interpolation

Choose the interpretation explicitly in `Options`:

| Boundary mode | Input boundary entry | GG face value | LS sample displacement |
|---|---|---|---|
| `MirrorCell` (default) | Value at `2*F-C_owner` | `(q_owner+q_boundary)/2` | `2*(F-C_owner)` |
| `Face` | Value at face centroid `F` | `q_boundary` | `F-C_owner` |

`MirrorCell` preserves the reference GG boundary averaging. A Neumann condition must first be converted to a value at the selected location by the BC module; the gradient operator does not consume normal derivatives directly.

Interior GG interpolation is arithmetic by default. `Interpolation::AuxiliaryWeights` uses the existing owner weight: `q_face=w*q_owner+(1-w)*q_neighbour`. These weights do not correct tangential skewness by themselves.

LS uses inverse-distance-squared weighting. It includes all face neighbours **and physical boundary samples**, obtained from `CellFaces` CSR. It does not use `CellCellsByFace`, whose entries describe real cells only. Optional auxiliary displacement vectors are reused; their boundary convention must be `2*(F-C_owner)`. With `BoundaryValues::Face`, the operator halves that vector.

## Schemes and preparation

| Scheme | Required auxiliary data | Apply organization | Additional persistent memory |
|---|---|---|---|
| `GreenGaussCell` | `CellFaces` | One owned cell per work item; local accumulation | None |
| `GreenGaussFace` | None | Clear requested owned output, then scatter faces with atomics | None |
| `LeastSquaresFace` | `CellFaces` | One owned cell per work item; cached weighted differences | Three coefficients per CSR entry, plus possible layout padding |

All schemes borrow face connectivity and cell/face geometry from `MeshView`. GG auxiliary interpolation needs `FaceInterpolationWeights`. LS optionally reuses `FaceCellDistanceVector`.

GG is evaluated using `q_face-q_cell`, exploiting closed-cell surface-vector cancellation. This makes constant fields produce zero directly and reduces cancellation from large constant offsets. The mesh must contain a complete, correctly oriented face set for every owned cell.

LS setup uses double intermediates, a cell-local length scale and relative Cholesky pivot checks. It stores coefficients in `Coefficient` (default `GeometryReal`). Singular or severely ill-conditioned 3-D stencils cause a diagnosed `mpi::fatal`, not a silent zero gradient. There is no fixed neighbour-count limit and no hidden regularization. This is a normal-equations method, not a QR/SVD solver for arbitrarily ill-conditioned stencils.

The local constructor validates and prepares once, then fences the supplied execution-space instance. It is not collective. No preparation is repeated by `apply`.

## Execution and lifetime

- Every apply call uses the constructor's `Exec` instance. It allocates no application buffers, performs no MPI calls and inserts no fence. Backend execution may still be synchronous on CPU.
- A subsequent kernel on the same instance sees the result in order. Before external access, another execution-space instance or destruction, synchronize explicitly.
- The plan and all borrowed mesh/auxiliary/field owners must outlive submitted work. Destruction does not fence. Recreate the plan after motion, reordering, migration or adaptation; changing counts alone is not a sufficient validity check.
- Cell GG and LS sum in `Accumulator` (default `AccumulationReal`) and convert when storing. Face GG atomically accumulates in the **output value type**; addition order and rounding can vary.
- Runtime scheme selection is one host-side switch per apply. Device functors contain no virtual calls or factory lookup. `apply_set` fuses a compile-time batch of components; large batches may increase register pressure.
- A restricted range requires readiness only for the samples actually used by its cells. Face GG skips unrelated faces before reading field values. The caller identifies independent ranges; the operator does not classify interior cells automatically.

## Verification and limits

Tests cover constant and affine fields, skewed/unequal-width cells, both boundary conventions, auxiliary and computed displacements, layouts/strides, batches, partial ranges, untouched tails, precision choices, rescaled meshes, degree 90, empty partitions, diagnosed singularity and actual two-rank field/gradient exchange.

Manufactured constant/affine checks use `64*epsilon*max(1,abs(expected))`, where epsilon is the largest machine epsilon of the field, coefficients, accumulator and geometry types. This rounding-level tolerance applies to the well-conditioned test stencils.

Affine exactness for GG depends on face interpolation reproducing affine face values. Arithmetic averaging is not generally affine-exact on arbitrary meshes. These tests do not prove convergence on every mesh family or performance within a fixed percentage of hand-written CUDA.

Vertex-neighbour LS is intentionally excluded: the current one-layer face halo does not guarantee all remote vertex-only neighbours. A complete node stencil requires a matching expanded halo before it can be decomposition-independent.
