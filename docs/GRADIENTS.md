# Finite-volume gradients

`krona::numerics::gradient::GradientOperator` evaluates cell-centred gradients on a static 3-D mesh. It is independent of equations, boundary-condition types and MPI scheduling.

## Data contract

- Faces are grouped: owned-owned, owned-ghost, physical boundary. Normals are unit vectors pointing outward from the owner; volumes and areas are positive.
- The input has at least `n_cells + n_faces - n_inner_faces` entries. `[0,n_own)` contains owned values, `[n_own,n_cells)` MPI ghosts, and entry `n_cells + f - n_inner_faces` the value at the centroid of physical boundary face `f`. The boundary block contains face values, not virtual cell values.
- Input ghosts and physical boundary values must be ready before use. The operator performs no exchange and imposes no physical BC.
- The output needs at least `n_own` entries. Only `[0,n_own)` is written; optional `CellRange{begin,end}` restricts it further. All entries outside the requested range remain untouched. MPI ghost gradients, if needed, are filled by a separate exchange. No physical boundary output block is required.
- Input and output storage must not overlap. Output components must not overlap each other. All participating views must be accessible to `Exec` and have the same scalar value type.
- A field with `N` input components has `3*N` output components: `3*c+0`, `3*c+1`, `3*c+2` are the x, y, z derivatives of component `c`. Thus a vector gradient uses tensor order `xx,xy,xz,yx,yy,yz,zx,zy,zz`, with the first index denoting the input component.
- `FieldView::size()` and rank-one component extents may exceed the minimum. LayoutLeft, LayoutRight and strided component subviews are supported.

## Boundary location and interpolation

Physical boundary input values are always located at the face centroid `F`. GG uses `q_face=q_boundary` directly, and LS uses the displacement `F-C_owner`. There is no boundary-location option and no mirrored physical boundary cell.

The BC module must supply the face value before applying the gradient. A Neumann condition must first be converted to a face value using the BC module's reconstruction; the gradient operator does not consume normal derivatives directly.

Interior GG interpolation is arithmetic by default. `Interpolation::AuxiliaryWeights` uses the existing owner weight: `q_face=w*q_owner+(1-w)*q_neighbour`. Physical boundaries use the supplied face value under either setting; the auxiliary boundary weight is `0`. These weights do not correct tangential skewness by themselves.

LS uses inverse-distance-squared weighting. It includes all face neighbours **and physical boundary samples**, obtained from `CellFaces` CSR. It does not use `CellCellsByFace`, whose entries describe real cells only. Optional auxiliary displacement vectors are reused with the common geometry convention: interior and MPI-interface vectors are `C_neighbour-C_owner`, and physical boundary vectors are `F-C_owner`. Without the cache, LS computes these vectors from centroids. The interpolation option does not affect LS.

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

## Storage requirements

The free function `krona::numerics::gradient::requirements(Options)` returns the mandatory groups before creating any gradient operator:

```cpp
namespace gradient = krona::numerics::gradient;

const gradient::Options options{
    .scheme = gradient::Scheme::GreenGaussCell,
    .interpolation = gradient::Interpolation::AuxiliaryWeights
};
const auto needed = gradient::requirements(options);
// needed.mesh, needed.connectivity, needed.geometry
```

The result is a `gradient::Requirements` with `MeshData`, `AuxConnType` and `AuxGeomType` masks. The function is `constexpr`; for valid options it creates no plan, allocates no data and performs no preparation. It is a free function, not a `GradientOperator` method.

| Scheme | `needed.mesh` | `needed.connectivity` | `needed.geometry` |
|---|---|---|---|
| `GreenGaussCell` | `FaceCells \| Geometry` | `CellFaces` | `FaceInterpolationWeights` for auxiliary interpolation; otherwise `None` |
| `GreenGaussFace` | `FaceCells \| Geometry` | `None` | `FaceInterpolationWeights` for auxiliary interpolation; otherwise `None` |
| `LeastSquaresFace` | `FaceCells \| Geometry` | `CellFaces` | `None` |

Current `MeshData` groups are coarse: LS requires attached cell/face geometry groups, although it does not use volumes, areas or normals. `FaceCellDistanceVector` is an optional LS cache and is not included in the mandatory geometry mask.

A solver can combine these masks with those of other operators using `|`, build the shared host caches, and create storage once for the whole configuration. MPI halo maps are requirements of the exchange module and must be selected separately when exchanges are used.

The query does not replace constructor validation. The operator still checks the supplied descriptors, validates the selected arrays through `mesh/compute/validation.hpp`, and prepares its own LS coefficients. GG checks owned-cell volumes; LS retains its stencil conditioning checks. Missing mandatory data causes a diagnosed failure; the operator does not build shared mesh connectivity or geometry.

## Execution and lifetime

- Every apply call uses the constructor's `Exec` instance. It allocates no application buffers, performs no MPI calls and inserts no fence. Backend execution may still be synchronous on CPU.
- A subsequent kernel on the same instance sees the result in order. Before external access, another execution-space instance or destruction, synchronize explicitly.
- The plan and all borrowed mesh/auxiliary/field owners must outlive submitted work. Destruction does not fence. Recreate the plan after motion, reordering, migration or adaptation; changing counts alone is not a sufficient validity check.
- Cell GG and LS sum in `Accumulator` (default `AccumulationReal`) and convert when storing. Face GG atomically accumulates in the **output value type**; addition order and rounding can vary.
- Runtime scheme selection is one host-side switch per apply. Device functors contain no virtual calls or factory lookup. `apply_set` fuses a compile-time batch of components; large batches may increase register pressure.
- A restricted range requires readiness only for the samples actually used by its cells. Face GG skips unrelated faces before reading field values. The caller identifies independent ranges; the operator does not classify interior cells automatically.

## Verification and limits

Tests cover constant and affine fields, skewed/unequal-width cells, boundary face values, auxiliary and computed displacements, layouts/strides, batches, partial ranges, untouched tails, precision choices, rescaled meshes, degree 90, empty partitions, diagnosed singularity and actual two-rank field/gradient exchange.

Manufactured constant/affine checks use `64*epsilon*max(1,abs(expected))`, where epsilon is the largest machine epsilon of the field, coefficients, accumulator and geometry types. This rounding-level tolerance applies to the well-conditioned test stencils.

Affine exactness for GG depends on face interpolation reproducing affine face values. Arithmetic averaging is not generally affine-exact on arbitrary meshes. These tests do not prove convergence on every mesh family or performance within a fixed percentage of hand-written CUDA.

Vertex-neighbour LS is intentionally excluded: the current one-layer face halo does not guarantee all remote vertex-only neighbours. A complete node stencil requires a matching expanded halo before it can be decomposition-independent.
