# Gradients

Header: `<krona/numerics/gradient/gradient_operator.hpp>`

The module reconstructs three-dimensional cell gradients on a static local
mesh. It reads owned cells, MPI ghost cells and physical boundary values,
and writes gradients only for owned cells.

`GradientOperator` is an immutable plan bound to one execution-space instance.
Construction validates the selected inputs and prepares any persistent
coefficients. Application calls reuse that plan without allocating working
buffers or performing MPI communication.

## Schemes and requirements

All schemes currently require
`MeshData::FaceCells | MeshData::Geometry`.
Use `requirements(options)` to obtain the required storage groups before
constructing the plan.

| Scheme | Auxiliary connectivity | Auxiliary geometry | Application |
| --- | --- | --- | --- |
| `GreenGaussCell` | `CellFaces` | Interpolation weights when selected | One kernel over owned cells; no atomics |
| `GreenGaussFace` | None | Interpolation weights when selected | Clear selected outputs, then traverse faces with atomic updates |
| `LeastSquaresFace` | `CellFaces` | Optional displacement vectors | One kernel over owned cells using prepared coefficients |

The defaults are:

```cpp
gradient::Options options{
    .scheme = gradient::Scheme::LeastSquaresFace,
    .interpolation = gradient::Interpolation::Arithmetic,
    .pivot_tolerance = 1.0e-12
};
```

Green–Gauss uses arithmetic interpolation unless `AuxiliaryWeights` is selected.
For an internal face, the auxiliary weight belongs to the owner:

```text
q_face = w_owner * q_owner + (1 - w_owner) * q_neighbour
```

Physical boundary values are used directly, without owner–neighbour
interpolation.

Least squares uses inverse-distance-squared weighting. Its coefficients are
prepared once for every owned-cell-to-face CSR entry. Preparation uses
`double`, scales each stencil before forming the normal matrix, and applies
relative Cholesky pivot checks. Singular or insufficiently three-dimensional
stencils are rejected; there is no pseudoinverse fallback.

`Interpolation` does not affect least squares. If displacement vectors are
attached, preparation uses them; otherwise it computes displacements from
cell and face centroids.

## Data contracts

Faces must follow the mesh module's grouped ordering:

- `[0, n_internal_faces)`: owned–owned;
- `[n_internal_faces, n_inner_faces)`: owned–ghost;
- `[n_inner_faces, n_faces)`: physical boundaries.

Each input component must contain at least:

```text
n_cells + (n_faces - n_inner_faces)
```

entries. Physical boundary face `f` reads its value at:

```text
n_cells + (f - n_inner_faces)
```

These entries represent values at boundary face centroids, not values at
additional ghost-cell centres. Boundary conditions must populate them before
the gradient reads them.

Each output component must contain at least `n_own` entries. Ghost and boundary
output entries, when present, are preserved.

Input and output use the same floating-point value type. Inputs may be
read-only; outputs must be mutable. Their memory spaces must be accessible
from `Exec`. Component views may have non-unit strides.

Input storage must not overlap output locations being written. Output
components must not overlap one another. The operator does not detect aliasing.

## Construction and fields

The following example assumes initialized runtimes, a ready `host_mesh`, and
`host_connectivity` containing the owned-cell `CellFaces` cache.

```cpp
#include <krona/fields/fields_manager.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>

namespace mc = krona::mesh::compute;
namespace fields = krona::fields;
namespace gradient = krona::numerics::gradient;

using Exec = krona::execution::ExecutionSpace;
Exec exec;

gradient::Options options{};
const auto needed = gradient::requirements(options);

mc::MeshStorage<Exec> mesh(
    host_mesh, {.data = needed.mesh}, exec);

mc::AuxConnectivityStorage<Exec> connectivity(
    host_mesh, host_connectivity, needed.connectivity, exec);

gradient::GradientOperator<Exec> op(
    mesh.view(), connectivity.view(), {}, options, exec);

fields::FieldSizes sizes{
    .own_cells = static_cast<std::size_t>(host_mesh.n_own),
    .cells = static_cast<std::size_t>(host_mesh.n_cells),
    .faces = static_cast<std::size_t>(host_mesh.n_faces),
    .bfaces = static_cast<std::size_t>(
        host_mesh.n_faces - host_mesh.n_inner_faces),
    .nodes = static_cast<std::size_t>(host_mesh.n_nodes)
};

fields::FieldsManager<Exec> registry(sizes, exec);

auto q = registry.add_field<krona::StorageReal>(
    "q", fields::FieldLocation::CellExtnd);

auto dq = registry.add_field<krona::StorageReal, 3>(
    "grad_q", fields::FieldLocation::OwnCell,
    fields::FieldInitialization::Uninitialized);

// q starts at zero. For a calculation, populate owned values,
// exchange MPI ghosts and impose physical boundary values.
op.apply(q.as_const(), dq);

exec.fence("gradient example complete");
```

The final fence closes this example. It is not required between correctly
ordered operations submitted to the same execution-space instance.

To use auxiliary Green–Gauss interpolation weights, attach a ready host cache:

```cpp
gradient::Options weighted_options{
    .scheme = gradient::Scheme::GreenGaussCell,
    .interpolation = gradient::Interpolation::AuxiliaryWeights
};

mc::AuxGeometryStorage<Exec> geometry(
    host_mesh, host_geometry,
    krona::mesh::AuxGeomType::FaceInterpolationWeights, exec);

gradient::GradientOperator<Exec> weighted(
    mesh.view(), connectivity.view(), geometry.view(),
    weighted_options, exec);
```

The gradient module consumes existing mesh caches; it does not build
connectivity or auxiliary geometry.

## Applying the plan

### Whole fields

For `N` input components, provide `3*N` output components:

```cpp
auto velocity = registry.add_field<krona::StorageReal, 3>(
    "velocity", fields::FieldLocation::CellExtnd);

auto velocity_gradient =
    registry.add_field<krona::StorageReal, 9>(
        "velocity_gradient", fields::FieldLocation::OwnCell,
        fields::FieldInitialization::Uninitialized);

op.apply(velocity.as_const(), velocity_gradient);
```

Output component `3*c + axis` stores the derivative of input component `c`
along `x`, `y` or `z`.

### Individual components and batches

`apply_component` accepts four rank-one Kokkos Views, including strided subviews:

```cpp
op.apply_component(
    q.as_const().component(0),
    dq.component(0), dq.component(1), dq.component(2));
```

`apply_set` fuses a compile-time batch of scalar components. Each `std::array`
must contain a homogeneous View type; all inputs and outputs use the same
scalar value type.

```cpp
auto temperature = registry.add_field<krona::StorageReal>(
    "temperature", fields::FieldLocation::CellExtnd);

auto dt = registry.add_field<krona::StorageReal, 3>(
    "grad_temperature", fields::FieldLocation::OwnCell,
    fields::FieldInitialization::Uninitialized);

op.apply_set(
    std::array{q.component(0), temperature.component(0)},
    std::array{dq.component(0), dt.component(0)},
    std::array{dq.component(1), dt.component(1)},
    std::array{dq.component(2), dt.component(2)});
```

The batch passes component pointers and strides directly in the kernel
descriptor; it does not allocate a device pointer table.

### Owned-cell ranges

All three entry points accept a half-open `CellRange`:

```cpp
// split is an owned-cell index chosen by the caller.
op.apply(q, dq, {0, split});
op.apply(q, dq, {split, host_mesh.n_own});
```

The default `{0, -1}` selects all owned cells. An empty range submits no work.
Full input and owned-output extents are still required for partial ranges.

## Execution and lifetime

Construction is synchronous: validation uses host-result reductions, LS
preparation runs on the supplied `Exec`, and construction fences that instance
before returning. Construct the plan during setup, not inside the time loop.

Application calls perform descriptor checks and enqueue kernels without an
explicit fence. Actual host blocking depends on the execution backend.

Use the same execution-space instance for dependent field updates, halo
unpacking, gradient evaluation and consumers. Work submitted through another
instance requires explicit ordering.

The operator performs no halo exchange. Input ghosts and boundary entries must
be ready before the corresponding reads. A halo `finish_async()` submitted on
the same instance can precede `apply()` without an intervening fence; that halo
operation still waits for MPI completion on the host.

If gradient ghosts are needed later, allocate an output with cell or extended
cell storage and exchange its owned results separately.

The operator borrows mesh, auxiliary and field storage. Keep their owners and
the plan alive until submitted work finishes. The plan cannot be copied or
moved, and destruction does not provide a completion fence.

## Validation

Construction checks counts, grouped faces, selected storage shapes, face
topology and the required CSR structure and incidence.

Green–Gauss additionally validates face metrics, owned-cell geometry and any
selected interpolation weights. Ghost-cell volumes are not used.

LS validates any attached displacement vectors and rejects invalid, singular
or ill-conditioned preparation stencils. Volumes, areas and normals are not
used by the LS reconstruction, although the current storage requirements
include their groups.

`pivot_tolerance` must be finite and strictly between zero and one. It is
validated for every scheme and used by LS preparation.

Validation is local and synchronous. It does not establish MPI message
agreement, verify field values or boundary conditions, detect storage
aliasing, or prove mesh closure and numerical consistency. Attached caches
must correspond to the current mesh. Application calls do not repeat mesh
validation.

## Precision

The template parameters are:

```cpp
GradientOperator<Exec, Coefficient, Accumulator>
```

`Coefficient` controls persistent LS coefficient storage and defaults to
`GeometryReal`. `Accumulator` defaults to `AccumulationReal`.

Their application semantics differ by scheme:

| Scheme | Contribution arithmetic | Sum accumulation | Final output |
| --- | --- | --- | --- |
| `GreenGaussCell` | `Accumulator` | `Accumulator` | Converted once to the field type |
| `LeastSquaresFace` | `Accumulator` | `Accumulator` | Converted once to the field type |
| `GreenGaussFace` | `Accumulator` | Field type, through atomics | Updated directly |

Consequently, `Accumulator=double` does not provide double-precision summation
for `GreenGaussFace` with `float` output. Each contribution is converted to
`float` before addition. Cancellation can therefore produce larger errors
than in the cell-based schemes. Atomic update order also prevents a general
bitwise reproducibility guarantee.

This is the current intentional contract; the face scheme does not allocate
a separate accumulation buffer.

## Known limitations and possible improvements

- The plan targets static three-dimensional stencils. Moving meshes,
  adaptation and renumbering require rebuilding affected caches and plans;
  there is no refresh or revision-tracking API.
- LS currently requests complete geometry storage groups, including metrics
  it does not use. Finer storage selection could reduce this requirement.
- Green–Gauss has no skewness-correction iteration and is not generally
  affine-exact on arbitrary skewed meshes.
- `GreenGaussFace` traverses all local faces even for a partial owned range.
  Range-specific face lists could be considered if profiling justifies them.
- Optional wider accumulation for the face scheme would require an explicit
  workspace or another accumulation strategy, with additional memory and
  execution costs.
- Large fused batches increase per-thread working storage. Choose batch sizes
  through profiling rather than assuming that larger batches are always faster.

*Note*: `required_input_size()` reports the extended input extent.
`coefficient_bytes()` reports allocated LS coefficient bytes; Green–Gauss
plans have no coefficient allocation.