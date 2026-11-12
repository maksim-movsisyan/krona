# Cell gradients

`krona::numerics::gradient::GradientOperator` computes cell-centred gradients of scalar or multicomponent fields on a static 3D mesh. It writes owned cells only and reuses one mesh-dependent plan for different fields and repeated updates.

| Header | Purpose |
| --- | --- |
| `krona/numerics/gradient/types.hpp` | Schemes, interpolation options, requirements and owned-cell ranges. |
| `krona/numerics/gradient/gradient_operator.hpp` | Prepared gradient operator and asynchronous application methods. |
| `krona/numerics/gradient/correction.hpp` | Borrowed gradient adapters and local geometric projections. No dependency from other modules (except fields). |

## Input and output

Input components use the extended cell layout:

| Interval | Values required by the operator |
| --- | --- |
| `[0, n_own)` | Owned-cell values at cell centroids. |
| `[n_own, n_cells)` | Current MPI ghost values at their actual cell centroids. |
| `[n_cells, n_cells + n_bfaces)` | Boundary values at external face centroids; `n_bfaces = n_faces - n_inner_faces`. |

External face `f` reads input slot `n_cells + (f - n_inner_faces)`. These slots contain physical face values, not values at reflected ghost-cell centres. `required_input_size()` returns `n_cells + n_bfaces`.

The operator does not fill boundaries or exchange ghosts. In an application, update owned values, complete the required input halo exchange, fill physical boundary slots, then submit the gradient calculation. Boundary values can come from the scalar/physical boundary managers or your own code. Patch labels and condition objects are not inputs to this operator.

Outputs need at least `n_own` entries per component. A call replaces gradients in the requested owned-cell interval; output initialization is not required for those cells. Other entries, including any allocated ghost or boundary slots, remain unchanged. Allocate `n_cells` output entries and perform a separate halo exchange if a later algorithm needs ghost gradients.

For an input with `N` components, `apply_async()` writes `3*N` output components:

| Input component | Output components |
| --- | --- |
| `c` | `3*c + 0`: x derivative; `3*c + 1`: y derivative; `3*c + 2`: z derivative. |
| Scalar | `[d(phi)/dx, d(phi)/dy, d(phi)/dz]`. |
| Velocity `(u, v, w)` | `[du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz]`. |

Input and output components in one call must have the same floating-point value type. Const input is supported; output must be mutable. All views must be accessible to `Exec`, with valid storage and positive strides. Field location tags are not checked. SoA and AoS layouts are supported, and input/output layouts may differ.

Keep written outputs disjoint from input data and from each other. Aliasing is not detected. Application calls check descriptors and ranges, not the finiteness of every field value; supply valid numerical inputs.

## Select a scheme and prepare the mesh data

```cpp
namespace gradient = krona::numerics::gradient;
gradient::Options options{}; // LeastSquaresFace by default.
const auto needed = gradient::requirements(options);
```

`requirements()` returns the mandatory mesh, auxiliary-connectivity, and auxiliary-geometry masks for the selected options. It does not allocate data or submit work.

All current schemes require grouped faces and `MeshData::FaceCells | MeshData::Geometry`. Use one consistent prepared mesh and matching caches. Cell centroids must include owned and ghost cells. The face normal is unit length and points out of the owner; the neighbour lies on the opposite side for interior faces. Cell/face geometry and numbering must remain unchanged while the plan is reused.

| Scheme | Required auxiliary data | Preparation and application |
| --- | --- | --- |
| `LeastSquaresFace` | `CellFaces` connectivity. No mandatory geometry cache. | Precomputes three least-squares coefficients per owned-cell/face incidence. Each application traverses owned-cell rows and combines sample differences. |
| `GreenGaussCell` | `CellFaces`; interpolation weights when selected. | Each owned-cell iteration sums its face contributions and writes its gradient once. |
| `GreenGaussFace` | No connectivity cache; interpolation weights when selected. | Clears selected output cells, then traverses local faces and atomically adds contributions to selected owned owners/neighbours. |

The least-squares calculation does not use volumes, areas, or normals numerically, but the current API still requires the full geometry group and its array extents. Green–Gauss requires positive owned volumes and valid face metrics.

`FaceCellDistanceVector` is an optional least-squares cache. Interior vectors are `C_neighbour - C_owner`; external vectors are `F - C_owner`. Without it, the operator computes these differences from centroids during preparation. A supplied cache must match the current mesh geometry. No displacement cache is needed for Green–Gauss.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `scheme` | `LeastSquaresFace` | Selects one of the three algorithms above. |
| `interpolation` | `Arithmetic` | Green–Gauss interior-face interpolation: arithmetic mean, or attached `FaceInterpolationWeights`. Ignored by least squares. |
| `pivot_tolerance` | `1e-12` | Relative least-squares Cholesky pivot threshold. Must be finite and strictly between zero and one, even when another scheme is selected. |

For Green–Gauss, an interior face value is `w*phi_owner + (1-w)*phi_neighbour`. `Arithmetic` uses `w = 0.5`; `AuxiliaryWeights` reads the owner weight from the auxiliary cache. The host builder obtains it from projected distances to the face plane. Boundary face values are read directly from the appended slots, without interpolation.

Both Green–Gauss implementations compute the same discrete expression:

$$
\nabla\phi_P \approx \frac{1}{V_P}\sum_{f\in P}
 A_f\,\mathbf n_{P,f}\,(\phi_f-\phi_P).
$$

Their traversal and summation order differ. The face implementation uses atomic additions, so floating-point results need not be bitwise identical or reproducible across execution schedules. Neither interpolation mode adds a skewness correction or limiter automatically.

Least squares fits sample differences using weights `1 / |d|^2`:

$$
\min_{\mathbf g}\sum_s
 \frac{(\mathbf d_s\cdot\mathbf g-(\phi_s-\phi_P))^2}{|\mathbf d_s|^2}.
$$

A sample is the opposite cell across an interior face, including a stored MPI ghost, or the external face centroid. The stencil uses face adjacency, not all node neighbours. Preparation scales displacements, forms a 3-by-3 matrix, and checks its Cholesky pivots relative to the matrix trace. A singular or insufficiently resolved 3D stencil terminates setup; there is no automatic lower-dimensional fit or scheme fallback.

## Complete scalar example

Assume MPI and Kokkos are initialized and `host_mesh` is a prepared `mesh::host::MeshPart` with geometry and grouped faces. The following example computes the gradient of `phi = 2*x - 3*y + 0.75*z`. It sets all input samples analytically, including ghosts and boundary slots; a solver would obtain those entries through halo exchange and boundary filling instead.

```cpp
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/fields/field_storage.hpp>
#include <array>

namespace gradient = krona::numerics::gradient;
namespace device = krona::mesh::device;
namespace host = krona::mesh::host;
namespace fields = krona::fields;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::StorageReal;
```

Build and retain the required data, then create the operator:

```cpp
Exec exec{};
gradient::Options options{};
const auto needed = gradient::requirements(options);

const auto host_connectivity = host::build_aux_connectivity(
    host_mesh, needed.connectivity);
const auto host_geometry = host::build_aux_geometry(host_mesh, needed.geometry);
device::MeshStorage<Exec> mesh{host_mesh, {.data = needed.mesh}, exec};
device::AuxConnectivityStorage<Exec> connectivity{host_connectivity, exec};
device::AuxGeometryStorage<Exec> geometry{host_geometry, exec};
gradient::GradientOperator<Exec> op{
    mesh.view(), connectivity.view(), geometry.view(), options, exec};

const auto owned = static_cast<std::size_t>(host_mesh.n_own);
fields::FieldStorage<Real, 1, Exec> phi{
    "phi", op.required_input_size(), fields::FieldLocation::CellExtnd, exec};
fields::FieldStorage<Real, 3, Exec> grad_phi{
    "grad_phi", owned, fields::FieldLocation::OwnCell, exec};
```

Fill the input, calculate, and copy the result to the host:

```cpp
auto initial = phi.create_host_buffer();
for (krona::LocalIndex i = 0; i < host_mesh.n_cells; ++i)
    initial(i) = 2*host_mesh.cell_centroid_x[i]
               - 3*host_mesh.cell_centroid_y[i]
               + Real{0.75}*host_mesh.cell_centroid_z[i];
for (krona::LocalIndex f = host_mesh.n_inner_faces; f < host_mesh.n_faces; ++f) {
    const auto i = static_cast<std::size_t>(host_mesh.n_cells)
                 + static_cast<std::size_t>(f - host_mesh.n_inner_faces);
    initial(i) = 2*host_mesh.face_centroid_x[f]
               - 3*host_mesh.face_centroid_y[f]
               + Real{0.75}*host_mesh.face_centroid_z[f];
}

auto observed = grad_phi.create_host_buffer();
krona::execution::complete(exec, [&] {
    Kokkos::deep_copy(exec, phi.view().kokkos_view(), initial);
    op.apply_async(phi.view().as_const(), grad_phi.view());
    Kokkos::deep_copy(exec, observed, grad_phi.view().kokkos_view());
});
// Host reads are now safe. For a well-conditioned stencil:
// observed(i, 0) ~= 2; observed(i, 1) ~= -3; observed(i, 2) ~= 0.75.
```

Reuse `op` for subsequent field updates. Changing the field values does not require another constructor call. Changing mesh geometry, topology, numbering, caches, or options requires a new matching plan. The operator does not detect arbitrary changes through borrowed arrays.

To select another scheme, change `options` before calling `requirements()` and constructing the data/plan. For example:

```cpp
options.scheme = gradient::Scheme::GreenGaussFace;
options.interpolation = gradient::Interpolation::AuxiliaryWeights;
// Recompute requirements and construct a new matching plan.
```

When a requirement mask is `None`, an empty corresponding device auxiliary view is also valid. The generic example builds empty caches in that case. Keep host cache objects alive on HostSpace, where device storage borrows their arrays.

## Application interfaces

| Method | Inputs and outputs |
| --- | --- |
| `apply_async(input, output, range = {})` | Input `FieldView<T, N>` and mutable output `FieldView<T, 3*N>`. Computes every input component with the ordering above. |
| `apply_component_async(input, x, y, z, range = {})` | One rank-one input view and three rank-one output views. Supports selected, strided components. |
| `apply_set_async(inputs, x, y, z, range = {})` | Four `std::array` objects with the same positive compile-time length. Entry `k` selects input scalar `inputs[k]` and its three output views. Fuses the batch into each algorithm's traversal. |

For one component of an existing multicomponent field:

```cpp
op.apply_component_async(phi.view().component(0),
                   grad_phi.view().component(0),
                   grad_phi.view().component(1),
                   grad_phi.view().component(2));
op.execution_space().fence("Selected gradient ready");
```

For a batch, views can refer to different allocations or components. The following computes two scalar components with different values in one traversal:

```cpp
fields::FieldStorage<Real, 2, Exec, Kokkos::LayoutRight> scalars{
    "scalars", op.required_input_size(), fields::FieldLocation::CellExtnd, exec};
fields::FieldStorage<Real, 6, Exec, Kokkos::LayoutRight> gradients{
    "gradients", owned, fields::FieldLocation::OwnCell, exec};
const auto q = scalars.view();
const auto g = gradients.view();
const auto source = phi.view().as_const();
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

krona::execution::complete(exec, [&] {
    Kokkos::parallel_for("prepare_scalar_batch",
        Policy(exec, 0, op.required_input_size()), KOKKOS_LAMBDA(std::size_t i) {
            q(i, 0) = source(i);
            q(i, 1) = Real{2}*source(i);
        });
    op.apply_set_async(std::array{q.component(0), q.component(1)},
                 std::array{g.component(0), g.component(3)},
                 std::array{g.component(1), g.component(4)},
                 std::array{g.component(2), g.component(5)});
});
```

For this example, `op.apply_async(q, g)` would produce the same component mapping. `apply_set_async()` is useful when the desired inputs or output positions do not form one complete field. Each individual `std::array` has one concrete view type; use compatible view descriptors when collecting components from different layouts. All inputs and outputs in a batch still share one scalar precision.

## Compute a subset of owned cells

`CellRange{begin, end}` is a half-open owned-cell interval. Defaults are `begin = 0`, `end = -1`; `-1` resolves to `n_own`.

```cpp
const krona::LocalIndex split = host_mesh.n_own / 2;
op.apply_async(phi.view(), grad_phi.view(), {0, split});
op.apply_async(phi.view(), grad_phi.view(), {split, -1});
op.execution_space().fence("Both owned ranges ready");
```

Ranges must satisfy `0 <= begin <= end <= n_own`. An empty range submits no kernels. Views must still satisfy the full input and owned-output extent contracts, even for partial or empty ranges. Input neighbours may lie outside the selected range.

The cell schemes traverse only selected cell rows. `GreenGaussFace` clears only the selected cells but scans all local faces for each nonempty range call; splitting its application into many ranges repeats that scan. Output entries outside the selected interval are preserved.

## Project gradients for explicit corrections

`correction.hpp` provides a numerical projection, not a boundary condition or a diffusion flux. It reads existing gradients and applies a caller-supplied geometric vector. It does not compute or exchange gradients itself.

Create a borrowed reader from an existing `3*N` gradient field:

```cpp
#include <krona/numerics/gradient/correction.hpp>

const auto gradients = gradient::cell_gradients(grad_phi.view()); // Scalar component 0.
gradients.template validate<Exec>(owned);
const gradient::GradientCorrection correction{gradients};
```

For scalar component `c`, use `cell_gradients(field, c)`: it selects field components `3*c`, `3*c+1`, `3*c+2`. Alternatively, `cell_gradients(gx, gy, gz)` accepts three rank-one Kokkos views, including strided views. The adapter stores read-only `fields::ComponentDescriptor`s and the minimum input extent; it does not retain allocation ownership. All three components share a memory space, may have different floating-point types, and are read in their common scalar type. `size()` returns the accessible cell count. Optional `validate<Exec>(count)` checks accessibility, extent and pointer/stride descriptors locally without fencing or inspecting values.

For a cell gradient, the projection is:

$$
\delta_P=\mathbf t\cdot\nabla\phi_P.
$$

Evaluate it inside your own kernel:

```cpp
using CorrectionPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
Kokkos::View<Real*, typename Exec::memory_space> increments("gradient/increments", owned);
Kokkos::parallel_for("project_gradients", CorrectionPolicy(exec, 0, owned),
    KOKKOS_LAMBDA(std::size_t cell) {
        increments(cell) = correction.increment(cell, Real{0.25}, Real{-0.5}, Real{0.125});
    });
// For the affine field above, every result is approximately 2.09375.
// Keep increments and gradient storage alive until this kernel finishes.
```

For an interior face, `interior(owner, neighbour, owner_weight, tx, ty, tz)` returns:

$$
\delta_f=\mathbf t\cdot
\left[w\nabla\phi_P+(1-w)\nabla\phi_N\right].
$$

`owner_weight` multiplies the **owner** gradient. The caller supplies the actual interpolation weight; there is no automatic midpoint, clamping or geometry lookup.

| Method | Inputs and result |
| --- | --- |
| `increment(cell, tx, ty, tz)` | One cell index and a geometric vector; returns its dot product with that cell's gradient. |
| `interior(owner, neighbour, w, tx, ty, tz)` | Two cell indices, owner weight and geometric vector; returns the projection of the interpolated gradient. |
| `operator()(context)` | Structural adapter: reads `context.owner`, `tx`, `ty`, `tz`, then calls `increment`. |
| `interior(context, tx, ty, tz)` | Structural adapter: reads `context.owner`, `neighbour`, `owner_weight`. |
| `NoCorrection` | Same evaluation forms, returning zero without a gradient provider. |

A boundary tangential increment uses `t = x_face - x_owner - h*n`, where `n` is the unit outward normal and `h = (x_face - x_owner).dot(n)`. An interior nonorthogonal diffusion projection can use `t = A*(n - d/(d.dot(n)))`, where `d = x_neighbour - x_owner`, `n` is the unit owner-to-neighbour normal, `A` is the face area and the denominator is nonzero. These vectors have different units: the caller chooses the vector and applies any transport coefficient, flux sign or other scaling. No area or coefficient is added by the correction policy itself. Boundary and diffusion modules retain their existing implementations until explicitly migrated to this interface.

`GradientCorrection` can also hold a custom device-callable provider with the interface `gradients(cell)[axis]`, axes `0,1,2`. Such a provider may read another layout or compute gradients on demand; it need not use `CellGradients` or allocate storage. Its validity and lifetime are caller contracts.

All accessed indices must be in range; gradient components, geometric vectors and interpolation weights must be finite. Evaluation does not validate or clamp numerical data and may overflow like ordinary arithmetic. A consuming operator must check the resulting increment according to its error policy. Method scalar arguments use a floating-point type; their `Real` determines projection arithmetic and the result type, and gradient values are converted to it. If an interior neighbour is an MPI ghost, exchange its gradient before evaluation; owned gradients alone are insufficient.

The reader and correction are passed by value into kernels. Construction performs no field-data copies, allocations, kernel launches or MPI. Evaluation runs within the caller's kernel, with no additional submission or synchronization. Order gradient production and correction consumption on the same `Exec` instance, or synchronize their dependency explicitly. Keep all backing allocations alive at stable addresses until every consumer finishes.

## Precision and plan storage

`GradientOperator<Exec, Coefficient, Accumulator>` has three independent choices:

| Type | Role |
| --- | --- |
| Field value type `T` | Input/output storage precision, selected by the views passed to each call. |
| `Coefficient`, default `GeometryReal` | Stored least-squares coefficient precision. Least-squares preparation itself uses `double` intermediates. |
| `Accumulator`, default `AccumulationReal` | Contribution arithmetic and cell-local accumulation precision. |

For `GreenGaussFace`, each contribution is converted to the output type before an atomic addition. A wider `Accumulator` therefore does not turn the output array into a wider accumulation buffer.

The plan borrows compact mesh/auxiliary descriptors and owns only its least-squares coefficients. Green–Gauss stores no coefficient array. Least squares stores three coefficients per owned-cell/face incidence, shared by all field components and later calls. `coefficient_bytes()` reports the coefficient view's storage span in bytes; it is zero for Green–Gauss and empty least-squares plans.

`scheme()`, `name()`, `required_input_size()`, `coefficient_bytes()`, and `execution_space()` inspect the plan without submitting computation. The operator is neither copyable nor movable. Application calls do not allocate per-field work buffers or retain the supplied field bindings for later calls.

## Asynchrony and lifetime

| Operation | Completion behaviour |
| --- | --- |
| `requirements(options)` | Host-only mask query; no kernels or allocation. |
| `GradientOperator(...)` | Blocking local setup. Validates descriptors/topology/required geometry, prepares least-squares coefficients if selected, and fences `exec` before returning. |
| `apply_async()`, `apply_component_async()`, `apply_set_async()` | Validate descriptors/range on the host, then submit kernels to the stored execution-space instance. No final fence; output values may still be pending on return. |
| `cell_gradients(...)`, reader `validate<Exec>(count)` | Host descriptor setup/checks only; no allocation, kernels or completion wait. |
| `GradientCorrection` / `NoCorrection` evaluation | Local device-callable arithmetic inside the caller's kernel; no submission, synchronization or MPI. |
| `execution_space().fence()` | Waits for work on that instance. |

The operator performs no MPI communication, halo exchange, or barrier during either construction or application. A local fence does not wait for other ranks. A synchronous backend may complete kernels during submission; the application methods provide no portable completion guarantee on return.

Queue producers, gradient kernels, and consumers on the same execution-space instance for ordering. Synchronize explicitly before dependent work on another instance or conflicting host/MPI access. If ghost gradients are needed, compute owned gradients into a sufficiently large output field, then use a separate halo exchanger with the required completion ordering.

Keep the plan, mesh/cache backing arrays, and field buffers alive until all submitted work using them is complete. On HostSpace, this includes host mesh/cache arrays borrowed by device storage. There is no destructor fence: finish outstanding work before destroying the plan or releasing its data. Do not modify topology, geometry, or cached metrics while their kernels are pending.