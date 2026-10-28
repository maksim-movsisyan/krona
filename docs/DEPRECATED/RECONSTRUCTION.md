# Reconstruction

KRONA's reconstruction module evaluates cell-based profiles at a face point. It supports first-order states, linear and quadratic reconstruction, MUSCL/TVD profiles, explicit three-point QUICK stencils, and precomputed cell limiter factors. Several variables can be reconstructed together with one geometry or stencil descriptor.

The module does not own solution fields, compute gradients or Hessians, exchange halo data, impose physical boundary conditions, or evaluate numerical fluxes. These remain responsibilities of the calling solver and the corresponding KRONA modules.

## 1. Headers and execution model

```cpp
#include <krona/numerics/reconstruction/reconstruction.hpp>
// Optional integration with KRONA mesh views:
#include <krona/numerics/reconstruction/mesh_stencil.hpp>

namespace rec = krona::numerics::reconstruction;
```

| Header | Contents |
| --- | --- |
| `types.hpp` | Geometry descriptors, tensors, input bundle, component coupling, ranges |
| `pack.hpp` | Borrowed component packs and field/gradient adapters |
| `limiters.hpp` | TVD ratio limiters and cell-bound limiter policies |
| `schemes.hpp` | Reconstruction policies and explicit QUICK stencil builders |
| `reconstruction.hpp` | Point evaluation and asynchronous cell limiter preparation |
| `mesh_stencil.hpp` | Optional adapter for grouped mesh and cell-to-face connectivity |

The core depends on KRONA execution types and Kokkos. Only the optional mesh adapter depends on mesh views. Field adapters use the public structure of a field view without including the fields module.

| Operation | Where it is called | Work performed |
| --- | --- | --- |
| `pack`, `field_pack`, `gradient_pack`, `gradients`, `data` | Host setup | Construct small descriptors; no array allocation, kernel, or synchronization |
| `MeshStencil` construction | Host setup | Borrow arrays and check descriptors; no device computation |
| `evaluate`, `evaluate_side` | Inside the caller's kernel | Compute states at one point; no kernel submission, allocation, MPI, or fence |
| QUICK stencil builders | Host or device | Compute three interpolation weights from supplied geometry |
| `compute_limiters_async` | Host | Check descriptors and submit one cell-parallel kernel into the supplied execution-space instance |

The evaluation functions are `KOKKOS_INLINE_FUNCTION` functions. Calling `evaluate` does not launch a parallel loop: the caller chooses the loop, execution policy, output storage, and subsequent flux calculation.

## 2. Values, gradients, and batches

### Borrowed component packs

`pack` combines one or more rank-one Kokkos views:

```cpp
// Existing rank-one views, already populated in an accessible memory space.
auto values = rec::pack(pressure, velocity_x, velocity_y, velocity_z, temperature);

// values(cell, 0) = pressure(cell)
// values(cell, 4) = temperature(cell)
static_assert(decltype(values)::components == 5);
```

A `ComponentPack<T, N, Space>` stores a pointer and an element stride for each component. It neither allocates nor owns the underlying arrays. Even when the original Kokkos views are managed, the pack does not retain their allocation trackers. Keep the backing owners alive until all dependent kernels and exchanges have completed.

Components in a single pack must have the same floating-point base type and memory space. They may have different strides and extents; the pack's `size()` is the minimum extent. If any input component is const, the entire resulting pack is read-only. Strided subviews are supported without repacking data.

This allows batches assembled from separate scalar fields, components of vector fields, or a mixture of both. The component order is explicit and is preserved in every returned state.

### KRONA field adapters

For an existing KRONA field view:

```cpp
auto values = rec::field_pack(state_view);
auto gradients = rec::gradient_pack(state_gradient_view);
auto input = rec::data(values, gradients);
```

`field_pack` includes every component of the field in its original order. Its structural interface is `value_type`, `memory_space`, static `components`, `size()`, and `component(c)`.

`gradient_pack` expects three stored components per reconstructed variable, in the gradient module's order:

```text
variable c: 3*c + 0 = d/dx, 3*c + 1 = d/dy, 3*c + 2 = d/dz
```

For example, five variables require a fifteen-component gradient field. This adapter does not calculate gradients.

Alternatively, assemble the directional derivatives explicitly:

```cpp
auto values = rec::pack(p, u, v, w, T);
auto gradients = rec::gradients(
    rec::pack(px, ux, vx, wx, Tx),
    rec::pack(py, uy, vy, wy, Ty),
    rec::pack(pz, uz, vz, wz, Tz));
auto input = rec::data(values, gradients);
```

`gradients(cell, c)` returns `rec::Vector<...>{gx, gy, gz}`. Values and gradients must describe the same variables in the same order. Their storage precisions may differ; the same-type requirement applies within each individual component pack.

### The input bundle

```cpp
auto input = rec::data(values, gradients, factors, hessians, moments);
```

Arguments after `values` have defaults. Supply only the objects used by the chosen scheme:

| Accessor | Device interface | Meaning |
| --- | --- | --- |
| `values` | `(cell, component)` | Base cell value |
| `gradients` | `(cell, component)` | Three gradient components |
| `factors` | `(cell, component)` | Precomputed factor in `[0, 1]` |
| `hessians` | `(cell, component)` | True second derivatives in a symmetric tensor |
| `moments` | `(cell)` | Central geometric second-moment tensor |

`values` declares static `components`. Factors declare either the same component count or one component, which is broadcast to every variable. The default `UnitFactors` returns one; using a `Limited` policy without supplying computed factors therefore leaves the underlying increment unchanged.

`Data` stores its accessors by value. The standard packs are borrowed descriptors; a custom accessor determines its own ownership behavior. Device-captured accessors and policies must be usable in the selected execution space.

Evaluation uses `ComputeReal` by default. An explicit template argument selects the arithmetic and result type independently of storage:

```cpp
auto states = rec::evaluate<double>(scheme, face, input);
```

The module does not automatically convert an entire field or create a second storage buffer for this choice.

### Meaning of a stored value

For finite-volume use, the base values are normally cell averages. Linear reconstruction about a volume centroid preserves the average of an affine profile. At quadratic order, the distinction between an average and a point sample matters explicitly: use the moment correction in `Quadratic` or the cell-average QUICK builder. The point-valued QUICK builder has a separate contract.

The accessors carry numbers, not a declaration of their mathematical interpretation. In particular, deriving a primitive variable from averaged conservative variables does not automatically make it the exact cell average of that primitive variable. The caller must choose a consistent reconstruction model for the stored quantities.

## 3. Geometry and the two sides of a face

`CellSide<Real>` describes reconstruction from one cell to one target point:

| Member | Meaning |
| --- | --- |
| `cell` | Current cell whose profile is evaluated |
| `other` | Opposite value sample, when required by the scheme |
| `offset` | `x_face - x_cell` |
| `span` | `x_other - x_cell` for a cell-to-cell interior stencil |
| `fraction` | Position along `span`; a negative value requests automatic projection |

Here, `cell` means the current reconstruction side. It is the owner for the left state and the neighbour for the right state.

Build both sides together:

```cpp
const auto face = rec::face_geometry<Real>(
    owner, neighbour,
    owner_to_face, neighbour_to_face,
    owner_fraction);
```

The helper sets the right-side cell to the neighbour, swaps the opposite sample, reverses `span`, and uses `1 - owner_fraction` when the fraction was supplied explicitly. Each side retains its own displacement to the same physical face point.

For a linear profile:

\[
q_L = \bar q_P + \nabla q_P\cdot(x_f-x_P),\qquad
q_R = \bar q_N + \nabla q_N\cdot(x_f-x_N).
\]

The neighbour gradient is not negated. Its displacement already has the appropriate direction. Left/right labels follow owner/neighbour ordering and do not select the upwind state; a flux calculation makes that decision later.

### Two fraction conventions

With a negative `fraction`, MUSCL/TVD policies compute the orthogonal projection

\[
\alpha = \frac{(x_f-x_C)\cdot d}{d\cdot d},\qquad d=x_D-x_C.
\]

An explicit fraction can instead describe the intersection of the centre line with the face plane:

\[
\alpha = \frac{(x_f-x_C)\cdot n_f}{d\cdot n_f}.
\]

The mesh adapter uses the latter convention. These fractions can differ on a skew mesh. Use a consistent convention when preparing limiter factors and evaluating the corresponding reconstruction. MUSCL/TVD require a nondegenerate span and a fraction in `[0, 1]`; these conditions have debug assertions, not a general runtime geometry validator.

## 4. Available policies and required inputs

Every policy returns an increment. The common evaluator adds the base value:

\[
q_f = \bar q_C + \Delta q_C.
\]

| Policy | Additional inputs beyond values | Precomputed cell factors? |
| --- | --- | --- |
| `FirstOrder` | None | No |
| `SecondOrderUpwind` | Gradient and target displacement | No |
| `LimitedLinear` | Linear inputs | Yes, or the default unit factors |
| `MusclKappa<>` | Gradient, opposite value, span, fraction | No |
| `Tvd<>` | Gradient, opposite value, span, fraction | No; evaluates its ratio limiter locally |
| `Quadratic` | Gradient, Hessian, central cell moments | No |
| `LimitedQuadratic` | Quadratic inputs | Yes, or unit factors |
| `Quick` | Explicit three-point indices and weights | No |
| `Limited<Scheme>` | Inputs of `Scheme` | Yes, or unit factors |

`requirements` is a compile-time bit mask used by the preparation code and mesh adapter. `Values` is zero because all reconstructions already require base values. A scheme's name alone does not establish the order, boundedness, or stability of a complete finite-volume discretization.

### First order and linear reconstruction

`FirstOrder` returns zero increment. `SecondOrderUpwind` uses

\[
\Delta q_C = \nabla q_C\cdot r,\qquad r=x_f-x_C.
\]

The linear policy contains no limiter. Its accuracy depends on the supplied gradient. It also performs no upwind selection: either side can be evaluated with this same profile.

### Unlimited kappa-MUSCL

Let `G = gradient(cell) dot span` and `jump = q_other - q_cell`. The implemented profile is

\[
\Delta q_C = \alpha[(1-\kappa)G+\kappa\,\mathrm{jump}]
              +\Delta q_{\rm skew}.
\]

```cpp
auto default_muscl = rec::MusclKappa<>{}; // kappa = 1/3
auto linear_muscl = rec::MusclKappa{Real{0}};
```

`kappa` controls the profile; it is not a limiter. With the default skew correction, `kappa = 0` gives the same increment as this module's gradient-based `SecondOrderUpwind`.

The familiar uniform-grid interpretations of `kappa = 1/3` for cell averages and `kappa = 1/2` for point-valued QUICK also depend on the gradient and midpoint stencil used. They do not establish general quadratic exactness on an unstructured mesh. Use the explicit QUICK or quadratic contracts when that distinction matters.

### TVD ratio reconstruction and Darwish–Moukalled

The default `Tvd<>` policy uses a Van Leer ratio limiter and the Darwish–Moukalled virtual upstream construction:

\[
r_{\rm TVD}=\frac{2G}{\mathrm{jump}}-1,\qquad
\Delta q_C=\alpha\,\psi(r_{\rm TVD})\,\mathrm{jump}
              +\Delta q_{\rm skew}.
\]

This ratio corresponds to an estimated virtual upstream value `q_U = q_D - 2*G`. It needs the current gradient and the opposite value, without searching for an upstream cell. If `jump` is exactly zero, `Tvd` skips the ratio and returns only the skew correction.

Available ratio limiters are:

| Type in `rec::tvd` | Role |
| --- | --- |
| `Minmod` | Ratio limiter |
| `Superbee` | Ratio limiter |
| `MonotonizedCentral` | Ratio limiter |
| `VanLeer` | Default ratio limiter |
| `VanAlbada` | Ratio limiter |
| `Koren` | Ratio limiter |

```cpp
using Scheme = rec::Tvd<rec::tvd::Minmod>;
Scheme scheme;

using LineOnly = rec::Tvd<rec::tvd::VanLeer,
                          rec::DarwishMoukalled,
                          rec::NoSkewCorrection>;
```

These functors compute `psi(r)`. They are different from the cell-bound limiter policies passed to `compute_limiters_async`, and are not interchangeable with them.

### Skew correction

`LinearSkewCorrection`, the default for MUSCL/TVD, adds

\[
\Delta q_{\rm skew}=\nabla q_C\cdot(r-\alpha d).
\]

It transfers the profile from the selected point on the centre line to the actual face point. Using the current cell gradient preserves affine fields exactly when that gradient is exact. For a smooth nonlinear field it approximates the gradient at the centre-line point.

The ratio limiter acts on the line-based term, not on this additional correction. Consequently, a TVD ratio limiter alone does not bound the full multidimensional reconstruction. `NoSkewCorrection` removes this term but also removes the corresponding affine-field correction on skew faces. A cell limiter can be applied to the entire candidate increment through `Limited<Scheme>`.

## 5. Evaluation inside a caller kernel

The following example uses caller-provided arrays containing two cells and reconstructs one face. All input arrays must already contain valid data. The output views are allocated by the caller; `evaluate` itself allocates no field storage.

```cpp
using Exec = krona::execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using Real = krona::ComputeReal;

// q, gx, gy, gz: existing rank-one Kokkos views with at least two cells.
auto input = rec::data(
    rec::pack(q),
    rec::gradients(rec::pack(gx), rec::pack(gy), rec::pack(gz)));

const auto face = rec::face_geometry<Real>(
    0, 1, rec::Vector<Real>{Real{0.5}, 0, 0},
          rec::Vector<Real>{Real{-0.5}, 0, 0}, Real{0.5});

Kokkos::View<Real**, Space> left("left", 1, 1);
Kokkos::View<Real**, Space> right("right", 1, 1);
Exec exec;
Kokkos::parallel_for("reconstruct_one_face",
    Kokkos::RangePolicy<Exec>(exec, 0, 1),
    KOKKOS_LAMBDA(const int f) {
        const auto state = rec::evaluate<Real>(
            rec::SecondOrderUpwind{}, face, input);
        left(f, 0) = state.left[0];
        right(f, 0) = state.right[0];
    });
```

`evaluate` returns `FaceValues<Real, N>` with fixed-size `left` and `right` arrays. `evaluate_side` returns one fixed-size array for one descriptor. In a flux kernel, consume these arrays immediately instead of storing all face states when separate storage is unnecessary.

For a batch, the same call reconstructs every component in pack order:

```cpp
// input contains the five-variable pack shown earlier.
const auto state = rec::evaluate<Real>(rec::Tvd<>{}, face, input);
// state.left[0]: pressure; state.left[1..3]: velocity; state.left[4]: temperature.
```

Component expansion in evaluation is compile-time. Geometry can be shared across variables, while gradient and value data remain component-specific. Large batches can increase register pressure; choose batch size using measurements on the target backend.

## 6. Mesh integration

Construct the optional adapter from existing views:

```cpp
// mesh_view: krona::mesh::compute::MeshView<Space>
// connectivity_view: krona::mesh::compute::AuxConnectivityView<Space>
rec::MeshStencil<Space, Real> stencil(mesh_view, connectivity_view);
```

The adapter borrows the grouped mesh and owned-cell `CellFaces` connectivity. It requires face owner/neighbour indices, cell centroids, face centroids, and face normals. Its constructor checks the relevant descriptors and extents; the underlying connectivity and geometry must already be valid.

It does not require cached interpolation weights, projected distances, or area-over-distance arrays. Displacements and the face-plane fraction are calculated from the borrowed geometry when used.

The adapter provides:

| Method | Meaning |
| --- | --- |
| `size()` | Number of owned cells |
| `required_input_size()` | Required value extent, including extended boundary slots |
| `size(cell)` | Number of incident faces for an owned cell |
| `operator()(cell, k)` | Opposite sample and reconstruction side for the incident face |
| `face<Scheme>(f)` | Two-sided geometry for an inner face |
| `boundary_side(f)` | One-sided geometry from a boundary owner to the face |

For interior and MPI-interface faces:

```cpp
using Scheme = rec::Tvd<>;
const auto n_inner = static_cast<std::size_t>(mesh_view.sizes.n_inner_faces);
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

Kokkos::parallel_for("reconstruct_inner_faces", Policy(exec, 0, n_inner),
    KOKKOS_LAMBDA(const std::size_t f) {
        const auto face = stencil.template face<Scheme>(f);
        const auto state = rec::evaluate<Real>(Scheme{}, face, input);
        // Consume state.left and state.right in the numerical flux calculation.
    });
```

Pass the actual scheme to `face<Scheme>`: the adapter uses its requirements to load the needed geometry and compute a fraction for policies using opposite values. `face<FirstOrder>` reads only endpoint indices during evaluation, although constructing `MeshStencil` still requires the full geometry descriptors listed above.

`face<Quick>` is not supported by this adapter: QUICK requires explicit upstream samples and weights rather than `CellSide` geometry.

### Extended boundary samples

For the grouped mesh, values used in limiter preparation follow this indexing:

```text
[0, n_cells)                         owned cells and real halo cells
[n_cells, n_cells + n_boundary_faces) boundary-face values
boundary sample = n_cells + face - n_inner_faces
```

These boundary samples are values located at faces, not additional ghost-cell centres. The adapter uses `span = offset` and `fraction = 1` for such a sample. `required_input_size()` is therefore `n_cells + n_faces - n_inner_faces`, even if the chosen scheme itself does not read opposite values: limiter preparation still samples them to determine bounds.

Before preparing factors, populate the required extended boundary slots through the appropriate boundary module. This adapter does not apply boundary conditions.

For an interior trace at a physical boundary:

```cpp
const auto side = stencil.boundary_side(boundary_face);
const auto interior_trace = rec::evaluate_side<Real>(
    rec::SecondOrderUpwind{}, side, input);
```

This reconstructs the owner profile at the boundary face. It does not produce an exterior state or replace a physical boundary closure. If the selected policy reads `side.other`, that extended face sample must also be current.

## 7. Precomputed cell limiter factors

`Limited<Scheme>` scales the complete increment of its underlying policy:

\[
q_f = \bar q_C + \theta_{C,c}\,\Delta q_{C,c},\qquad 0\leq\theta_{C,c}\leq1.
\]

`LimitedLinear` and `LimitedQuadratic` are aliases. A zero factor returns the base value without evaluating the underlying high-order increment.

### Preparation algorithm and interface

```cpp
rec::compute_limiters_async<Real>(
    exec, stencil, input, output,
    limiter, unlimited_scheme, coupling, range);
```

The last four arguments default to `BarthJespersen{}`, `SecondOrderUpwind{}`, `Componentwise{}`, and all owned cells.

For each selected owned cell, the kernel first obtains minimum and maximum values from the base cell and every stencil sample, for the whole component batch. It then evaluates the unlimited candidate increment at every stencil target, applies the cell-bound limiter, and takes the minimum factor for each component. An empty neighbourhood leaves its factors at one.

Prepare factors with the same unlimited scheme, parameters, geometry convention, and evaluation points that will later be consumed. Preparing factors for a linear increment does not bound an unrelated quadratic or skew-corrected candidate. The preparation function rejects a policy whose requirements already include `Factors`.

### Limited linear reconstruction

```cpp
// values includes owned, halo, and extended boundary samples.
// gradients contains the current gradients needed by this stage.
auto unlimited_input = rec::data(values, gradients);

// Caller owns theta_storage. Include halo rows if remote-side evaluation needs them.
auto theta = rec::field_pack(theta_view); // Same component count as values.

rec::compute_limiters_async<Real>(exec, stencil, unlimited_input, theta);

auto limited_input = rec::data(values, gradients, theta);
using Scheme = rec::LimitedLinear;
// A following kernel on exec may evaluate Scheme{} using limited_input.
// For MPI faces, complete the required gradient/factor exchange before remote reads.
```

The preparation writes owned rows only. It does not initialize or communicate a halo tail. The backing storage of `theta_view`, values, gradients, and the mesh must outlive all work using these descriptors.

### Componentwise and shared factors

`Componentwise` computes a separate factor for each variable. Its output must have exactly the input component count.

`SharedMinimum` takes the minimum over the entire batch. The output may have one component, broadcast by `Data::factor`, or the full component count, in which case every component receives the same factor:

```cpp
auto theta_shared = rec::pack(theta_scalar); // Rank-one writable view.
rec::compute_limiters_async<Real>(
    exec, stencil, unlimited_input, theta_shared,
    rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::SharedMinimum{});

auto limited_input = rec::data(values, gradients, theta_shared);
```

A shared factor can preserve a common scaling of the candidate increments, but it is not an equation-of-state admissibility check, characteristic limiter, or general positivity-preserving method.

### Cell-bound policies

`BarthJespersen` is the default. For each candidate, it limits the increment to the available change toward the stencil minimum or maximum. With valid finite inputs, its bound applies to the tested target points of that candidate profile. It does not establish bounds everywhere inside a cell or at unlisted face quadrature points.

`Venkatakrishnan` uses a smooth factor with optional regularization:

```cpp
auto smooth = rec::Venkatakrishnan{
    rec::ConstantRegularization<Real>{Real{1e-6}}};
rec::compute_limiters_async<Real>(
    exec, stencil, unlimited_input, theta, smooth);
```

Regularization epsilon has the same units as the reconstructed variable; its square enters the formula. A nonzero epsilon can relax local bounds, so this is not the same strict bound contract as Barth–Jespersen. Even at zero epsilon, this implementation may reduce an already bounded increment: an allowed change equal to the candidate magnitude gives a factor of `3/4`.

The default regularization is zero. A custom `regularization(context)` can choose epsilon by cell and component without requiring an additional array. For variables with different physical units, choose a component-appropriate regularization rather than assuming a single constant has the same meaning for every component.

### Limiting a complete TVD candidate

The ratio limiter and cell limiter can be composed deliberately:

```cpp
using Base = rec::Tvd<>;
const Base base{};

rec::compute_limiters_async<Real>(
    exec, stencil, unlimited_input, theta,
    rec::BarthJespersen{}, base);

const rec::Limited<Base> scheme{base};
auto limited_input = rec::data(values, gradients, theta);
// Evaluate scheme with limited_input at the same prepared target points.
```

Here, the cell factor scales the full TVD candidate, including its skew correction. Whether this extra limiting is desirable is a discretization choice made by the solver.

### Ranges and descriptor checks

`CellRange{begin, end}` is half-open. Only those owned rows are written. The default end means `stencil.size()`.

Even for a partial range, output storage must cover the full owned-cell count, and values must cover `stencil.required_input_size()`. Gradient extent checks are enabled by the `Gradients` requirement. Hessian and moment checks are enabled together by `Hessians`, as used by `Quadratic`. When those accessors expose `size()`, they are checked against the full owned count. A custom accessor should report its number of addressable cells through this hook.

The host function checks Kokkos lifetime, ranges, extents, output component count and mutability, and memory accessibility where an accessor exposes `memory_space`. It does not read device arrays to validate their numerical contents. Some per-entry conditions use `KOKKOS_ASSERT`; these must not be treated as release-mode validation. Accessors without metadata cannot receive those metadata checks.

## 8. Explicit QUICK stencils

QUICK uses a `ThreePointStencil<Real>` containing three explicit value indices, three weights, the base cell index, and `valid`. Samples are ordered upstream, centre, downstream; `cell` is the centre sample. The module does not locate these cells on an unstructured mesh.

### Cell-average interpretation

```cpp
auto side = rec::cell_average_quick_stencil<Real>(
    samples, centres, variances, target);
```

| Argument | Meaning |
| --- | --- |
| `samples` | Three indices into the value accessor |
| `centres` | Mean positions of those cells in one selected scalar coordinate `s` |
| `variances` | Central second moments `average((s - centre)^2)` in that coordinate |
| `target` | Face-point coordinate in the same coordinate system |
| optional `tolerance` | Relative determinant threshold; default `64 * epsilon<Real>` |

The weights fit a quadratic polynomial to the three cell averages. They satisfy

\[
\sum_i w_i=1,\qquad \sum_i w_i s_i=s_f,\qquad
\sum_i w_i(s_i^2+\sigma_i^2)=s_f^2.
\]

For uniform cells of width `h`, centres `(-h, 0, h)`, variance `h*h/12`, and target `h/2`, the weights are `(-1, 5, 2)/6`.

`point_quick_stencil(samples, centres, target)` explicitly uses point-value interpretation by setting all variances to zero. Its corresponding uniform midpoint weights are `(-1, 6, 3)/8`. These are different data contracts; point weights do not give the same quadratic exactness when applied to cell averages.

### Two-sided example

For a face at `s = 0.5` between cells centred at zero and one, select a different upstream cell for each side:

```cpp
const Kokkos::Array<Real, 3> variances{
    Real{1}/12, Real{1}/12, Real{1}/12};

const auto left = rec::cell_average_quick_stencil<Real>(
    Kokkos::Array<std::size_t, 3>{0, 1, 2},
    Kokkos::Array<Real, 3>{-1, 0, 1}, variances, Real{0.5});
const auto right = rec::cell_average_quick_stencil<Real>(
    Kokkos::Array<std::size_t, 3>{3, 2, 1},
    Kokkos::Array<Real, 3>{2, 1, 0}, variances, Real{0.5});

if (!left.valid || !right.valid)
    throw std::invalid_argument("Invalid QUICK geometry");

const rec::FaceStencil<rec::ThreePointStencil<Real>> face{left, right};
auto input = rec::data(values); // At least the four selected samples.
// Inside the caller's kernel:
const auto state = rec::evaluate<Real>(rec::Quick{}, face, input);
```

These descriptors can be reused across every component in a batch. QUICK evaluation reads values and precomputed weights; it does not require stored gradients or Hessians.

Invalid, nonfinite, or degenerate builder input returns `valid == false` and NaN weights. There is no silent first-order fallback. Check validity before submission and choose an explicit fallback or error policy in the application. The builder does not validate value-array indices; `Quick` itself only asserts stencil validity in debug builds.

This fit is exact for quadratic polynomials in the selected coordinate, not arbitrary three-dimensional quadratics. Projected variances depend on cell geometry. If a central tensor `M` is available, the variance along a unit direction `e` is `e^T M e`; otherwise the caller must obtain the required geometry separately.

`Limited<Quick>` is possible, but factor preparation then needs a custom stencil whose entries supply the corresponding three-point descriptors and whose samples define the intended bounds. The built-in `MeshStencil` does not construct this wider stencil or communicate its upstream values.

## 9. Mean-preserving quadratic reconstruction

For volume-averaged data, `Quadratic` evaluates

\[
q_f=\bar q_C+g_C\cdot r+
\tfrac12 H_C:(rr^T-M_C),\qquad
M_C=\frac1{V_C}\int_{V_C}(x-x_C)(x-x_C)^T\,dV.
\]

Here `x_C` is the volume centroid, `g_C` is the gradient there, and `H_C` contains true second derivatives. Subtracting `M_C` makes the quadratic contribution average to zero, so the polynomial retains the supplied cell average. This is a Taylor polynomial with supplied derivatives and geometry, not a fit assembled from three neighbouring values.

`SymmetricTensor<Real>` stores `xx, yy, zz, xy, xz, yz`. Off-diagonal entries are stored once; evaluation includes their factor of two in the tensor contraction. Do not supply Taylor coefficients already divided by two as Hessian entries.

Supply device-callable accessors with these signatures:

```cpp
// hessians(cell, component) -> rec::SymmetricTensor<...>
// moments(cell)            -> rec::SymmetricTensor<...>
auto quadratic_input = rec::data(
    values, gradients, rec::UnitFactors{}, hessians, moments);

// Inside the caller's kernel:
const auto state = rec::evaluate<Real>(rec::Quadratic{}, face, quadratic_input);
```

Moments are geometric and shared by every variable in a cell; Hessians are variable-dependent. A full stored symmetric moment tensor costs six scalar values per cell, rather than six per variable. The module does not allocate this storage or compute either moments or Hessians. First-order, linear, MUSCL, and TVD policies do not need these moments.

For limited quadratic evaluation, prepare factors using `Quadratic{}`, then supply those factors in place of `UnitFactors{}` and evaluate `LimitedQuadratic{}` at the same points. Accurate derivatives and moments establish the reconstruction's polynomial contract; face integration, flux evaluation, and time integration still determine the accuracy of the complete solver.

## 10. Asynchronous ordering and distributed use

`compute_limiters_async(exec, ...)` submits work to that execution-space instance. It does not fence, copy results to the host, allocate work arrays, or perform MPI communication. A backend may execute work synchronously, but callers should preserve the asynchronous contract.

For kernels submitted to the same execution-space instance, preserve this logical order:

1. Populate values, required boundary samples, and the derivatives used by the candidate.
2. Prepare the owned-cell factors.
3. Reconstruct states and consume them in subsequent kernels.

No fence is required between dependent kernels ordered on the same instance. Different execution-space instances, host reads, communication, and storage reuse need an explicit dependency or synchronization according to the participating APIs. A host-side call returning does not establish that an asynchronous factor kernel has completed.

For partition interfaces, there are two distinct dependencies:

| Stage | Data that must already be current |
| --- | --- |
| Owned factor preparation | All sampled values, including halo and extended boundary values; owned derivatives and moments required by the candidate |
| Reconstruction of a remote side | Remote cell values and every remote derivative, factor, or moment used by that policy |

The module never exchanges these arrays. Follow the existing halo module's producer/communication/consumer contract. In particular, owned factors must be ready before communication reads them, and received factors must be available before reconstructing a neighbour profile that uses them. A wider QUICK stencil may require values beyond the normal immediate-neighbour halo.

Keep all backing allocations alive and stable until work and dependent communication finish. Do not resize, replace, or modify arrays while kernels read them. Mesh geometry must remain unchanged for the duration of the submitted work.

Factors and geometric stencil weights have no revision tracking. Recompute factors when their input values, derivatives, candidate parameters, or evaluation geometry change. Refresh QUICK weights when their defining geometry changes. Exchanging stale factors does not make them valid for a new reconstruction stage.

## 11. Extension interfaces

A custom reconstruction policy supplies a requirements mask and a device-callable increment:

```cpp
struct CustomLinear {
    static constexpr unsigned requirements = rec::Gradients;

    template<class R, class Side, class D>
    KOKKOS_INLINE_FUNCTION R increment(
        const Side& side, const D& input, std::size_t component) const
    {
        return rec::dot_as<R>(input.gradients(side.cell, component), side.offset);
    }
};
```

Do not add the base value inside `increment`; the evaluator does that. A policy may require a different side descriptor, as `Quick` does. `FaceStencil<Left, Right>` allows distinct descriptor types for the two sides, provided the selected policy supports both.

A custom cell-bound limiter accepts `LimiterContext<Real>{cell, component, value, minimum, maximum, increment}` and returns a finite factor in `[0, 1]`. A custom TVD ratio policy accepts `(gradient_along_span, jump)`; a ratio limiter accepts the resulting scalar ratio.

A custom factor-preparation stencil supplies:

```text
size()                  number of owned cells
required_input_size()   required value extent
size(cell)              number of entries for that cell
operator()(cell, k)     entry with .sample and .side
```

`sample` participates in the bounds; `side` describes a candidate evaluation point. All cells/samples and derivative accesses must be valid for the supplied accessors. Expose `memory_space` and cell-count `size()` metadata where applicable so the host-side checks can inspect them.

## 12. Current limitations and possible extensions

Current contracts to keep in mind:

- The module evaluates point states. It does not integrate a reconstructed profile over a face, choose a Riemann solver, or assemble a transport equation.
- Reconstruction variables are chosen by the caller. There is no conversion between primitive and conservative variables, characteristic projection, or thermodynamic admissibility check.
- TVD ratio functions alone do not prove multidimensional TVD, positivity, or stability of a complete scheme. Skew correction and time integration need separate consideration.
- Bounds from cell limiting concern the supplied samples and tested points. Regularized Venkatakrishnan limiting can relax those bounds.
- Numerical validity of fields, gradients, topology, and geometry is a caller precondition. Descriptor checks and debug assertions are not complete device-data validation.
- QUICK sample selection, wide-halo preparation, gradient/Hessian construction, and geometric moment construction are external to this module.
- The built-in adapter assumes the current grouped mesh and boundary-face indexing. It does not update descriptors or cached quantities automatically after geometry or topology changes.

Possible extensions, not currently implemented, include reusable mesh-specific QUICK stencil plans, geometric moment and Hessian builders, characteristic or positivity-preserving limiting, and reconstruction at multiple face quadrature points. Moving or adapted meshes would additionally need explicit refresh rules for geometry-dependent descriptors, weights, and limiter factors.
