# Reconstruction

`krona::numerics::reconstruction` reconstructs cell data at a face point. It returns one state from each adjacent cell; the flux calculation decides how to use those states.

```cpp
#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/numerics/reconstruction/mesh_stencil.hpp> // Optional mesh adapter.
namespace rec = krona::numerics::reconstruction;
```

The core needs Kokkos and KRONA execution/type definitions. It accepts small data providers and has no mesh, fields, MPI, boundary-condition, or flux dependency. `mesh_stencil.hpp` adds the device-mesh dependency. Field adapters accept `FieldView` structurally, without including the fields module.

## Main objects

| Object / operation | Responsibility |
|---|---|
| `data(values, gradients, factors, hessians, moments)` | Groups borrowed input providers. Only supply the providers required by the selected scheme. |
| A scheme, e.g. `SecondOrderUpwind{}` | Computes the increment relative to the cell value. |
| `CellSide`, `FaceGeometry` | Describes the cells and geometry for evaluating one or both sides. |
| `MeshStencil` | Adapts ready mesh geometry and `CellFaces`; also supplies samples for limiter preparation. |
| `evaluate<Real>(scheme, face, data)` | Returns `FaceValues<Real, N>` with `left[N]` and `right[N]`. |
| `evaluate_side<Real>(scheme, side, data)` | Returns `Kokkos::Array<Real, N>` for one side. |
| `compute_limiters_async(...)` | Writes cell limiter factors into caller-provided storage. |

For `MeshStencil::face()`, **left is owner and right is neighbour**, including MPI ghost neighbours. Left/right do not mean upstream/downstream. All components are reconstructed in pack order; the module assigns no physical meaning to them.

## Available methods

Every method reads the base cell values. The additional inputs below must exist for every cell/sample that the evaluation accesses.

| Scheme | Additional inputs | Behaviour |
|---|---|---|
| `FirstOrder` | None | Returns the cell value unchanged. |
| `SecondOrderUpwind` | Gradients | Linear Taylor reconstruction: `q + gradient · offset`. Despite its name, it evaluates either side; it does not select the upwind side. |
| `LimitedLinear` | Gradients, cell factors | Linear Taylor increment multiplied by a precomputed factor. Alias of `Limited<SecondOrderUpwind>`. |
| `Quadratic` | Gradients, Hessians, central cell moments | Mean-preserving quadratic Taylor reconstruction. Derivatives and moments must be supplied by the application. |
| `LimitedQuadratic` | Same as `Quadratic`, plus factors | Multiplies the entire linear-plus-quadratic increment by the cell factor. |
| `MusclKappa<>` | Gradients, other-side values | Unlimited MUSCL profile. Default `kappa = 1/3`; `MusclKappa{0.5}` gives the point-valued QUICK profile on a uniform 1-D midpoint stencil. |
| `Tvd<>` | Gradients, other-side values | Darwish–Moukalled virtual-upstream ratio with a flux-limiter function. Default limiter: `tvd::VanLeer`. |
| `Quick` | Explicit three-point stencil and its sample values | Weighted quadratic interpolation. No automatic upstream-cell search or additional halo construction. |
| `Limited<Scheme>` | Inputs of `Scheme`, plus factors | Applies a cell factor to that scheme's complete increment, including its skew correction. |

`evaluate<Real>()` defaults to `ComputeReal`. It converts provider values to the requested arithmetic/result type. Storage types can differ from `Real`; the base value and increments are returned in `Real`.

### MUSCL and TVD geometry

For a side, let `d = span`, `r_f = offset`, `f = fraction`, and `jump = q_other - q_cell`. The default transverse correction is

$$
C=\nabla q\cdot r_f-f\,\nabla q\cdot d.
$$

`MusclKappa` adds `f * ((1-kappa) * gradient·d + kappa * jump) + C` to the cell value. `Tvd` adds `f * psi(r) * jump + C`, with `r = 2 * (gradient·d) / jump - 1`. For an exactly zero jump, TVD returns the transverse correction alone.

Available TVD functions are `tvd::Minmod`, `Superbee`, `MonotonizedCentral`, `VanLeer`, `VanAlbada`, and `Koren`. For example, use `Tvd{tvd::Minmod{}}` to select Minmod.

Both methods default to `LinearSkewCorrection`. Select `NoSkewCorrection` as the last template parameter to omit the transverse term, e.g. `Tvd<tvd::VanLeer, DarwishMoukalled, NoSkewCorrection>`. The default correction preserves affine fields at skew face points, but a line-based TVD limiter does not bound that transverse term. Neither the TVD name nor the MUSCL parameter establishes multidimensional boundedness, positivity, or the order of a complete solver. To limit the complete increment at the sampled points, prepare cell factors for that same scheme and use `Limited<Scheme>`.

## Input providers and geometry

Values provide `components` and `operator()(cell, component)`. For limiter preparation they also provide `size()`. Use these allocation-free adapters:

| Adapter | Input / result |
|---|---|
| `pack(a, b, ...)` | Rank-one Kokkos views, including strided subviews. One component per view; same scalar type ignoring const and same memory space. `size()` is the shortest component length. If any input is const, the whole pack is read-only. |
| `field_pack(field)` | All components of a KRONA `FieldView`, preserving component order. |
| `gradient_pack(field)` | A `FieldView` with `3*N` components in the gradient module's order: `3*c + {0,1,2}`. Returns a vector for `(cell, c)`. |
| `gradients(x, y, z)` | Three component packs, each with `N` components, giving the x/y/z derivatives. |

```cpp
auto values = rec::pack(rho, u, v, w, temperature); // Rank-one views.
auto input = rec::data(values, rec::gradient_pack(gradient_field));
// gradient_field has 15 components in the same variable order.
```

Factors provide `components` and `(cell, component)`: either `N` components or one component broadcast to all variables. Omitted factors are `UnitFactors` (always one). Hessians provide `(cell, component)` and moments provide `(cell)`, returning objects with `xx, yy, zz, xy, xz, yz`; `SymmetricTensor<Real>` is the supplied tensor type.

`CellSide` contains:

| Member | Meaning |
|---|---|
| `cell` | Base cell index. |
| `other` | Other value index, used by MUSCL/TVD; it can be an extended boundary slot. |
| `offset` | Vector from the base cell centroid to the evaluation point. |
| `span` | Vector from the base cell centroid to the other sample position. |
| `fraction` | MUSCL/TVD line fraction, required in `[0,1]`. A negative value requests `dot(offset, span) / dot(span, span)`. |

`face_geometry(owner, neighbour, owner_to_face, neighbour_to_face, owner_fraction)` creates both sides. It computes `span` from the offsets; an explicit right fraction is `1 - owner_fraction`. With the default negative fraction, each side uses orthogonal projection. Explicit fractions can represent intersection with the face plane instead.

All accessed indices, array lengths, component counts and geometry must be valid. In particular, projection requires a nonzero span. Mesh plane fractions require a nonzero `dot(span, normal)` and must lie in `[0,1]` when consumed by MUSCL/TVD. Finite inputs are caller preconditions; device assertions are not a replacement for validation.

## Using the mesh adapter

Construct `MeshStencil<Space, Real>(mesh_view, aux_connectivity_view)` from ready data:

- grouped face intervals;
- mesh groups `FaceCells | CellGeometry | FaceGeometry`;
- auxiliary connectivity `CellFaces`, with rows for owned cells;
- consistent indices, centroids and face normals. The constructor checks descriptors/extents, not every topology entry or geometric denominator.

Even for first-order evaluation, the constructor currently requires these groups. `face<FirstOrder>()` itself avoids geometry loads.

| Call | Result |
|---|---|
| `stencil.size()` | Number of owned cells, i.e. limiter-output rows. |
| `stencil.required_input_size()` | `n_cells + n_faces - n_inner_faces`, the `CellExtnd` value length. |
| `stencil(cell, k)` | A limiter sample and the cell-side geometry for the k-th incident face. |
| `stencil.face<Scheme>(f)` | Both sides of an **inner** face, `f < n_inner_faces`. Includes owned/ghost faces. |
| `stencil.boundary_side(f)` | Interior side of a physical boundary face. |

For limiter bounds, inner-face samples are real neighbouring cells, including ghosts. A physical boundary face samples slot `n_cells + (f - n_inner_faces)`, representing a value at the face centroid. Fill those slots through the application's boundary-condition pipeline. They are not extra ghost cells.

`boundary_side()` reconstructs only the interior trace. It does not apply a physical condition or construct an exterior state. MUSCL/TVD read its boundary slot as `other`; if that is not the intended boundary algorithm, select an explicit one-sided scheme such as `SecondOrderUpwind` or `LimitedLinear` for this trace, then apply the boundary closure.

## Limited-linear pipeline

For each update:

1. Fill owned values and physical-boundary slots; exchange value ghosts.
2. Compute current owned gradients. The gradient module can write the `3*N` layout consumed by `gradient_pack()`.
3. Compute owned limiter factors with current values and owned gradients.
4. Exchange gradients and factors before evaluating the ghost side of an owned/ghost face.
5. Evaluate face states in the flux/residual kernel, or store them in caller-owned arrays.

Allocate fields once. In the following setup, `fields` is a `FieldsManager<Exec>` with the correct mesh counts; `device_mesh` includes `Halo` as well as the stencil's mesh groups, and `device_aux` includes `CellFaces`. `comm` matches the mesh. Choose nonconflicting MPI tags.

```cpp
using Real = krona::StorageReal;
using Space = typename Exec::memory_space;
constexpr int N = 5;

auto q = fields.template add_field<Real, N>(
    "reconstruction/q", krona::fields::FieldLocation::CellExtnd);
auto g = fields.template add_field<Real, 3*N>(
    "reconstruction/gradient", krona::fields::FieldLocation::Cell);
auto theta = fields.template add_field<Real, N>(
    "reconstruction/factor", krona::fields::FieldLocation::Cell);

const auto m = device_mesh.view();
const rec::MeshStencil<Space> stencil(m, device_aux.view());

krona::halo::Exchange<Real, Exec> values_halo(
    m, device_mesh.halo_plan(), comm, 31, exec);
values_halo.add_field(q);
values_halo.prepare();

krona::halo::Exchange<Real, Exec> derived_halo(
    m, device_mesh.halo_plan(), comm, 32, exec);
derived_halo.add_field(g);
derived_halo.add_field(theta);
derived_halo.prepare();

Kokkos::View<krona::ComputeReal***, Space> states(
    "reconstruction/states", m.sizes.n_inner_faces, 2, N);
```

Once the application has filled the current owned and boundary values, perform an update. Here `gradient_operator` is a prepared gradient plan using the same `exec`; its `apply()` writes owned gradients asynchronously.

```cpp
exec.fence("q ready for halo exchange");
values_halo.exchange();
gradient_operator.apply(q, g);

const auto raw = rec::data(rec::field_pack(q), rec::gradient_pack(g));
rec::compute_limiters_async(exec, stencil, raw, rec::field_pack(theta));

exec.fence("owned derivatives and factors ready for exchange");
derived_halo.exchange();

const auto input = rec::data(raw.values, raw.gradients, rec::field_pack(theta));
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
Kokkos::parallel_for("reconstruct/inner", Policy(exec, 0, m.sizes.n_inner_faces),
    KOKKOS_LAMBDA(std::size_t f) {
        const auto face = stencil.template face<rec::LimitedLinear>(f);
        const auto result = rec::evaluate(rec::LimitedLinear{}, face, input);
        for (int c = 0; c < N; ++c) {
            states(f, 0, c) = result.left[c];
            states(f, 1, c) = result.right[c];
        }
    });
exec.fence("face states ready");
```

The stored `states` array is optional: call `evaluate()` directly inside the flux kernel to consume the states immediately. Physical boundary faces use `evaluate_side()` followed by the application's boundary closure:

```cpp
// Inside a boundary-face kernel; f >= n_inner_faces.
const auto interior = rec::evaluate_side(
    rec::LimitedLinear{}, stencil.boundary_side(f), input);
// Pass interior to the selected physical boundary/flux algorithm.
```

## Choosing the method at runtime

Schemes are C++ types. Dispatch on the host and pass a concrete scheme into the kernel. This example reuses the stencil, input descriptors and output array above; `input` must already contain the current providers needed by the selected branch.

```cpp
enum class Method { FirstOrder, Linear, LimitedLinear, Muscl, Tvd };

auto launch = [&](auto scheme) {
    using Scheme = decltype(scheme);
    Kokkos::parallel_for("reconstruct/selected", Policy(exec, 0, m.sizes.n_inner_faces),
        KOKKOS_LAMBDA(std::size_t f) {
            const auto face = stencil.template face<Scheme>(f);
            const auto result = rec::evaluate(scheme, face, input);
            for (int c = 0; c < N; ++c) {
                states(f, 0, c) = result.left[c];
                states(f, 1, c) = result.right[c];
            }
        });
};

switch (method) { // Application-selected Method value.
case Method::FirstOrder:    launch(rec::FirstOrder{}); break;
case Method::Linear:        launch(rec::SecondOrderUpwind{}); break;
case Method::LimitedLinear: launch(rec::LimitedLinear{}); break;
case Method::Muscl:         launch(rec::MusclKappa{}); break;
case Method::Tvd:           launch(rec::Tvd{rec::tvd::VanLeer{}}); break;
}
```

There is no per-face runtime scheme switch in this example. Each instantiated kernel reads only what its scheme uses. Preparation should follow the selected method too: first order needs neither gradients nor factors, and unlimited linear/MUSCL/TVD need no cell factors. If selecting `Limited<Tvd<...>>`, prepare factors using the same TVD object, not `SecondOrderUpwind`.

## Preparing cell factors

`compute_limiters_async(exec, stencil, input, output, limiter, scheme, coupling, range)` defaults to `BarthJespersen`, `SecondOrderUpwind`, `Componentwise`, and all owned cells.

For each selected cell it:

1. Finds component minima/maxima over the cell value and all stencil samples.
2. Evaluates the **unlimited** scheme increment at every stencil point.
3. Computes a factor per point and retains the minimum per component.
4. Writes either those factors or their shared minimum across components.

| Choice | Contract |
|---|---|
| `BarthJespersen{}` | Restricts the sampled reconstructed values to their local component bounds, using the same data, scheme and points during consumption. It does not bound untested points. |
| `Venkatakrishnan{}` | Smoothed limiter with zero regularization by default. Can reduce an already admissible increment; at `allowed/increment = 1`, the factor is `3/4`. |
| `Venkatakrishnan{ConstantRegularization<Real>{epsilon}}` | Uses finite `epsilon >= 0`, in the reconstructed variable's units. Its square enters the formula. Nonzero regularization can relax the local bounds. A callable regularization can vary with cell/component. |
| `Componentwise{}` | Output has exactly `N` components; each variable gets its own factor. |
| `SharedMinimum{}` | One factor across all `N` variables. Output may have one component or `N` identical components. |
| `CellRange{begin, end}` | Writes `[begin, end)` only; all other cells, including ghosts, remain unchanged. Default end means `stencil.size()`. |

Output must be writable and cover `stencil.size()` even for a partial range. Values must cover `stencil.required_input_size()`. Required gradient/Hessian/moment providers must cover all owned rows, even for a partial range. Use an unlimited scheme: passing `Limited<...>` to this preparation is rejected at compile time.

A custom limiter is a device-callable `operator()(LimiterContext<Real>)` returning a finite factor in `[0,1]`. The context contains `cell`, `component`, `value`, `minimum`, `maximum`, and the unlimited `increment`. A custom stencil provides `size()`, `required_input_size()`, `size(cell)`, and `(cell,k) -> StencilEntry`; each entry supplies a bound-sample index and side geometry whose base cell is the row being limited. Include every evaluation point for which the factor's bound contract is needed.

## Quadratic and QUICK

For `Quadratic`, the application supplies true second derivatives and central volume moments

$$
M_i=\frac{1}{V_i}\int_{V_i}(x-x_i)(x-x_i)^T\,dV,
\qquad
q_f=\bar q_i+\nabla q_i\cdot r+\tfrac12 H_i:(rr^T-M_i).
$$

The centre `x_i` must be the volume centroid; values are cell averages. Off-diagonal Hessian/moment entries are stored once, and the evaluation includes their factor of two. The subtraction of `M_i` preserves the cell average, also when the whole increment is multiplied by one cell factor. The module does not compute Hessians or moments, and `MeshStencil` does not supply them.

```cpp
// Inside an inner-face kernel. hessians and moments are application providers.
const auto quadratic = rec::data(raw.values, raw.gradients, rec::UnitFactors{},
                                 hessians, moments);
const auto result = rec::evaluate(rec::Quadratic{},
    stencil.template face<rec::Quadratic>(f), quadratic);
```

For `Quick`, select three samples explicitly for each side. `cell_average_quick_stencil()` takes sample indices, projected cell centres, projected central variances, and the target coordinate, all in the same coordinate system. It returns `ThreePointStencil`, with the centre sample as its base `cell`.

```cpp
const auto side = rec::cell_average_quick_stencil(
    Kokkos::Array<std::size_t, 3>{0, 1, 2},
    Kokkos::Array<double, 3>{-1.0, 0.0, 1.0},
    Kokkos::Array<double, 3>{1.0/12, 1.0/12, 1.0/12}, 0.5);
krona::mpi::require(side.valid, "Invalid QUICK stencil");
// Inside a kernel, using values with the matching sample indices:
// auto state = rec::evaluate_side(rec::Quick{}, side, rec::data(values));
```

The weights here are `[-1/6, 5/6, 1/3]`. For point-valued data, use `point_quick_stencil()` instead; on the same uniform geometry its weights are `[-1/8, 6/8, 3/8]`. Do not use point weights for finite-volume cell averages.

These fits reproduce quadratics in the selected projected coordinate, not arbitrary 3-D quadratic fields. Invalid/degenerate geometry returns `valid == false` and NaN weights; there is no automatic fallback. Check validity before evaluation. The default degeneracy tolerance is `64 * epsilon` of the stencil's real type.

For two-sided evaluation, pass `FaceStencil{left_stencil, right_stencil}` to `evaluate(Quick{}, ...)`. Select both sample sets yourself; all samples must already be locally accessible and current. A face-neighbour halo need not contain a more distant QUICK sample. `MeshStencil::face<Quick>()` does not construct a three-point stencil and cannot be used for QUICK evaluation.

## Custom reconstruction

A custom scheme supplies `requirements` and `increment<Real>(side, data, component)`, returning an increment, not the complete face value. `evaluate()` adds the base value itself. For example:

```cpp
struct DampedLinear {
    static constexpr unsigned requirements = rec::Gradients;
    double damping = 0.5;

    template<class Real, class Side, class Data>
    KOKKOS_INLINE_FUNCTION Real increment(
        const Side& side, const Data& d, std::size_t c) const
    {
        return static_cast<Real>(damping)
             * rec::dot_as<Real>(d.gradients(side.cell, c), side.offset);
    }
};
```

Requirement flags are `Values` (zero), `Gradients`, `Factors`, `Hessians`, `Moments`, and `OtherValues`. Set them accurately: preparation and mesh adapters use them to select accesses/checks. For custom schemes needing a different side structure or stencil, provide that structure explicitly instead of assuming `MeshStencil` constructs it.

## Ownership, execution and errors

| Operation | Execution / ownership |
|---|---|
| `pack`, `field_pack`, `gradient_pack`, `gradients`, `data` | Descriptor construction; no data copy or allocation. Standard packs retain raw pointers/strides, not buffer ownership. |
| `MeshStencil` construction | Local host descriptor checks; no allocation, kernel launch, fence or MPI. Borrows mesh/auxiliary arrays. |
| `face_geometry`, QUICK stencil factories | Inline calculations in the calling host/device context; no queued work or allocations. |
| `evaluate`, `evaluate_side` | Inline calculations in the caller's kernel; no independent launch, synchronization, allocation or MPI. Results are returned by value. |
| `compute_limiters_async` | Host checks followed by one kernel on the supplied `exec`. No fence, temporary arrays, halo exchange or implicit ghost update. Empty ranges submit no kernel after validation. |

Keep all backing arrays alive and unchanged in layout/topology until submitted work finishes. Required buffers must be accessible from the chosen execution space. Factor outputs and saved face outputs must not overlap inputs that the kernels still read; the module does not detect aliasing. Recompute factors when their values, derivatives, scheme parameters, stencil or evaluation points change.

Kernels submitted to the same execution-space instance can consume earlier results in order without an intermediate fence. Before host reads, independent execution instances, external MPI use, destruction or reallocation, establish completion of the relevant producers. The pipeline above uses explicit fences before blocking halo exchanges and after its final kernel. Reconstruction itself performs no MPI communication.
