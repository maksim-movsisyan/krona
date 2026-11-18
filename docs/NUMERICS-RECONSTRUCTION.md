# Reconstruction

The reconstruction module evaluates a field at a face point. An inner face has two traces, each based on its own cell. A physical face has one interior trace. The transport operator chooses which trace to use and applies the boundary condition.

Include `krona/numerics/reconstruction/reconstruction.hpp` for the built-in formulas, field adapters, and limiter preparation. Include `krona/numerics/reconstruction/stencil_data.hpp` and link `krona::reconstruction_mesh` when using the mesh adapter. For explicit geometry or a custom stencil, link `krona::reconstruction`.

## Formulas and their inputs

Include `schemes.hpp` for all built-in formulas, or a header under `schemes/` for one family. `polynomial.hpp` covers constant, linear, and quadratic profiles. `line.hpp` covers MUSCL and TVD, `quick.hpp` contains QUICK and its weights, and `limited.hpp` applies prepared cell factors.

Pass cell values in component order. Use `fields::pack(...)` for scalar views, multi-component fields, or a mixture of both. All providers must use the same cell and component ordering.

| Formula | What it uses |
|---|---|
| `FirstOrder{}` | The base cell value. |
| `Linear{gradients}` | Gradient of each component and displacement to the point. |
| `Quadratic{gradients, hessians, moments}` | The linear part and a quadratic term that preserves the cell average. |
| `MusclKappa{gradients, kappa}` | Local derivative, opposite sample and line geometry. The default kappa is 1/3. |
| `Tvd{gradients, tvd::Minmod{}}` | Local derivative and opposite sample, with a ratio limiter on the line increment. Defaults to Van Leer if the limiter is omitted. |
| `Quick{}` | Three explicit sample indices and weights for each side. |

`gradient_pack(g)` interprets a field with `3*N` components in the order `3*c + axis`. Use `gradients(gx, gy, gz)` when the three axes already have separate packs. Each axis pack follows the value component order. These adapters borrow the original memory and preserve its strides.

TVD and MUSCL can use `NoSkewCorrection{}` or the default `LinearSkewCorrection{}`. The default shifts the line result to the actual face point using the local gradient. TVD limits the line term. A cell limiter can bound the full increment, including the skew correction, at its evaluation points. The familiar uniform one-dimensional MUSCL values of kappa refer to particular choices of sampling. Quadratic exactness on a general 3D mesh requires multidimensional geometry and derivatives.

## Evaluate a face

If the caller already knows the point, pass its side geometry and evaluate directly in the consuming kernel.

```cpp
namespace rec = krona::numerics::reconstruction;

const auto linear = rec::Linear{gradients};
const rec::CellSide<Real> side{cell, 0, offset};
const auto state = rec::evaluate_side<Real>(linear, side, values);
```

`offset` points from the volume centroid to the evaluation point. The other sample, span, and fraction are needed by line formulas. Use `face_geometry(owner, neighbour, owner_offset, neighbour_offset, fraction)` to orient both sides consistently. An omitted fraction asks the formula for orthogonal projection onto the centre-to-centre segment. The mesh adapter computes the fraction from the intersection of that segment with the face plane.

For a mesh-based flux, bind the field and the two formulas once and copy the descriptor into the kernel.

```cpp
const rec::StencilData<Space, Real> stencil(mesh_view, cell_faces);
const auto profiles = rec::face_profiles(stencil, values, rec::Linear{gradients});

// In the consuming kernel.
const auto traces = profiles.template inner_values<Real>(face);
const auto inside = profiles.template boundary_values<Real>(physical_face);
```

The left state belongs to the owner and the right state belongs to the neighbour, which may be a ghost on an MPI interface. Scalar transport uses `inner_increment<Real>(face, from_owner, component)` and `boundary_increment<Real>(face, component)` for the correction to its cell-based upwind flux.

`StencilData` borrows grouped face connectivity, cell centroids, face centroids and normals, and `CellFaces` rows for the owned cells. The caller provides validated mesh geometry. The side formula determines what the adapter loads. A constant profile needs the cell index. A polynomial needs the point offset. A line formula also needs the opposite sample and span. Physical-face sample indices follow owned and ghost cells in a `CellExtnd` field.

## Prepare and use cell factors

The limiter first finds the minimum and maximum of the cell value and its bound samples. It then checks the unlimited profile at each evaluation point and takes the smallest admissible factor. `Componentwise` keeps separate factors. `SharedMinimum` uses the strictest factor across variables. Writable output can be a mixed-precision component pack. Provide space for all owned cells even when selecting a smaller `CellRange`.

When the flux uses a different formula at a physical face, pass both formulas to the same profile object before limiting.

```cpp
const auto raw = rec::face_profiles(
    stencil, values,
    rec::Tvd{gradients, rec::tvd::Minmod{}},
    rec::Linear{gradients});

rec::compute_limiters_async<Real>(exec, raw, factors, rec::BarthJespersen{});
const auto profiles = rec::limited(raw, factors);
```

This is the scalar outflow pattern. With positive outward mass flux the boundary increment comes from the adjacent owned cell's linear profile. At inflow the scalar transport operator uses its inflow condition. Factor preparation checks the interior trace on all incident physical faces, independently of flow sign, and includes their sampled values in the bounds. The same factor scales all increments of the cell. If a boundary point needs a factor of 1/4 while all other points allow 1, the cell uses 1/4.

`BarthJespersen` constrains the evaluated traces to their sampled bounds to the precision of the arithmetic and stored factors. `Venkatakrishnan` smooths the response. For example, it returns 3/4 when the increment exactly reaches the allowed bound and the regularization is zero. Nonzero regularization can relax the bounds. Use `Venkatakrishnan{ConstantRegularization<Real>{epsilon}}` to supply a smoothing scale in the variable's units.

Prepare on current data in this order.

1. Update owned values, exchange the value halo, and fill physical-face samples.
2. Compute owned derivatives and factors. Queue work on the same execution instance or establish the needed ordering.
3. Exchange the derivatives and factors before evaluating MPI ghost traces.

The module allocates no per-launch work arrays. The limiter launch is asynchronous and performs no exchange. The caller retains every backing array through kernel completion. Keep factor destinations separate from inputs. A custom limiter receives a `LimiterContext<Real>` and returns a finite factor in [0,1]. A custom cell stencil may supply `size()`, `size(cell)`, `sample(cell,slot)`, `point(cell,slot)`, and `required_input_size()` for direct limiter preparation without a mesh. It can also supply `point_count(cell)` to check several points without repeating the bound samples.

## Quadratic and QUICK

For `Quadratic`, use the volume centroid as the origin, the cell average as the base value, and the true second derivatives in the Hessian. The volume moment tensor holds the averages of the products of the displacement components. The formula adds

$$
\nabla q_i\cdot r + \tfrac12 H_i:(rr^T-M_i).
$$

Store mixed terms once in each symmetric tensor. The evaluation includes their factor of two. Subtracting the volume moments makes the increment average to zero. The caller supplies the derivatives and moments.

QUICK interpolates along one projected coordinate. Call `cell_average_quick_stencil` with three indices, the matching projected centroids, the central second moments along the line, and the target coordinate. For point values instead of cell averages use `point_quick_stencil`. On a uniform midpoint stencil the cell-average weights are `{-1/6, 5/6, 1/3}` and the point-sample weights are `{-1/8, 6/8, 3/8}`. Check the returned `valid` flag when preparing the stencil. A degenerate fit or nonfinite weight gives an invalid stencil.

Use a custom face stencil with `face(f)` returning `FaceStencil{left,right}` and `boundary_side(f)` returning one prepared side. Its `required_input_size()` must cover every sample it can use, including the furthest local index and any remote samples represented locally. For example, indices {0,1,5} require at least six values. Transport validation checks that extent before launching. The caller assembles and populates those samples. QUICK reproduces polynomials along its projected line, including quadratic cell averages when their moments are supplied.

A custom formula declares its `Geometry` and implements `increment<Real>(side, values, component)`. Add a host `visit_inputs(visitor)` that calls `visitor(name, provider)` for auxiliary fields that should have their extents checked. A provider can report a memory space and a size. This keeps validation and ownership with the inputs of the specific formula.

## Higher spatial orders

A common 3D finite-volume approach fits a degree-`k` polynomial to neighbouring cell averages while retaining the central cell average. Fit in scaled local coordinates using cell moments. For degrees 2, 3, and 4, the central average fixes one coefficient and leaves 9, 19, and 34 unknowns respectively. The neighbour rows must supply enough independent constraints. Additional rows can improve a least-squares fit. Choose and validate the stencil before time stepping on a static mesh.

A fitted degree-2 polynomial can support third-order spatial discretization in smooth regions. Degrees 3 and 4 are corresponding starting points for fourth and fifth order. The full operator also needs flux quadrature, boundary treatment, and verification at the desired order. For unsteady problems, choose the time integrator and time step so the temporal error does not dominate. Validate polynomial reproduction separately from convergence of the assembled residual or integrated solution. WENO or CWENO blends candidate polynomials to handle discontinuities. A fixed-depth halo is an implementation choice based on the sampled cells or on exchanged intermediate derivatives and polynomial coefficients.
