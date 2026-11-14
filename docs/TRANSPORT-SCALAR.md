# Scalar transport

`krona_scalar_transport` solves the finite-volume equation

$$
\frac{\partial(\rho\phi)}{\partial t}
+\nabla\cdot(\rho\mathbf u\phi)
-\nabla\cdot(\Gamma\nabla\phi)=s.
$$

The app advances one scalar on a static, partitioned 3D mesh. It uses a
prescribed velocity, uniform density `rho`, and a uniform isotropic diffusion
coefficient `Gamma`. Convection and diffusion can be enabled independently.
Steady mode omits the time derivative. The source `s` is specified per unit
volume.

## Run the app

Build with `KRONA_BUILD_APPLICATIONS=ON` and `KRONA_ENABLE_TRILINOS=ON`. To
build the mesh partitioner too, enable `KRONA_ENABLE_CGNS` and
`KRONA_ENABLE_PARTITIONING`. The transport app reads the partitioner's HDF5
output. Use the same MPI size to partition and solve, and use compatible
geometry precision and index widths.

Run these commands from `examples/transport/scalar` with the executables on
your `PATH`.

```sh
mpirun -np 2 krona_mesh_partitioner diffusion/mesh.cgns diffusion/mesh.h5
mpirun -np 2 krona_scalar_transport diffusion/mesh.h5 diffusion/transport.toml diffusion/bc.toml \
    --case diffusion --vtu diffusion/vtk -v
```

The directory also contains the Gaussian advection-diffusion example and
`mesh.toml` inputs for regenerating both CGNS meshes. See the
[examples README](../examples/transport/scalar/README.md).

| Option | Effect |
| --- | --- |
| `--field <name>` | Selects a scalar from both TOML files. Default `phi`. |
| `--case <name>` | Selects the C++ setup for `custom`, `diffusion`, or `gaussian`. Default `custom`. |
| `--vtu <dir>` | Writes the final owned-cell field to `scalar.pvtu` and the boundary geometry to `scalar_bnd.pvtu`. |
| `-v`, `--verbose` | Prints rank-zero run summaries. |
| `-vv` | Also prints linear iteration counts, full residuals, and accepted steps. |
| `-h`, `--help` | Shows command-line usage. |

Named options also accept `--option=value`.

## Transport TOML

The selected `[scalars.<field>]` table is required. Its subtables are optional.
This example enables transient convection, diffusion, and a constant source
for `phi`. Use a matching BC file and prescribe a velocity in the C++ setup
for a custom run with convection.

```toml
[scalars.phi.physics]
density = 1.0
convection = true
diffusion = true
diffusion_coefficient = 0.002

[scalars.phi.time]
scheme = "bdf2"
dt = 0.01
steps = 50

[scalars.phi.numerics.gradient]
method = "green_gauss_cell"
interpolation = "auxiliary_weights"

[scalars.phi.numerics.reconstruction]
method = "tvd"
tvd_limiter = "van_leer"
skew_correction = true

[scalars.phi.numerics.diffusion]
nonorthogonal_correction = true

[scalars.phi.source]
value = 0.1
```

### Physics and time

| Section | Key | Default and meaning |
| --- | --- | --- |
| `physics` | `density` | `1.0`. Positive uniform density for cell mass and face mass flux. |
| `physics` | `convection`, `diffusion` | Both `true`. Toggle the corresponding contributions. |
| `physics` | `diffusion_coefficient` | `0.01`. Positive when diffusion is enabled. This is `Gamma`. The diffusivity in a constant-density analytical solution is `Gamma / rho`. |
| `time` | `scheme` | `backward_euler`, `bdf2`, or `steady`. Default `backward_euler`. BDF2 starts with one Backward Euler step. |
| `time` | `dt`, `steps` | Positive fixed step size and positive step count. Defaults `0.01` and `1`. Omit both for `steady`, which requires an enabled spatial operator. |
| `source` | `type`, `value` | `"constant"` and `0.0`. By default the custom case uses this uniform source density. A C++ source can use the value as a parameter. |

### Spatial schemes

| Section | Key | Default and meaning |
| --- | --- | --- |
| `numerics.gradient` | `method` | `least_squares_face` (default), `green_gauss_cell`, `green_gauss_face`, or `none`. |
| `numerics.gradient` | `interpolation` | `arithmetic` (default) or `auxiliary_weights` for Green–Gauss interior-face values. |
| `numerics.gradient` | `pivot_tolerance` | `1e-12`. Least-squares stencil tolerance in `(0, 1)`. |
| `numerics.reconstruction` | `method` | `upwind` (default), `linear` (second-order upwind), `muscl_kappa`, or `tvd`. |
| `numerics.reconstruction` | `kappa` | `1/3` by default. A number in `[-1, 1]` for `muscl_kappa`. |
| `numerics.reconstruction` | `tvd_limiter` | `van_leer` (default), `minmod`, `superbee`, `monotonized_central`, `van_albada`, or `koren`. Applies to `tvd`. |
| `numerics.reconstruction` | `skew_correction` | `true`. Adds a transverse Taylor term for MUSCL/TVD. The linear scheme uses the full cell-to-face displacement. |
| `numerics.reconstruction` | `cell_limiter` | `none` (default), `barth_jespersen`, or `venkatakrishnan`. Applies a cell factor to the high-order convective increment. |
| `numerics.reconstruction` | `regularization` | `0.0`. Nonnegative Venkatakrishnan epsilon in units of the scalar. |
| `numerics.diffusion` | `nonorthogonal_correction` | `false`. Adds an explicit correction based on face gradients. Requires diffusion. |
| `numerics.diffusion` | `require_orthogonal`, `orthogonality_tolerance` | Orthogonality is required by default when the correction is disabled. The tolerance defaults to `1e-6` and must lie in `[0, 1)`. |

Gradient preparation runs when high-order convection or corrected diffusion
needs it. Both require a gradient method other than `none`. The default
least-squares method needs a nondegenerate 3D stencil. MUSCL and TVD use a
one-sided linear trace at outflow with the selected cell factor. QUICK and
quadratic reconstruction are library schemes that require extra inputs.
QUICK needs an explicit upstream and outflow stencil. Quadratic reconstruction
needs Hessians and cell moments.

### Linear solver and outer iterations

| Section | Key | Default and meaning |
| --- | --- | --- |
| `linear_solver` | `method`, `preconditioner` | `GMRES` and `RELAXATION` (Jacobi by default). The names select a Belos method and an Ifpack2/MueLu preconditioner. Use `NONE` to disable preconditioning. |
| `linear_solver` | `relative_tolerance`, `absolute_tolerance` | `max(1e-8, 20*eps)` and `0`. The relative reference is the norm of the current RHS, or the initial linear residual norm for a zero RHS. |
| `linear_solver` | `maximum_iterations`, `restart_length` | `500` and `50`. Restart length applies to methods with a Krylov restart. |
| `linear_solver` | `verify_residual` | `false`. Enable to recompute the true residual of the frozen linear system after the solve. |
| `outer_iteration` | `relative_tolerance`, `absolute_tolerance` | `max(1e-6, 100*eps)` and `0`. The relative reference is the larger of the initial full residual norm and the base RHS norm for the step. |
| `outer_iteration` | `maximum_iterations`, `relaxation` | `100` and `1`. Relaxation in `(0, 1]` determines how much of the linear update to apply. |

Here `eps` is the larger machine epsilon of `SolverReal` and `StorageReal`.
Each tolerance pair requires `0 <= relative < 1`, `absolute >= 0`, and at least
one positive value. GMRES is suitable for the generally nonsymmetric upwind
matrix. Choose CG only when the assembled system is symmetric positive
definite. See [linear algebra](LINALG.md) for solver details.

In C++, `transport::scalar::parse_config(text, field)` parses existing TOML
text locally. `read_config(path, field, comm)` reads the file on rank zero,
broadcasts its text, and calls the same parser on every rank. The parser
validates the selected scalar's keys, so one file can hold several scalars.

## Boundary data, initial field, velocity, and source

### Boundary conditions

Use the separate scalar BC TOML. Match every labelled mesh patch by
`patch_id` and `name`, then supply its `scalar.<field>` table. Handle unlabelled
faces in `[unlabelled.scalar.<field>]` when they exist.

```toml
[[boundary_condition]]
patch_id = 0
name = "inlet"

[boundary_condition.scalar.phi]
type = "FIXED_VALUE"
value = "profiles/inlet.csv"
```

A `FIXED_VALUE` can also use a number. A CSV path is resolved against rank
zero's working directory. The BC module loads and interpolates the profile
at local face centroids during setup. The app uses `FixedValueInflow`, so any
boundary face with negative outward mass flux needs a `FIXED_VALUE` for the
transported scalar, including faces experiencing backflow. For an
impermeable patch, supply an exact zero in the mass-flux provider. See
[scalar boundary conditions](BOUNDARY-SCALAR.md) for all BC types and the
CSV format.

### Initial field and velocity

With `--case custom`, edit these headers in `apps/scalar_transport` and rebuild:

- In `initial_field.hpp`, return `phi(x, y, z, 0)` from the `Case::custom` branch of
  `initial_field()`. The app calls it at owned cell centroids. In steady mode,
  it provides the first iterate.
- In `velocity.hpp`, return the prescribed velocity from `velocity()` at the
  face centroid and supplied time. `MassFlux` computes
  `m = rho * area * dot(u, n)` using the local owner-outward normal. When a flow
  solver supplies mass fluxes directly, convection can consume a
  `FaceMassFlux{view}` indexed by local face ID. Fluxes on MPI interface copies
  must agree after accounting for their respective normal directions.
- In `analytic_solution.hpp`, enable the custom case in `has_analytic_solution()`
  and add its formula to `analytic_solution()` if an analytical solution is
  available. The formula must match the chosen velocity, source, initial data,
  and BCs.

For example, replace the custom branch in `initial_field()` with a Gaussian
centred at `x = 0.5` with standard deviation `0.1`.

```cpp
if (problem == Case::custom) {
    const Real d = (x - static_cast<Real>(0.5)) / static_cast<Real>(0.1);
    return Kokkos::exp(-static_cast<Real>(0.5) * d * d);
}
```

The custom case starts with zero scalar and zero velocity by default. The
`diffusion` and `gaussian` cases provide complete C++ setups for the bundled
examples. Their analytical comparisons use source-free transient solutions.

### Volume source

Set `[scalars.<field>.source].value` for a uniform source. To prescribe a
source that depends on space or time, edit the `Case::custom` branch of
`source_density()` in `apps/scalar_transport/source.hpp`, then rebuild and
run with `--case custom`. The function receives the cell centroid, the
evaluation time, and the configured value as `uniform`. For example, name
the coordinate and time arguments and use this branch:

```cpp
if (problem == Case::custom)
    return uniform * (Real{1} + x) * Kokkos::exp(-time);
```

Return source density per unit volume. `SourceOperator` multiplies it by cell
volume. The driver evaluates the source at the new step time for both RHS
assembly and the full residual. It uses time zero in steady mode. At the
library level, `SourceOperator` also accepts
`transport::scalar::CellSource{view}` for an existing rank-one device array.

## Solve pipeline

### 1. Prepare mesh, fields, and operators

Import the HDF5 partition and attach the finite-volume mesh, owned-cell
`CellFaces` connectivity, and auxiliary geometry requested by the selected
schemes. Diffusion uses normal distances and area divided by normal distance.
Corrected diffusion also uses interpolation weights and nonorthogonality
vectors. Configure and finalize one `scalar::TabulatedManager` against the
mesh's patch catalog.

The scalar field stores owned cells, MPI ghosts, and appended physical-face
samples. Gradients and limiter factors cover real cells, including ghosts,
when required. The separate accepted time layers hold owned values only.
Build one face assembly plan and reuse its CSR sparsity, matrix, solver, and
halo exchanges for the run.

### 2. Assemble the base equation at the new time

Freeze the accepted layers and evaluate the velocity and boundary data at
the new step time. Clear the matrix and RHS once for this step. Each enabled
term adds its contribution to the same system.

| Term | Matrix | Base RHS |
| --- | --- | --- |
| Time derivative | Diagonal `w0 * rho * V`. | Previous-layer terms `-wj * rho * V * phi_previous`. |
| Convection | Donor-cell upwind contributions, including the outflow diagonal. | Prescribed inflow contributions. |
| Diffusion | Two-point diffusion and the implicit part of boundary relations. | Prescribed diffusive boundary terms with zero explicit correction. |
| Source | Zero. | Integrated source `s * V`. |

The history starts with accepted layer `n`, with weight `w1`. The candidate
layer `n+1` uses `w0`. BDF2 also reads `n-1` once it is available. Save the
assembled RHS as `base_rhs` and commit the matrix and RHS. The matrix
coefficients are rebuilt at each time step and reused across its outer
iterations. Steady mode performs one spatial solve at time zero.

### 3. Prepare the current iterate

Exchange scalar ghosts. When gradients are needed, fill the physical-face
samples from the BC manager and compute owned gradients. The first boundary
pass uses the available gradients for nonorthogonal correction, starting
from zero at initialization. If both nonorthogonal correction and cell
limiting are active, refresh the face samples with the new gradients before
computing limiter factors. Compute those factors using the selected
unlimited reconstruction and exchange the gradients and any active factors.

Boundary samples serve gradient and limiter stencils. Spatial assembly
obtains the boundary contribution directly from the BC manager.

### 4. Solve with frozen corrections and check the full residual

Restore `base_rhs` for each outer iteration. Add the current high-order
convective increment and nonorthogonal diffusion correction to that RHS.
Solve `A*phi = rhs` using the current owned field as the initial guess. Apply
optional relaxation, refresh the halo and reconstruction data, then evaluate
the full residual from the updated field.

$$
r_i = s_i V_i
      - \sum_f \bigl(F_{\mathrm{conv},i,f} + F_{\mathrm{diff},i,f}\bigr)
      - \sum_j w_j (\rho V\phi)_i^{n+1-j}.
$$

Here the face fluxes are integrated and signed outward from cell `i`.
Steady mode drops the last sum. The stopping norm is the global Euclidean
norm over owned-cell residuals, with no cell-volume division. Accept the
step when it falls below or equals

$$
\max\left(\text{absolute tolerance},
          \text{relative tolerance}\,\max(\|r_{\mathrm{initial}}\|_2, \|b_{\mathrm{base}}\|_2)\right).
$$

This check also runs on the initial iterate. A converged Belos solve only
establishes convergence for its frozen RHS. The outer check includes the
corrections recomputed from the new iterate. Linear or outer nonconvergence
terminates the run. A steady diffusion problem with only prescribed fluxes
or gradients needs an external reference or nullspace treatment.

### 5. Accept the step and export

After acceptance, advance time and rotate the owned history for the next
step. At the end, `--vtu` exports the final solution. For cases with an
analytical solution the app reports volume-weighted RMS error and maximum
absolute error at cell centroids. VTU output also includes
`<field>_exact` and the signed `<field>_error`.

## Execution and synchronization

The application uses one execution-space instance. Gradient, boundary,
limiter, and additive transport kernels enqueue work on that instance.
The driver's halo exchanges complete before their consumers proceed.
`boundary::scalar::Status::check()` waits locally for boundary diagnostics.
`AssemblyStatus::check(comm)` waits for assembly and reduces its diagnostics
across ranks. Commit native matrix and vector access before calling the
blocking collective linear solver. Residual norms and VTU output also
require all participating MPI ranks.

See the operator guides for [convection](TRANSPORT-SCALAR-CONVECTION.md),
[diffusion](TRANSPORT-SCALAR-DIFFUSION.md), and
[time integration](TRANSPORT-SCALAR-TRANSIENT.md). The
[gradient](NUMERICS-GRADIENT.md) and
[reconstruction](NUMERICS-RECONSTRUCTION.md) guides describe stencil contracts
and the lower-level APIs.