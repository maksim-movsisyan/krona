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
    --case diffusion -v
```

The directory also contains the Gaussian advection-diffusion example and
`mesh.toml` inputs for regenerating both CGNS meshes.
| Option | Effect |
| --- | --- |
| `--field <name>` | Selects a scalar from both TOML files. Default `phi`. |
| `--case <name>` | Selects the C++ setup for `custom`, `diffusion`, or `gaussian`. Default `custom`. |
| `--vtu <dir>` | Overrides the output directory from TOML. |
| `-v`, `--verbose` | Prints rank-zero run summaries. |
| `-vv` | Also prints linear iteration counts and accepted steps. |
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

[scalars.phi.outer_iteration]
iterations = 3

[scalars.phi.output]
directory = "vtk"
every = 10
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
| `outer_iteration` | `iterations`, `relaxation` | `1` and `1`. Each step uses the specified positive number of outer iterations. Relaxation in `(0, 1]` determines how much of the linear update to apply. |

Here `eps` is the larger machine epsilon of `SolverReal` and `StorageReal`.
The linear tolerance pair requires `0 <= relative < 1`, `absolute >= 0`, and at least
one positive value. GMRES is suitable for the generally nonsymmetric upwind
matrix. Choose CG only when the assembled system is symmetric positive
definite. See [linear algebra](LINALG.md) for solver details.

For an older configuration, replace `outer_iteration.maximum_iterations`
and the two outer tolerances with the desired fixed `outer_iteration.iterations`.

### Output

| Section | Key | Default and meaning |
| --- | --- | --- |
| `output` | `directory` | Use a nonempty path to enable VTU output or pass `--vtu <dir>` on the command line. Relative paths are resolved from the working directory. |
| `output` | `every` | `0` writes only the final field. A positive integer writes every N accepted transient steps and the final step. |

Periodic snapshots go in `<directory>/step_<number>`, numbered from 1. Each
contains `scalar.pvtu` and `scalar_bnd.pvtu` with their corresponding rank
pieces. Steady and final-only runs write directly to the output directory.

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
volume. The driver evaluates the source at the new step time during base RHS
assembly. It uses time zero in steady mode. At the
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
assembled RHS as `base_rhs` when explicit corrections are enabled, then commit
the matrix and RHS. The matrix
coefficients are rebuilt at each time step and reused across its outer
iterations. Steady mode runs the same outer loop at time zero.

### 3. Prepare the current iterate

Before each correction pass, exchange scalar ghosts. When gradients are needed, fill the physical-face
samples from the BC manager and compute owned gradients. The first boundary
pass uses the available gradients for nonorthogonal correction, starting
from zero at initialization. If both nonorthogonal correction and cell
limiting are active, refresh the face samples with the new gradients before
computing limiter factors. Compute those factors using the selected
unlimited profiles for inner and physical faces, then exchange the gradients and any active factors.

Boundary samples serve gradient and limiter stencils. Spatial assembly
obtains the boundary contribution directly from the BC manager.

### 4. Solve with frozen corrections

For each outer iteration with explicit corrections, restore `base_rhs` and
add the current high-order convective increment and nonorthogonal diffusion
correction. Solve `A*phi = rhs` using the current owned field as the initial
guess and apply optional relaxation. The next pass builds its corrections
from the updated field. With both corrections disabled, all outer iterations
reuse the assembled RHS.

The driver runs exactly `outer_iteration.iterations` passes per step. Linear
solver tolerances still apply to each frozen system. The timestep accepts the
last iterate after checking assembly diagnostics. A fixed number of passes
can leave a defect in the fully corrected discrete equation. A steady
diffusion problem with only prescribed fluxes or gradients needs an external
reference or nullspace treatment.

### 5. Accept the step and export

Advance time and rotate the owned history after the outer loop. Write a
snapshot when requested by `output.every` or when the final step is accepted.
For cases with an analytical solution the app reports volume-weighted RMS
and maximum absolute errors at the final time. Each VTU snapshot also includes
`<field>_exact` and the signed `<field>_error` for these cases.

## Execution and synchronization

The application uses one execution-space instance. Gradient, boundary,
limiter, and additive transport kernels enqueue work on that instance.
The driver's halo exchanges complete before their consumers proceed.
Base assembly checks diagnostics before the solve. The outer passes accumulate
boundary and correction flags and check them once after the loop.
Both `boundary::scalar::Status::check()` and `AssemblyStatus::check()` wait
on the local execution instance and check that rank's flags. A kernel error
is reported through `mpi::require`, which aborts the MPI job.
Native matrix and vector accesses are committed before the blocking collective
linear solve. The outer loop avoids the additional full residual pass and its
MPI norm reduction. VTU output involves all participating ranks.

See the operator guides for [convection](TRANSPORT-SCALAR-CONVECTION.md),
[diffusion](TRANSPORT-SCALAR-DIFFUSION.md), and
[time integration](TRANSPORT-SCALAR-TRANSIENT.md). The
[gradient](NUMERICS-GRADIENT.md) and
[reconstruction](NUMERICS-RECONSTRUCTION.md) guides describe stencil contracts
and the lower-level APIs.
