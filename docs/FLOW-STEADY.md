# Steady compressible flow

The steady flow solver integrates the five compressible equations for a
calorically perfect gas on a partitioned, static 3-D mesh. The executable is
`krona_flow_steady`.
Set `enabled = true` in `[flow.transport]` to include laminar viscous
transport and heat conduction. The default is inviscid flow.
Build with `KRONA_BUILD_APPLICATIONS=ON` and `KRONA_ENABLE_TRILINOS=ON`.

```sh
mpirun -np N krona_flow_steady mesh.h5 flow.toml bc.toml [options]
```

Use the MPI size that produced `mesh.h5`. The application accepts
`--case custom|channel|wedge`, `--vtu DIRECTORY`, `-v` and `-vv`.
`--vtu` overrides the TOML output directory. `-vv` prints the spatial residual,
five equation norms, Krylov iteration count and elapsed time at each update.
The default `custom` case
uses the uniform reference state from TOML. Edit
[initial_field.hpp](../apps/flow_steady/initial_field.hpp) for a different spatial
initialization. See the [examples](#examples) for ready-to-run cases.

## Solver configuration

The application reads the TOML file on rank zero and broadcasts the text
before parsing it on all ranks. `[flow.initial]` is required. The other
flow subtables are optional. Use the prefix `flow.` when migrating existing
configuration tables. For a formerly viscous case, also set
`[flow.transport]` with `enabled = true`.

| Table | Keys and defaults |
| --- | --- |
| `[flow.physics]` | `gamma = 1.4`, `gas_constant = 287.05` J/(kg K). |
| `[flow.initial]` | Positive `pressure` in Pa and `temperature` in K, plus `velocity = [0, 0, 0]` in m/s by default. |
| `[flow.numerics]` | `flux = "hllc"` (also `"hll"` or `"rusanov"`), plus the reconstruction options below. |
| `[flow.numerics.gradient]` | Cell gradient method and interpolation settings listed below. |
| `[flow.transport]` | `enabled = false`. See [molecular transport](#molecular-transport) for the viscous case. |
| `[flow.pseudo_time]` | `cfl = 1.0`, `relaxation = 1.0`, `maximum_iterations = 200`, `relative_tolerance = 1e-8`, `absolute_tolerance = 1e-8`. |
| `[flow.linear_solver]` | `method = "GMRES"`, `preconditioner = "RELAXATION"`, `relative_tolerance = 1e-6`, `absolute_tolerance = 0`, `maximum_iterations = 100`, `restart_length = 30`, `verify_residual = false`. See [LINALG.md](LINALG.md). |
| `[flow.output]` | Optional `directory` for the final parallel VTU datasets. |

Put physical boundary assignments and optional CSV profiles in the separate
`bc.toml` described in [BOUNDARY-PHYSICAL.md](BOUNDARY-PHYSICAL.md). The app
uses the compressible ideal-gas model with its characteristic conditions
and evaluates the exterior state from the current interior state. Relative
paths use the process working directory. Pseudo-time iterations evaluate
boundary data at physical time zero.

## Spatial reconstruction

Choose `reconstruction` in `[flow.numerics]`.

| Value | Interior face profile |
| --- | --- |
| `first_order` | Piecewise constant. This is the default. |
| `linear` | Linear extrapolation from each cell centre to the face centroid. Use this value for existing `second_order` cases. |
| `muscl_kappa` | Line-based MUSCL with `kappa = 1/3` by default. Accepted range is [-1, 1]. |
| `tvd` | Line-based reconstruction with `tvd_limiter = "van_leer"` by default. Also accepts `minmod`, `superbee`, `monotonized_central`, `van_albada` and `koren`. |

For `muscl_kappa` and `tvd`, `skew_correction` defaults to true.
All higher-order choices use a linear interior trace at physical faces.
Set `limiter = "barth_jespersen"` to limit the reconstruction increments
component by component. The default `"none"` skips the cell limiter.
TVD's slope-ratio limiter still applies when `limiter = "none"`.

The compact inner Jacobian uses cell-centred frozen Rusanov blocks.
The finite-difference boundary Jacobian holds the current reconstruction
increment and limiter factors fixed.

## Cell gradients

The optional `[flow.numerics.gradient]` table selects the scheme.

| Key | Options and defaults |
| --- | --- |
| `method` | `"least_squares_face"` by default. Also accepts `"green_gauss_cell"` and `"green_gauss_face"`. |
| `interpolation` | Green–Gauss uses `"arithmetic"` by default or geometric `"auxiliary_weights"`. The application prepares `FaceInterpolationWeights` for the weighted method. |
| `pivot_tolerance` | Relative Cholesky pivot threshold for least squares. Default `1e-12`, with range `(0, 1)`. |

The solver computes gradients when viscous transport or higher-order
convection requires them. It exchanges primitive ghosts and fills physical
face samples before computing the owned gradients. Least-squares stencils
must span three dimensions. Viscous runs use the first gradient for a
tangential boundary correction, then recompute the gradient with the updated
samples. Any cell limiter factors are then computed from the selected
profiles. The solver exchanges gradients and factors before assembly.
Viscous transport uses the unlimited gradients.

## Molecular transport

Set `enabled = true` in `[flow.transport]` to include laminar shear, viscous
work and heat conduction. Moving solid walls can prescribe tangential
velocity on a static mesh.

| Key | Options and defaults |
| --- | --- |
| `viscosity` | `"constant"` or `"sutherland"`. Default `"constant"`. |
| `dynamic_viscosity` | Nonnegative constant dynamic viscosity. Default `1.8e-5` Pa s. |
| `reference_viscosity`, `reference_temperature`, `sutherland_temperature` | Sutherland parameters. Defaults `1.716e-5` Pa s, `273.15` K and `110.4` K. |
| `bulk_viscosity` | Nonnegative bulk viscosity. Default `0` Pa s. |
| `conductivity` | `"constant"` or `"constant_prandtl"`. Default `"constant_prandtl"`. |
| `thermal_conductivity` | Nonnegative constant conductivity. Default `0.025` W/(m K). |
| `prandtl` | Positive Prandtl number used in `lambda = mu*cp/Pr`. Default `0.72`. |

## Pseudo-time iteration

The application owns the conservative densities $U_i$, the primitives
and their gradients. The solver first exchanges the primitive
ghost values and prepares the selected reconstruction. It then computes
the spatial residual $R$ and its global
Euclidean norm. The residual contains the negative sum of the area-integrated
outward fluxes of each cell.

For each owned cell, the inviscid part of the local pseudo-step uses the face
areas $A_f$, unit normals $\mathbf n_f$, cell-centred velocity
$\mathbf u_i$, sound speed $c_i$, and volume $V_i$.

$$
\sigma_i = \sum_{f\in i} A_f (|\mathbf u_i\cdot\mathbf n_f|+c_i),
\qquad
\Delta\tau_i = \frac{\mathrm{CFL}\,V_i}{\sigma_i}.
$$

Viscous transport adds $2\nu_i/h_f$ to the speed inside the sum, where
$h_f$ is the projected centre separation at an interior face or the
centre-to-face normal distance at a wall, and

$$
\nu_i = \max\left(
  \frac{4\mu_i/3+\zeta_i}{\rho_i},
  \frac{\lambda_i}{\rho_i c_v}
\right).
$$

The solver assembles the spatial Jacobian $J$ with the [Euler operator](TRANSPORT-EULER.md),
with frozen Rusanov blocks on inner faces and the default finite-difference
boundary linearization. The [viscous operator](TRANSPORT-VISCOUS.md) adds
its compact Jacobian when enabled. The solver adds $\sigma_i/\mathrm{CFL}$ to the five
scalar diagonals of cell $i$'s BSR block. With $D_i = (V_i/\Delta\tau_i) I_5$,
the solver solves

$$
(J+D)\,\delta U=R,
\qquad U^{k+1}=U^k+\mathrm{relaxation}\,\delta U.
$$

The next residual is evaluated from the updated state. The solver stops on

$$
\|R_k\|_2 \le
\max(\mathrm{absolute\ tolerance},\,
     \mathrm{relative\ tolerance}\,\|R_0\|_2).
$$

This norm uses the unscaled owned conservative residual entries. The app
reports the initial and final norms and returns a failure exit code if the
iteration limit is reached before convergence.

The per-equation norms are the global Euclidean norms of the stored RHS
entries for mass, the three momentum components and energy. The solver
accumulates their sums of squares while it packs its linear RHS and uses one
MPI sum. The square root of the sum over all five components gives the
norm used for the steady stopping test and the zero-guess linear solve.

Detailed output measures time in milliseconds. It reports field preparation
(primitive halo, boundary samples and gradients), residual assembly (including
limiting, gradient exchange and norms), Jacobian assembly and commit, the
linear solve (including preconditioning), and the state update. The total
excludes the iteration callback and file output. Each number is the maximum of
that duration across ranks, so the phase maxima can add up to more than the
maximum total. The final residual evaluation reports only the phases it runs.


## Examples

The three case directories under `examples/transport/flow` include
`flow.toml`, `bc.toml`, the mesh generator input `mesh.toml`, and a
solver-ready `mesh.h5` with **two MPI partitions**. Run from the selected
case directory with the KRONA executables on your `PATH`.

| Case directory | Run command | What to check |
| --- | --- | --- |
| [channel](../examples/transport/flow/channel) | `mpirun -np 2 krona_flow_steady mesh.h5 flow.toml bc.toml --case channel -vv` | A 5% pressure bump in a supersonic channel decays toward the uniform inlet state. The reference values are 100,000 Pa, 300 K and 650 m/s. The mesh is 32 × 3 × 3 cells. |
| [wedge](../examples/transport/flow/wedge) | `mpirun -np 2 krona_flow_steady mesh.h5 flow.toml bc.toml --case wedge -v` | Mach 3 flow over a ramp with an oblique shock. The three-block mesh has 27,000 cells. The case uses first-order reconstruction with HLL. |
| [conduction](../examples/transport/flow/conduction) | `mpirun -np 2 krona_flow_steady mesh.h5 flow.toml bc.toml -vv` | Two plates at 300 K and 320 K give `T(x) = 300 + 2000*x` K. Outward heat flux densities are +6000 and -6000 W/m². The mesh is 12 × 2 × 2 cells. |

The final VTU index is `vtk/euler.pvtu` for the channel and wedge and
`vtk/navier_stokes.pvtu` for the conduction case. To rebuild the input
mesh and partition it for another MPI size, run the following from a case
directory. Pass the same rank count to the solver.

```sh
krona_mesh_generator mesh.toml mesh.cgns
mpirun -np N krona_mesh_partitioner mesh.cgns mesh-N.h5
```

## C++ interface

Include `<krona/solvers/flow/solver.hpp>` and link `krona::solver_flow`.
Construct `krona::solvers::flow::Solver` with the mesh, connectivity,
auxiliary geometry, halo map, the equation of state, and a device-copyable
transport property model. Bind the three writable component packs with
`fields::pack(...)`.

| Pack | Component order | Coverage |
| --- | --- | --- |
| Conservative state | `rho, rho*u, rho*v, rho*w, rho*E` | Exactly the owned cells. |
| Primitive state | `p, u, v, w, T` | Owned and ghost cells plus physical face slots when gradients are needed. |
| Primitive gradients | Three Cartesian derivatives per primitive component, in primitive order. | Owned and ghost cells. Required when `Options::needs_gradients()` is true. |

Each pack borrows the component allocations. The caller keeps these
disjoint allocations, the mesh and the communicator alive. The solver owns
the compact matrix, linear solver and workspace. Initialize the two state
packs together with `initialize(sampler)` or equivalent caller initialization.

`iterate(boundary)` completes one pseudo-time update and returns the
residual norm of its input state. The returned owned state has been updated.
Prepared gradients and ghosts describe the input state used for assembly.
`solve_steady(boundary)` also checks convergence and returns the global
residual norm and any prepared gradients at the final state. The iteration
report contains local reconstruction and Riemann fallback counts.
Set `Options::measure_iteration_time` to populate local phase durations in
seconds in the iteration report.
Call the solver collectively using the same execution instance as the
finalized boundary manager.
