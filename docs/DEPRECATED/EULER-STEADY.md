# Steady compressible flow

The steady flow solver integrates the five compressible equations for a
calorically perfect gas on a partitioned, static 3-D mesh. Both
`krona_euler_steady` and `krona_navier_stokes_steady` use the same solver.
Set `enabled = true` in `[flow.transport]` to include laminar viscous
transport and heat conduction. The default is inviscid flow.
Build with `KRONA_BUILD_APPLICATIONS=ON` and `KRONA_ENABLE_TRILINOS=ON`.

```sh
mpirun -np N krona_euler_steady mesh.h5 flow.toml bc.toml [options]
```

Use the MPI size that produced `mesh.h5`. The application accepts
`--case custom|channel|wedge`, `--vtu DIRECTORY`, `-v` and `-vv`.
`--vtu` overrides the TOML output directory. `-vv` includes the spatial
residual and Krylov iteration count at each update. The default `custom` case
uses the uniform reference state from TOML. Edit
[initial_field.hpp](../apps/euler_steady/initial_field.hpp) for a different spatial
initialization. See the [channel example](../examples/transport/euler/channel/README.md).

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
| `[flow.transport]` | `enabled = false`. See [molecular transport settings](NAVIER-STOKES-STEADY.md) for the viscous case. |
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

The solver exchanges primitive ghosts and fills physical face samples
before computing face-based least-squares gradients. Each stencil must
span three dimensions. It prepares any cell limiter factors using the
selected profiles, then exchanges the gradients and factors. An inadmissible
reconstructed trace falls back to the cell-centre state.

The compact inner Jacobian uses cell-centred frozen Rusanov blocks.
The finite-difference boundary Jacobian holds the current reconstruction
increment and limiter factors fixed.

## Pseudo-time iteration

The application owns the conservative densities $U_i$ and the primitive
field. The solver first exchanges the primitive
ghost values and prepares the selected reconstruction. It then computes
the spatial residual $R$ and its global
Euclidean norm. The residual contains the negative sum of the area-integrated
outward fluxes of each cell.

For each owned cell in an inviscid run, the solver defines a local pseudo-step using the face
areas $A_f$, unit normals $\mathbf n_f$, cell-centred velocity
$\mathbf u_i$, sound speed $c_i$, and volume $V_i$.

$$
\sigma_i = \sum_{f\in i} A_f (|\mathbf u_i\cdot\mathbf n_f|+c_i),
\qquad
\Delta\tau_i = \frac{\mathrm{CFL}\,V_i}{\sigma_i}.
$$

It assembles the spatial Jacobian $J$ from [the Euler operator](TRANSPORT-EULER.md),
with frozen Rusanov blocks on inner faces and the default finite-difference
boundary linearization. It then adds $\sigma_i/\mathrm{CFL}$ to the five
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


## C++ interface

Include `<krona/solvers/flow/solver.hpp>` and link `krona::solver_flow`.
Construct `krona::solvers::flow::Solver` with the mesh, its halo map,
writable owned conservative state and writable primitive field. Provide
the equation of state and a device-copyable transport property model.
The primitive field is ordered `p,u,v,w,T`. Include MPI ghosts and, when
`Options::needs_gradients()` is true, the physical boundary slots.

The solver owns the compact matrix, linear solver and workspace. The
caller keeps the mesh, communicator and fields alive. Initialize the
fields together using `initialize(sampler)` or an equivalent caller
initialization. `iterate(boundary)` performs one pseudo-time update and
returns the residual norm of its input state. `solve_steady(boundary)`
also checks convergence and returns the global residual norm at the final
state. Both operations complete their update before returning. The report
for an individual iteration contains local reconstruction and Riemann
fallback counts. Call these methods collectively using the same
execution instance as the finalized boundary manager.
