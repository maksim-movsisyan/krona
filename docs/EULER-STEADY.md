# Steady Euler application

`krona_euler_steady` solves the inviscid compressible Euler equations for a
calorically perfect gas on a partitioned, static 3-D mesh. It uses
selectable first- or second-order spatial reconstruction and implicit backward Euler in pseudo-time.
Build with `KRONA_BUILD_APPLICATIONS=ON` and `KRONA_ENABLE_TRILINOS=ON`.

```sh
mpirun -np N krona_euler_steady mesh.h5 euler.toml bc.toml [options]
```

Use the MPI size that produced `mesh.h5`. The application accepts
`--case custom|channel`, `--vtu DIRECTORY`, `-v` and `-vv`.
`--vtu` overrides the TOML output directory. `-vv` includes the spatial
residual and Krylov iteration count at each update. The default `custom` case
uses the uniform reference state from TOML. Edit
[initial_field.hpp](../apps/euler_steady/initial_field.hpp) for a different spatial
initialization. See the [channel example](../examples/transport/euler/channel/README.md).

## Solver configuration

The app reads `euler.toml` on rank zero and broadcasts the text before
parsing it on all ranks. `[euler.initial]` is required. All other tables
are optional.

| Table | Keys and defaults |
| --- | --- |
| `[euler.physics]` | `gamma = 1.4`, `gas_constant = 287.05` J/(kg K). |
| `[euler.initial]` | Positive `pressure` in Pa and `temperature` in K, plus `velocity = [0, 0, 0]` in m/s by default. |
| `[euler.numerics]` | `flux = "hllc"` (also `"hll"` or `"rusanov"`), `reconstruction = "first_order"` (also `"second_order"`), `limiter = "none"` (also `"barth_jespersen"` for second order). |
| `[euler.pseudo_time]` | `cfl = 1.0`, `relaxation = 1.0`, `maximum_iterations = 200`, `relative_tolerance = 1e-8`, `absolute_tolerance = 1e-8`. |
| `[euler.linear_solver]` | `method = "GMRES"`, `preconditioner = "RELAXATION"`, `relative_tolerance = 1e-6`, `absolute_tolerance = 0`, `maximum_iterations = 100`, `restart_length = 30`, `verify_residual = false`. See [LINALG.md](LINALG.md). |
| `[euler.output]` | Optional `directory` for the final parallel VTU datasets. |

Put physical boundary assignments and optional CSV profiles in the separate
`bc.toml` described in [BOUNDARY-PHYSICAL.md](BOUNDARY-PHYSICAL.md). The app
uses the compressible ideal-gas model with its characteristic conditions
and evaluates the exterior state from the current interior state. Relative
paths use the process working directory. Pseudo-time iterations evaluate
boundary data at physical time zero.

## Spatial reconstruction

Select one of the two orders in `[euler.numerics]`.

- `first_order` uses piecewise-constant cell states.
- `second_order` reconstructs the primitive field linearly to each face centroid.
  After exchanging primitive ghosts, the app fills physical face samples,
  computes face-based least-squares gradients and exchanges them. Each stencil
  must span three dimensions. Set `limiter = "barth_jespersen"` to bound each
  component's face extrapolation by the local stencil values. The limiter can
  reduce the order locally, for example near extrema or discontinuities.

The wall samples impose the inviscid slip trace. The boundary Riemann solver
uses an exterior state built from the reconstructed interior trace.
The compact Jacobian continues to use cell-centred frozen Rusanov blocks
on inner faces. On a physical boundary the finite-difference Jacobian
holds the current reconstruction increment and limiter factors fixed.

## Pseudo-time iteration

The app owns the conservative densities $U_i$ and keeps an extended
primitive field for the spatial operator. It first exchanges the primitive
ghost values and prepares the selected reconstruction. It then computes
the spatial residual $R$ and its global
Euclidean norm. The residual contains the negative sum of the area-integrated
outward fluxes of each cell.

For each owned cell, the app defines a local pseudo-step using the face
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
the app solves

$$
(J+D)\,\delta U=R,
\qquad U^{k+1}=U^k+\mathrm{relaxation}\,\delta U.
$$

The next residual is evaluated from the updated state. The app stops on

$$
\|R_k\|_2 \le
\max(\mathrm{absolute\ tolerance},\,
     \mathrm{relative\ tolerance}\,\|R_0\|_2).
$$

This norm uses the unscaled owned conservative residual entries. The app
reports the initial and final norms and returns a failure exit code if the
iteration limit is reached before convergence.