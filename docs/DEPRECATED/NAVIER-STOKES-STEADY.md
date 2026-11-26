# Steady Navier–Stokes application

`krona_navier_stokes_steady` solves the stationary compressible
Navier–Stokes equations for a calorically perfect gas on a partitioned
static 3-D mesh. Enable `KRONA_BUILD_APPLICATIONS` and
`KRONA_ENABLE_TRILINOS`. Use the rank count that produced the solver mesh.

```sh
mpirun -np N krona_navier_stokes_steady mesh.h5 navier_stokes.toml bc.toml -vv
```

Use the common `[flow]` configuration from the
[steady compressible flow guide](EULER-STEADY.md) and set
`enabled = true` in `[flow.transport]`. The separate physical boundary
file supports the assignments and profiles documented in
[BOUNDARY-PHYSICAL.md](BOUNDARY-PHYSICAL.md). On a static mesh, a moving
solid wall can prescribe tangential motion. The output directory can be
overridden with `--vtu DIRECTORY`.

The remaining keys in `[flow.transport]` select molecular properties.

| Key | Options and defaults |
| --- | --- |
| `viscosity` | `"constant"` or `"sutherland"`. Default `"constant"`. |
| `dynamic_viscosity` | Nonnegative constant dynamic viscosity. Default `1.8e-5` Pa s. |
| `reference_viscosity`, `reference_temperature`, `sutherland_temperature` | Sutherland parameters. Defaults `1.716e-5` Pa s, `273.15` K and `110.4` K. |
| `bulk_viscosity` | Nonnegative bulk viscosity. Default `0` Pa s. |
| `conductivity` | `"constant"` or `"constant_prandtl"`. Default `"constant_prandtl"`. |
| `thermal_conductivity` | Nonnegative constant conductivity. Default `0.025` W/(m K). |
| `prandtl` | Positive Prandtl number used in `lambda = mu*cp/Pr`. Default `0.72`. |

## Iteration

Each update exchanges primitive ghosts and prepares physical face samples.
A first least-squares gradient supplies one explicit boundary
correction pass. The solver recomputes the gradients from those corrected
samples and exchanges the resulting gradients. Viscous transport uses these
unlimited gradients with any of the convective reconstruction schemes.
The optional Barth–Jespersen limiter applies to Euler reconstruction.
Physical conditions are evaluated at physical time zero.

The [Euler](TRANSPORT-EULER.md) and [viscous](TRANSPORT-VISCOUS.md)
operators add to one residual and one 5 by 5 BSR system. The solver uses
the acoustic estimate `|u_i·n_f|+c_i` plus a molecular scale
`2*nu_i/h_f` in its pseudo-time diagonal, where

$$
\nu_i = \max\left(
 \frac{4\mu_i/3+\zeta_i}{\rho_i},
 \frac{\lambda_i}{\rho_i c_v}
\right).
$$

Here $h_f$ is the projected centre separation for an interior face and
the centre-to-face distance for a physical boundary. The local diagonal
shift equals the sum over faces of area times the combined speed,
divided by CFL. This estimate controls the pseudo-time iteration.
The converged equations use the spatial residual. The application
solves for an increment of the conservative densities and regenerates
primitives from the updated state.

The [conduction example](../examples/transport/navier_stokes/conduction/README.md)
uses isothermal plates and symmetry boundaries.
