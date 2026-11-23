# Compressible viscous transport

`transport::viscous::ViscousOperator` adds molecular momentum diffusion,
Fourier heat transfer and viscous work to a compressible finite-volume
residual and a compact Jacobian. Link `krona::transport_viscous`. The
operator accepts a writable block view and an entry map. The optional native
Trilinos adapter is in `krona/transport/block_assembly.hpp`. It uses
`krona::linalg`.

The operator borrows a validated `transport::OperatorPlan<Exec, Real>`
with a host patch catalog. Build one with `viscous::ViscousPlan` or
reuse an existing `euler::EulerPlan`. Keep the plan, its mesh storage,
the input fields and the destination arrays alive through completion.
Construction checks the material and EOS parameters on the host.
The operator uses Newtonian stress and Fourier conduction from
[the local viscous module](NUMERICS-VISCOUS.md) and a molecular property model
such as [TransportProperties](TRANSPORT-PROPERTIES.md).

## Inputs and outputs

| Input | Contract |
| --- | --- |
| Primitive pack | `p,u,v,w,T` in that order. Pressure and absolute temperature must be positive. |
| Gradient pack | `3*component + axis`. Velocity and temperature gradients are read. Use the unlimited gradients in the viscous operator. |
| Inner visits | Owned-cell rows. The full face range reads owned and MPI ghost states and gradients. The owned-owned range can run while ghost data are in flight. |
| Boundary visits | Owned-cell fields and the finalized physical boundary manager on the same mesh, numbering and execution instance. The flux is evaluated from the condition and owner state. |
| Residual and matrix | Writable additive destinations. The residual uses mass, Cartesian momentum and total energy. The matrix uses the corresponding conservative densities as columns. Input and output storage must be disjoint. |

The caller clears the destinations at the start of a new assembly and owns
gradient computation, physical boundary-slot publication, halo exchange
and the linear solver. `Real` controls local arithmetic,
`Accumulator` controls inner row sums and area integration, and the
destination views determine stored precision. Boundary additions are atomic
in the destination precision. Positive area and projected centre distances
are required for the processed faces.

For an owner-oriented unit normal, the outward flux density is

$$
 F_d = (0,-\tau n,-\lambda\nabla T\cdot n-u_f\cdot(\tau n)).
$$

The residual subtracts the integrated outward flux. The Jacobian
differentiates the positive outward flux sum. Both sides of a shared face
use the same oriented flux, with opposite signs in their balances. Each
rank writes its own rows. There is zero molecular mass transport. The
energy equation includes viscous work through the face flux.

## Face discretization

For an inner face let `d = x_N - x_O`, `h = d·n > 0` and
`a = clamp((x_f - x_O)·n/h, 0, 1)`. The interpolated gradient is
`g = (1-a) g_O + a g_N`. For each velocity component and temperature,

$$
 \nabla_f \phi = g + n\frac{\phi_N-\phi_O-g\cdot d}{h}.
$$

This retains the tangential gradient and recovers the exact gradient of an
affine field when the supplied cell gradients are exact. Velocity and
temperature at the face use the same interpolation plus the explicit
skew correction `g·(x_f - x_O - a*d)`. If the corrected absolute
temperature is inadmissible, coefficient evaluation uses the positive
interpolated temperature. Pressure for the property provider is the
interpolated cell pressure. Material laws evaluate at this face state.

| Physical condition | Viscous flux |
| --- | --- |
| Isothermal wall | Prescribed wall velocity and temperature. Replaces the normal gradients using the physical boundary relations. |
| Adiabatic wall and heat-flux wall | Prescribed wall velocity and outward heat flux, including zero for adiabatic walls. Uses wall velocity in viscous work. |
| Slip wall and symmetry | Projects traction onto the normal, enforces zero heat flux and uses the physical face velocity. The normal viscous stress can contribute to momentum. |
| Supersonic and total-conditions inlet | Uses the physical inlet trace to set normal derivatives from the owner extrapolation. |
| Supersonic and pressure outlet | Outgoing flow uses zero normal derivatives of velocity and temperature, retaining the tangential gradients in the stress. At a pressure-outlet backflow, the inlet trace supplies the normal derivatives. |
| Custom physical condition | Its evaluated face trace supplies velocity and temperature for the normal derivatives. |

At the boundary, the tangential extrapolation uses
`x_f - x_O - h*n`. `WallHeatFlux` adapts the property model to
the physical manager when preparing gradient samples: a nonzero $q_n$
uses `dT/dn = -q_n/lambda(W_O)`. The energy flux uses the prescribed
$q_n$ directly. A nonzero prescribed heat flux needs positive conductivity.
The boundary residual and matrix use the same conditions, geometry and time.
Use the same equation of state in the operator and the boundary model.

## Compact Jacobian

The Jacobian holds molecular coefficients and the cell gradients fixed.
It includes derivatives of the two-point contribution and both factors in
the viscous work. The off-diagonal blocks connect neighbours by face.
The interior and boundary derivatives use the shared `flow` conversion
from primitive to conservative derivative columns.

`WallLinearization` uses analytic affine relations for no-slip walls,
slip walls and symmetry. Other conditions use `FiniteDifferenceBoundary`.
It perturbs the owner primitive state in `Real`, evaluates the flux with
frozen transport coefficients and gradients, and applies the analytic
chain rule. At a switch between characteristic branches the difference
gives a local numerical slope. Boundary blocks add to the owner diagonal.
To replace the default, pass a device-callable policy as the final
argument of `add_boundary_jacobian_async()` or `add_jacobian_async()`.
It receives `(BoundaryLinearizationInput, condition, model, properties)`
and returns `BoundaryJacobian<Real>` per unit area.

## Example assembly

The example assumes the primitive and gradient ghosts have been published,
the caller has initialized `residual` and `target.blocks`, and `plan` is a
`ViscousPlan` or `EulerPlan`.

```cpp
#include <krona/transport/viscous/operator.hpp>

namespace vi = krona::transport::viscous;

const vi::ViscousOperator viscous{plan, properties, gas};
vi::AssemblyStatus status(exec);
status.begin_async();
viscous.add_residual_async(values, gradients, residual, boundary, status);
viscous.add_jacobian_async(values, gradients, target, boundary, status);
status.check();
```

The operator can add to buffers that already contain Euler contributions.
Apply the pseudo-time diagonal and commit the complete matrix before
solving.

| Operation | Completion |
| --- | --- |
| Plan construction | Local and blocking while mesh validation completes. |
| `status.begin_async()` and `add_*_async()` | Queue local work on the plan's execution instance. |
| `status.check()` | Copies the diagnostic flags to the host, fences the instance and handles errors through MPI. |

The same execution instance orders preparation, residual and Jacobian
contributions. Finish external producers or exchanges before their
outputs are consumed. A failed diagnostic batch may have partially
written to its destinations.

See the [steady Navier–Stokes application](NAVIER-STOKES-STEADY.md)
for configuration and the pseudo-time iteration.
