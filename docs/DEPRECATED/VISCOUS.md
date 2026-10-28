# Local viscous transport

`krona::numerics::viscous` evaluates local Newtonian stresses, Fourier heat flux,
and total-energy viscous flux. It belongs beside the local Riemann solvers in
`numerics/viscous`, while material correlations remain in `transport/properties`.

The production headers depend only on KRONA execution types and Kokkos. There is
no EOS, mesh, MPI, field-storage, boundary-manager or Riemann-solver dependency.
The same constitutive calculations can be used by conservative compressible,
SIMPLE and coupled pressure-based spatial operators.

```cpp
#include <krona/numerics/viscous/flux.hpp>
namespace viscous = krona::numerics::viscous;
using Real = krona::ComputeReal;
```

## Physical contract

The implementation is three-dimensional, isotropic and Newtonian:

\[
\theta=\nabla\cdot\mathbf u,\qquad
\tau=\mu\left[\nabla\mathbf u+(\nabla\mathbf u)^T-\tfrac23\theta I\right]
      +\zeta\theta I.
\]

`mu` is dynamic shear viscosity; `bulk` is bulk viscosity zeta. Both must be
finite and nonnegative. `bulk=0` is the default **Stokes hypothesis**, not a
statement that the fluid is incompressible. The second viscosity coefficient
multiplying `theta*I` in an alternative notation is `zeta-2*mu/3`; do not pass
that generally negative quantity as `bulk`.

Pressure is excluded from `SymmetricStress`: it is already part of the inviscid
momentum flux. At exact zero divergence, the law reduces to
`mu*(grad(u)+transpose(grad(u)))`. The implementation uses the supplied gradient
and does not force its trace to zero during a pressure-correction iteration.
An operator using a different constitutive/discrete form can supply its own
symmetric tensor to `project_traction`.

For a unit normal n, normally directed from owner to neighbour (outward on a
physical boundary):

\[
\mathbf t=\tau\mathbf n,\qquad q_n=-\lambda\nabla T\cdot\mathbf n.
\]

`traction` means positive `tau*n`; it is **not** the outward diffusive momentum
flux, which has the opposite sign. `q_n` is the physical outward heat flux.
Positive `q_n` removes heat from the cell. Use equal and opposite integrated
flux contributions in the two adjacent cell balances. Conductivity lambda must be finite and
nonnegative. The module receives coefficients and gradients at the evaluation
point; it neither evaluates an EOS nor interpolates coefficients.

| Quantity | SI units |
| --- | --- |
| Stress, traction | Pa |
| Dynamic and bulk viscosity | Pa s |
| Conductivity | W/(m K) |
| Outward heat flux, total-energy flux density | W/m² |
| Viscous dissipation | W/m³ |

Use consistent dimensional or nondimensional inputs. No unit conversion occurs.

## Velocity-gradient layout

```cpp
viscous::VelocityGradient<Real> g{
    du_dx, du_dy, du_dz,
    dv_dx, dv_dy, dv_dz,
    dw_dx, dw_dy, dw_dz
};
// Alternatively, from three vectors exposing x,y,z:
const auto g2 = viscous::velocity_gradient<Real>(grad_u, grad_v, grad_w);
```

Rows correspond to velocity components; columns correspond to coordinate
derivatives: `g_ij = partial(u_i)/partial(x_j)`. This contract matters in both
stress construction and dissipation. No temperature gradient is part of this
structure. A 2-D flow embedded in a 3-D mesh still uses this three-dimensional
constitutive law and the factor `2/3`.

Normal, temperature-gradient and velocity arguments can be any simple
Cartesian objects exposing `.x`, `.y`, `.z`, including the existing boundary or
Riemann vector types. They need no operators or member functions. They are read
and converted to the requested arithmetic type, without borrowing arrays.
`Traction<Real>` is a small physical result with x/y/z components, rather than a
new general-purpose vector algebra API.

## Momentum, including nonthermal incompressible use

```cpp
const viscous::NewtonianStress law{};
const auto stress = law(g, mu);                  // Default bulk viscosity is zero.
const auto stress_bulk = law(g, mu, bulk);
const auto traction = law.traction(g, normal, mu, bulk);

if (!traction.valid()) {
    // Record traction.status in the consuming operator's diagnostics.
} else {
    const auto tx = traction.value.x;
    const auto ty = traction.value.y;
    const auto tz = traction.value.z;
}
```

A stress result contains `value.xx`, `yy`, `zz`, `xy`, `xz`, `yz`. Off-diagonal
entries are actual tensor entries, not doubled engineering shear components.
The tensor is symmetric.

`NewtonianStress::traction` uses the same constitutive formula directly and
avoids a second validation of a materialized stress result. If a stress has
already been computed or is supplied by another model:

```cpp
const auto traction = viscous::project_traction(stress, normal);
// Also accepts a plain SymmetricStress<Real> instead of Result<...>.
```

This path needs no pressure, density, EOS, temperature or temperature gradient.
It is the shared physical building block for a SIMPLE momentum equation as well
as a coupled conservative momentum equation. Assembly and implicit/explicit
splitting remain responsibilities of the spatial operator.

## Heat transfer and coefficient models

```cpp
// property_model is the previously constructed TransportProperties object.
const auto coefficients = property_model.evaluate<Real>(face_state);
if (!coefficients.valid()) {
    // Handle an invalid material-property result before using it.
}

const auto traction = viscous::NewtonianStress{}.traction(
    g, normal, coefficients.dynamic_viscosity, bulk);
const auto heat = viscous::FourierHeatFlux{}.template operator()<Real>(
    grad_temperature, normal, coefficients.thermal_conductivity);
const auto surface = viscous::surface_transport(traction, heat);
```

`surface.value` contains `traction` and `outward_heat_flux`. Both remain per unit
area. The `surface_transport` helper propagates a failed traction or heat result.
It also checks finite physical values before returning success.

If heat transfer is absent, omit it entirely:

```cpp
const auto surface = viscous::surface_transport(traction); // q_n = 0
```

No dummy thermal state or gradient is accessed. For a boundary with a directly
prescribed outward heat flux, one can also supply `Result<Real>{prescribed_qn}`
as the second argument; the combination checks its finite value. Boundary
condition evaluation itself remains outside this module.

`FourierHeatFlux` defaults to ComputeReal. For a different arithmetic type,
request it explicitly as above; its coefficient is then converted to that type.
For Newtonian stress, the arithmetic type comes from `VelocityGradient<Real>`.
Normal-unit tolerance accounts for both arithmetic precision and the stored
normal component precision. Normalization is never performed internally.

## Total-energy adapters and signs

The five-component container is selected by the caller. It must be an aggregate
constructible from `{mass, {mx,my,mz}, energy}`, exposing `.mass`, `.momentum.x`,
`.momentum.y`, `.momentum.z`, and `.energy`, all with the same scalar type as the
local calculation. This avoids an unnecessary dependency on the Riemann module
or a second, competing five-component flow type.

The existing `riemann::Conservative<Real>` already satisfies this contract:

```cpp
#include <krona/numerics/riemann/types.hpp>
using Flux = krona::numerics::riemann::Conservative<Real>;

const auto fv = viscous::outward_flux<Flux>(surface, face_velocity);
if (fv.valid()) {
    // inviscid.flux and fv.value use the same unit normal and area convention.
    const auto total_outward_flux = inviscid.flux + fv.value;
}
```

For an outward-flux balance,

\[
\partial_t U+\nabla\cdot(F_{\mathrm{inv}}+F_{\mathrm{diff}})=0,
\qquad
F_{\mathrm{diff},n}=(0,-\mathbf t,q_n-\mathbf u\cdot\mathbf t).
\]

`outward_flux` returns this diffusive contribution. If viscous transport is
instead written on the right-hand side,

\[
\partial_t U+\nabla\cdot F_{\mathrm{inv}}=\nabla\cdot F_{\mathrm{visc}},
\qquad
F_{\mathrm{visc},n}=(0,\mathbf t,\mathbf u\cdot\mathbf t-q_n),
\]

use `rhs_flux<Flux>(surface, face_velocity)`. These are opposite conventions for
the same contribution. Never add `rhs_flux` to an outward inviscid flux.

The velocity is the **face velocity used for viscous work**, including the
viscous face trace at a wall. The local module does not choose it from an owner,
neighbour, reconstruction, or inviscid mirror state.

For an integrated outward flux:

```cpp
const auto integrated = viscous::integrated_outward_flux<Flux>(
    surface, face_velocity, face_area);
```

Area is separate and strictly positive. No area vector is constructed, no square
root is taken, and no division by area is needed. Multiply by area exactly once.
Momentum-only/thermal-only operators can similarly multiply their traction or
heat flux components by the supplied area when assembling their own balances.

## Viscous dissipation for thermal equations

```cpp
const auto phi = viscous::NewtonianStress{}.dissipation(g, mu, bulk);
```

This evaluates the source density

\[
\Phi=\tau:\nabla\mathbf u
=2\mu D_{\mathrm{dev}}:D_{\mathrm{dev}}+\zeta\theta^2,
\qquad D=\tfrac12(\nabla\mathbf u+(\nabla\mathbf u)^T).
\]

It uses the nonnegative strain-based expression. Rigid rotation does not
produce viscous stress or dissipation. Multiply Phi by cell volume for an
integrated cell source when the governing equation requires it.

Do not add Phi as an extra source to the total-energy equation when its viscous
work is already accounted for by the conservative face flux. Temperature,
internal-energy and enthalpy formulations require their own corresponding
energy balance, pressure terms and thermodynamic factors. This helper does not
assemble those equations.

## Status, validation and execution

Every checked operation returns `Result<Value>` with `.value`, `.status` and
`.valid()`. Only `Status::Success` is valid. Failures carry NaNs in their numeric
result fields; they do not abort, launch diagnostics, or silently substitute a
zero flux.

| Status | Meaning |
| --- | --- |
| `InvalidGradient` | Nonfinite required velocity/temperature gradient |
| `InvalidCoefficient` | Nonfinite or negative mu, bulk viscosity or conductivity |
| `InvalidNormal` | Nonfinite or nonunit normal |
| `InvalidStress` | Nonfinite externally supplied stress tensor |
| `InvalidVelocity` | Nonfinite velocity supplied to the energy adapter |
| `InvalidArea` | Nonfinite or nonpositive integrated-flux area |
| `NonfiniteResult` | Unrepresentable/nonfinite calculated result |

Zero coefficients are allowed. Required gradients are still checked even when
a coefficient is zero; to omit thermal data, use the explicit nonthermal path.
Checks do not establish gradient accuracy, mesh consistency, coefficient
interpolation quality, or a material correlation's physical validity range.
Extreme intermediate overflow/underflow is not universally prevented.

There is no stateful manager, array allocation, history, fence, MPI exchange or
internal kernel launch. Use these functions directly inside your existing
Kokkos kernel. Host/device arithmetic is the same, subject to the selected
precision and backend floating-point behavior. A surrounding parallel_for may
execute asynchronously; input lifetimes, gradient/halo readiness and kernel
ordering remain the caller's responsibility. Collect returned statuses into an
existing operator status buffer and synchronize only when that operator needs
host-visible completion.

## Limits

This module does not calculate/reconstruct gradients, interpolate face values,
choose nonorthogonal corrections, apply boundary conditions, assemble matrices
or residuals, select a stable time step, or produce Jacobians. Those are separate
operator/discretization responsibilities.

It supports the stated Newtonian/Fourier continuum model, independently of the
EOS. Anisotropic conduction, mixture enthalpy diffusion, non-Newtonian or
nonsymmetric constitutive laws require explicit extensions. A supplied
symmetric stress can already be projected without using NewtonianStress.

## Formula reference

[MIRGE-Com governing equations](https://mirgecom.readthedocs.io/en/latest/mathematical/fluid.html)
provides the Newtonian stress, Fourier law and conservative viscous flux signs.
