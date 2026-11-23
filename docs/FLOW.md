# Conservative flow states

Include `<krona/flow/conservative.hpp>` or link `krona::flow` in CMake. The header
uses `math::Vector3<Scalar>` for Cartesian velocity and momentum.

| Type | Values |
|---|---|
| `flow::Conservative<Scalar>` | Mass, momentum and total energy per unit volume |
| `flow::InternalEnergyState<Scalar>` | Mass density, velocity and internal energy per unit mass |

## From specific quantities to conservative densities

Use `flow::to_conservative(rho, velocity, e)`. The caller supplies finite inputs,
positive mass density and a consistent energy reference. The conversion
computes momentum density as `rho * velocity` and energy density as
`rho * (e + |velocity|^2 / 2)`. The caller checks output representability
as required by its algorithm.

## From conservative densities to thermodynamic inputs

`flow::try_split_conservative(U, local)` recovers velocity and subtracts its
kinetic energy from total energy per unit mass. It returns false for invalid
density, nonfinite input, or a nonfinite intermediate or recovered value
that prevents the conversion. On failure, it leaves `local` unchanged.
The internal energy keeps the original material's reference and can be negative.

```cpp
namespace flow = krona::flow;
namespace thermo = krona::thermodynamics;

flow::InternalEnergyState<double> local;
thermo::AcousticState<double> acoustic;
// gas is a validated thermo::IdealGas<double> and U is a flow::Conservative<double>.
if (flow::try_split_conservative(U, local) &&
    gas.try_acoustic_from_rhoe(local.density, local.internal_energy, acoustic)) {
    // Use local.velocity, acoustic.pressure and acoustic.temperature.
}
```

Include `<krona/thermodynamics/ideal_gas.hpp>` for the ideal-gas calls in the
example. A different material can use the same first conversion and pass
`local.density` and `local.internal_energy` to its own EOS. Each function
computes one state in the calling host or device context.

## State transformation Jacobians

Include `<krona/flow/state_jacobian.hpp>` and pass an
`InternalEnergyState<Real>` with positive density and an admissible specific
internal energy. The EOS supplies `rhoe_derivatives(density, internal_energy)`,
which returns the partial derivatives of pressure and temperature.

- `primitive_from_conservative_jacobian(state, eos)` returns the derivative of
  `(p, u, v, w, T)` with respect to `(rho, rho*u, rho*v, rho*w, rho*E)`.
- `conservative_from_primitive_jacobian(state, eos)` returns the reverse
  derivative. The EOS map from `(rho, e)` to `(p, T)` must have an invertible
  derivative at the supplied state.

Both functions return a host/device `math::Matrix5<Real>` with one row per
output and one column per input.
