# Molecular transport properties

`krona::transport::properties` supplies dynamic viscosity and thermal conductivity
as local, device-callable functions of a state. It is independent of mesh
storage, gradients, boundary conditions, equation assembly and the EOS type.

Headers live in `include/krona/transport/properties/`. Use the umbrella header:

```cpp
#include <krona/transport/properties/properties.hpp>
namespace properties = krona::transport::properties;
```

This module evaluates material coefficients. It does not calculate viscous
stresses, heat fluxes or face interpolation; those belong to the consuming
numerical operator.

## Architecture

A `TransportProperties<Viscosity, Conductivity>` stores two policies by value:

1. Evaluate dynamic viscosity once.
2. Pass this viscosity and the state to the conductivity policy.
3. Return `Coefficients<Real>`.

The policies can be selected independently. There is no inheritance, virtual
dispatch, runtime model registry or hidden dependence on a specific EOS.

| Model | Parameters | State members read |
| --- | --- | --- |
| `ConstantViscosity<Parameter>` | `value` | None |
| `SutherlandViscosity<Parameter>` | `reference_viscosity`, `reference_temperature`, `sutherland_temperature` | `temperature` |
| `ConstantConductivity<Parameter>` | `value` | None; supplied viscosity is ignored |
| `ConstantHeatCapacity<Parameter>` | `value`, representing specific cp | None |
| `ConstantPrandtl<HeatCapacity, Parameter>` | `prandtl`, `heat_capacity` | Whatever the heat-capacity policy needs |

`Parameter` is the storage type of the model parameters, defaulting to
`ComputeReal`. The arithmetic type is selected independently during evaluation.

The supplied policies own scalar parameters only. They allocate no memory and
hold no references to the thermodynamic state. A custom policy may borrow arrays
or store an EOS explicitly, with the corresponding lifetime and device-copyability
requirements under the caller's control.

## Sutherland viscosity and constant Prandtl number

```cpp
using Real = krona::ComputeReal;

krona::thermodynamics::IdealGas<Real> gas;
gas.validate();

const auto model = properties::TransportProperties{
    properties::SutherlandViscosity{
        static_cast<Real>(1.716e-5), // mu_ref, Pa s
        static_cast<Real>(273.15),  // T_ref, K
        static_cast<Real>(110.4)    // S, K
    },
    properties::ConstantPrandtl{
        static_cast<Real>(0.72),
        properties::ConstantHeatCapacity{gas.cp()}
    }
};
model.validate(); // Host-side parameter validation before kernel submission.

const auto state = gas.from_pT(Real{101325}, Real{300});
const auto coefficients = model(state);
// coefficients.dynamic_viscosity
// coefficients.thermal_conductivity
```

Include `<krona/thermodynamics/ideal_gas.hpp>` when using the EOS in this example.
The property module itself does not include thermodynamics headers.

The numbers above are an illustrative air parameter set, not defaults or a
universal validity range. Every model parameter is explicit. Sutherland uses
absolute temperature and evaluates

\[
\mu(T)=\mu_{\mathrm{ref}}
\left(\frac{T}{T_{\mathrm{ref}}}\right)^{3/2}
\frac{T_{\mathrm{ref}}+S}{T+S}.
\]

The implementation uses the equivalent expression

\[
\mu(T)=\mu_{\mathrm{ref}}\sqrt{\frac{T}{T_{\mathrm{ref}}}}
\frac{1+S/T_{\mathrm{ref}}}{1+S/T},
\]

which avoids forming the three-halves power. There is one square root per
evaluation and no general-purpose `pow` call. This improves intermediate range
for ordinary model parameters, but does not make arbitrary extreme parameter
ratios representable.

The constant-Prandtl conductivity is

\[
\lambda=\frac{\mu c_p}{\mathrm{Pr}}.
\]

It reuses the viscosity computed by `TransportProperties`; Sutherland is not
called a second time. The heat-capacity policy is also evaluated once.

`ConstantHeatCapacity{gas.cp()}` stores a **snapshot value**. It does not retain
or query `gas`. If you subsequently change the EOS parameters, rebuild the
property model or update its explicit heat capacity and validate again.

The existing `thermodynamics::State<Real>` is accepted directly. It contains
temperature but no cp; the heat-capacity dependency is therefore explicit.
For a nonconstant cp, supply a different heat-capacity policy.

## Constants and states without temperature

```cpp
const auto liquid = properties::TransportProperties{
    properties::ConstantViscosity{static_cast<Real>(1.e-3)},
    properties::ConstantConductivity{static_cast<Real>(0.6)}
};
liquid.validate();

struct NonthermalState {};
const auto coefficients = liquid(NonthermalState{});
```

Constant models do not inspect the state, so a temperature member is not
required. This is also true of constant viscosity combined with constant cp and
constant Pr. A temperature-dependent policy requires the relevant state data;
there is no implicit temperature or silent default.

The returned viscosity is **dynamic viscosity** mu, in Pa s in SI. It is not
kinematic viscosity `mu/rho`. Thermal conductivity is in W/(m K), and specific
cp is in J/(kg K). The module performs no unit conversion or nondimensionalization.
Use consistent units for your entire model.

Thermal diffusivity `lambda/(rho*cp)` and scalar equation diffusion coefficients
are different quantities; the consuming equation determines which coefficient
its flux needs.

## Arithmetic precision and standalone models

```cpp
const auto normal = model(state);                 // Coefficients<ComputeReal>
const auto single = model.evaluate<float>(state); // Coefficients<float>
const auto wide = model.evaluate<double>(state);  // Coefficients<double>
```

All supplied policies convert parameters and required state members to the
requested `Real` before doing arithmetic. Using double parameter storage does
not silently force double arithmetic in `evaluate<float>`. A wider evaluation
cannot recover precision already lost in stored parameters or input fields.

Individual policies expose the same explicit evaluation convention:

```cpp
const auto mu = model.viscosity.evaluate<Real>(state);
const auto lambda = model.conductivity.evaluate<Real>(state, mu);
```

`evaluate(state)` defaults to `ComputeReal`; the composition's `operator()` is a
convenience for that default. There are no array casts or temporary buffers.

## Custom models

A viscosity policy provides:

```cpp
template<class Real, class State>
KOKKOS_INLINE_FUNCTION Real evaluate(const State& state) const;
```

A conductivity policy provides:

```cpp
template<class Real, class State>
KOKKOS_INLINE_FUNCTION Real evaluate(const State& state, Real viscosity) const;
```

A heat-capacity policy has the same signature as a viscosity policy. For example:

```cpp
struct LinearHeatCapacity {
    double cp_reference;
    double temperature_reference;
    double slope;

    void validate() const
    {
        krona::mpi::require(
            std::isfinite(cp_reference) && cp_reference > 0 &&
            std::isfinite(temperature_reference) && temperature_reference > 0 &&
            std::isfinite(slope),
            "LinearHeatCapacity: invalid parameters");
    }

    template<class Real, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state) const
    {
        return static_cast<Real>(cp_reference) + static_cast<Real>(slope)*
            (static_cast<Real>(state.temperature)-static_cast<Real>(temperature_reference));
    }
};

const auto model = properties::TransportProperties{
    properties::SutherlandViscosity{1.716e-5, 273.15, 110.4},
    properties::ConstantPrandtl{
        0.72, LinearHeatCapacity{1005., 300., 0.1}
    }
};
model.validate();
```

Include `<cmath>` for host-side `std::isfinite` in custom validation code.
This linear cp model is an interface example, not a calibrated material model.
Choose its temperature range and constants consistently with the EOS. The
property module does not modify EOS enthalpy or internal energy to match it.

An independent conductivity model can use temperature, pressure, density or
composition from the same state, ignoring the supplied viscosity. It does not
need a cp policy at all. No changes to the composition or consuming operator
are required when switching between these approaches.

A host `validate() const` is optional on custom policies. The composition calls
it when present, and `ConstantPrandtl` forwards validation to its cp policy.
Custom policies without this method are responsible for having no unchecked
configurable parameters, or for being validated explicitly by their owner.

## Validation and error contract

Call `model.validate()` on the host before use, and again after changing
parameters. It follows the existing KRONA `mpi::require` error convention. It
does not launch kernels, perform collectives, inspect field data or synchronize
pending device work. Do not call it from a device kernel.

| Parameter | Host validation |
| --- | --- |
| Constant mu, constant lambda | Finite and nonnegative |
| Sutherland mu_ref, T_ref | Finite and strictly positive |
| Sutherland S | Finite and nonnegative |
| Constant cp, Pr | Finite and strictly positive |

Zero constant coefficients are allowed explicitly, including zero conductivity
for a model without heat conduction. Sutherland is a positive-viscosity model;
zero reference viscosity is rejected. Default-constructed zero reference
parameters and zero cp/Pr are not a configured gas model and fail validation.

Pointwise evaluation has a separate contract:

- Sutherland returns NaN for nonfinite or nonpositive temperature. It does not
  clamp temperature, substitute a reference state or abort.
- `ConstantPrandtl` returns NaN for nonfinite or nonpositive computed cp, even
  when the supplied viscosity is zero.
- Constant policies ignore unused state data.
- Other invalid results, including overflow, propagate to the returned coefficients.

Use the optional `coefficients.valid()` check in the consuming kernel. It
requires finite, nonnegative mu and lambda. It does not check the physical
validity range of a correlation, state consistency with the EOS, strictly
positive coefficients, or underflow to zero. An operator requiring stricter
conditions must enforce them itself.

There is no internal error flag allocation or status-buffer ownership. The
calling operator can record failures in its existing asynchronous diagnostics.
Do not interpret an invalid result as permission to continue a solve with it.

## Kokkos execution and storage

Models can be copied into a kernel and evaluated inline:

```cpp
template<class Model, class States, class Output>
struct EvaluateProperties {
    Model model;
    States states;
    Output output;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const
    {
        output(cell) = model(states(cell));
        // A consuming operator can also record !output(cell).valid().
    }
};

using Exec = krona::execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using State = krona::thermodynamics::State<Real>;
using States = Kokkos::View<State*, Space>;
using Output = Kokkos::View<properties::Coefficients<Real>*, Space>;

// states is populated before submission; its owner must outlive the kernel.
States states("states", count);
Output output("properties", count);
// ... initialize states ...
Exec exec;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
Kokkos::parallel_for("properties/evaluate", Policy(exec, 0, count),
    EvaluateProperties<decltype(model), States, Output>{model, states, output});
```

This kernel is an optional precomputation pattern, not a required storage
layout. Evaluate directly from local state values inside a flux kernel when
that avoids unnecessary coefficient arrays. Conversely, cache coefficients
explicitly when several operators reuse them and the state is unchanged.
The property module makes neither decision for you.

A pointwise call is ordinary synchronous arithmetic within its calling thread.
The surrounding Kokkos kernel may execute asynchronously. There are no internal
allocations, launches, fences or MPI exchanges. Preserve state/model lifetimes,
order dependent work on the appropriate execution-space instance, and synchronize
only when the consuming application requires completion.

## Scope and extensions

The initial models describe scalar, isotropic molecular properties. They can be
used with gases or liquids when the selected correlation is appropriate; the
module does not restrict the EOS to an ideal gas. Sutherland itself is a gas
correlation, not a universal liquid-viscosity model.

Bulk viscosity, non-Newtonian constitutive laws, turbulent/eddy coefficients,
anisotropic conductivity, mixture transport and species diffusion are not
implemented. They require explicit model contracts; they must not be inferred
from Sutherland mu or a constant Pr. Additional scalar viscosity/conductivity
policies can already be added through the existing interface.

The next viscous-flux module can consume `Coefficients<Real>` without knowing
which models produced them.

## Formula references

- [NASA Wind-US transport property definitions](https://www.grc.nasa.gov/www/winddocs/user/files.html)
  gives the reference-temperature form of Sutherland's law.
- [CGNS governing flow equations](https://cgns.org/standard/SIDS/equation.html)
  describes transport model conventions, including constant-Prandtl conductivity.
