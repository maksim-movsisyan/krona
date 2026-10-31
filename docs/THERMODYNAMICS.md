# Thermodynamics

`krona::thermodynamics` provides local material closures and conversions between thermodynamic variables. The current module is header-only: `state.hpp`, `ideal_gas.hpp`, and `constant_density.hpp`.

It does not store fields, include velocity in the thermodynamic state, compute transport coefficients, or perform mesh traversal. Call a model directly on the host or inside your own Kokkos kernel.

```cpp
#include <krona/thermodynamics/ideal_gas.hpp>
#include <krona/thermodynamics/constant_density.hpp>
namespace thermo = krona::thermodynamics;
```

## Available models

| Model | Purpose | Default parameters |
|---|---|---|
| `IdealGas<Real>` | Single calorically perfect gas, with constant heat capacities and acoustic state conversions | `gamma = 1.4`, `gas_constant = 287.05`, `reference_temperature = 1`, `reference_pressure = 1` |
| `ConstantDensity<Real>` | Nonthermal incompressible material | `rho = 1000` |
| `ThermalConstantDensity<Real>` | Incompressible material with constant density and specific heat | `rho = 1000`, `specific_heat = 4200`, `reference_temperature = 273` |

These are the supplied models; there is no general real-gas, mixture, or tabulated EOS implementation in this module. The two constant-density models are material closures, **not acoustic EOS**: they have no sound-speed or `(rho,e)` inversion API.

`Real` defaults to `ComputeReal` and stores the model parameters. Evaluation methods deduce a separate scalar type `S` from their arguments and return results in `S`. Pass both arguments with the same floating-point type. `cv()` and `cp()` return the parameter type `Real`.

## Thermodynamic state

`State<S>` is a value object containing:

| Member | Meaning | SI units |
|---|---|---|
| `pressure` | Static thermodynamic pressure | Pa |
| `temperature` | Absolute temperature | K |
| `density` | Mass density | kg/m³ |
| `internal_energy` | Specific internal energy `e`, excluding kinetic energy | J/kg |
| `enthalpy` | Specific static enthalpy `h = e + p/rho` | J/kg |
| `entropy` | Specific entropy with the EOS's reference convention | J/(kg·K) |
| `sound_speed` | Adiabatic acoustic speed | m/s |

The implementation performs no unit conversion. Use one consistent unit system for inputs and parameters. `State` does not store heat capacities, velocity, specific total energy, or energy per unit volume. Its default-initialized zeros are not a valid ideal-gas state.

## IdealGas

Set parameters, call `validate()` on the host, then reuse the model:

```cpp
thermo::IdealGas<double> gas;
gas.gamma = 1.4;
gas.gas_constant = 287.05;
gas.reference_temperature = 300.0;
gas.reference_pressure = 101325.0;
gas.validate();

const auto q = gas.from_pT(101325.0, 300.0);
const auto restored = gas.from_rhoe(q.density, q.internal_energy);
const auto isentropic = gas.from_ps(90000.0, q.entropy);
const auto from_hs = gas.from_hs(q.enthalpy, q.entropy);
```

The model uses

$$
c_v=\frac{R}{\gamma-1},\qquad c_p=\gamma c_v,\qquad
\rho=\frac{p}{RT},\qquad e=c_vT,\qquad h=c_pT,
$$
$$
s=c_p\ln\frac{T}{T_{\rm ref}}-R\ln\frac{p}{p_{\rm ref}},
\qquad a=\sqrt{\gamma RT}.
$$

The reference parameters set the entropy zero: `s(T_ref, p_ref) = 0`. They do not shift the energy reference; this model always uses `e = cv*T` and `h = cp*T`.

| Method | Inputs and result |
|---|---|
| `validate()` | Host parameter check: all finite, `gamma > 1`, `R > 0`, and both reference values positive. Does not evaluate a field. |
| `cv()`, `cp()` | Constant specific heat capacities. |
| `from_pT(p, T)` | Returns all seven state members from pressure and temperature. |
| `from_rhoe(rho, e)` | Uses `T = e/cv` and `p = rho*R*T`, then evaluates the state. `e` is **specific internal energy**, not `rho*e` or total energy. |
| `from_ps(p, s)` | Inverts the entropy relation at the specified pressure and returns the state. |
| `from_hs(h, s)` | Uses `T = h/cp`, inverts pressure from entropy, and returns the state. |
| `rhoe_derivatives(rho, e)` | Returns pressure/temperature derivatives in `(rho,e)` coordinates; does not evaluate or validate a complete state. |

`RhoEDerivatives<S>` has these members:

| Member | Fixed variable | Ideal-gas value |
|---|---|---|
| `dp_drho_e` | Specific internal energy | `(gamma - 1)*e` |
| `dp_de_rho` | Density | `(gamma - 1)*rho` |
| `dT_drho_e` | Specific internal energy | `0` |
| `dT_de_rho` | Density | `1/cv` |

### Evaluation in a kernel

This example takes existing rank-one `pressure` and `temperature` views, with at least `count` current entries in memory accessible to `Exec`. `gas` is already validated. The application allocates the output explicitly:

```cpp
using Exec = krona::execution::ExecutionSpace;
using Space = typename Exec::memory_space;
Exec exec{};
Kokkos::View<double*, Space> density("density", count);
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

Kokkos::parallel_for("thermodynamics/density", Policy(exec, 0, count),
    KOKKOS_LAMBDA(std::size_t i) {
        const auto q = gas.from_pT(pressure(i), temperature(i));
        density(i) = q.density;
    });
exec.fence("density ready");
```

`from_pT()` returns the complete state, including entropy and its logarithms. If that state is needed by several operations, evaluate it once in the kernel and reuse the local result.

## Constant-density materials

```cpp
thermo::ConstantDensity<double> liquid{1000.0};
liquid.validate();
const double rho = liquid.density(0.0); // Argument is ignored.

thermo::ThermalConstantDensity<double> thermal_liquid{1000.0, 4200.0, 273.0};
thermal_liquid.validate();
const double thermal_rho = thermal_liquid.density(300.0);
const double e = thermal_liquid.internal_energy(300.0);     // 113400 J/kg.
const double h = thermal_liquid.enthalpy(100000.0, 300.0);  // 113500 J/kg.
```

For `ThermalConstantDensity`, with `C = specific_heat`:

$$
\rho=\text{constant},\qquad e=C(T-T_{\rm ref}),\qquad h=e+p/\rho.
$$

`e(T_ref) = 0`, so a negative internal energy below the reference temperature is not itself an error. Entropy, sound speed and temperature inversions are not supplied.

Both models validate finite positive density. The thermal model also requires finite positive specific heat and reference temperature. Their device-callable evaluation methods perform no input-domain checks; callers must supply physically admissible values. In particular, `density(T)` simply returns the parameter regardless of its argument.

## Requirements for a custom EOS

There is no base class, virtual dispatch, or runtime registration. Supply a device-copyable model with `const` evaluation methods, and pass it as the consumer's template argument/value. Different consumers require different capabilities.

### Basic acoustic interface

`AcousticEOS<EOS, S>` checks exactly two expressions:

```cpp
static_assert(thermo::AcousticEOS<thermo::IdealGas<double>, double>);
// eos.from_pT(S pressure, S temperature) -> thermo::State<S>
// eos.from_rhoe(S density, S specific_internal_energy) -> thermo::State<S>
```

The return type must be exactly `State<S>` to satisfy this concept. The concept does not check physical validity, invertibility, device accessibility, or `validate()`. Existing adapters often use structural calls without constraining their templates with this concept.

For the standard interface:

- Return a consistent state: both conversions must describe the same material, with `h = e + p/rho` and an appropriate adiabatic sound speed. Define the energy and entropy references explicitly.
- Implement host `void validate() const` for model parameters: the physical-boundary and Euler wrappers call it.
- Mark methods used inside kernels `KOKKOS_INLINE_FUNCTION`. Supporting multiple evaluation scalar types normally means templating them on `S`.
- Keep any table buffers accessible from the kernel's execution space and alive until completion. The model must not rely on host-only pointers, containers or virtual calls inside device evaluation.
- Define the admissible input region and failure policy. Do not silently substitute another material or return an unrelated state after a failed inversion. Device evaluation cannot propagate a C++ exception to the host; use the project's appropriate device failure/status strategy.

The current compressible Euler/physical-boundary paths assume finite positive pressure, temperature, density and sound speed. An EOS admitting negative pressures is not automatically usable throughout those paths. A different internal-energy zero can be supported, but the selected numerical algorithms must allow it.

### Example: an explicit energy reference

This adapter keeps ideal-gas pressure/temperature relations but shifts both specific energies by a constant. It demonstrates a custom type satisfying the basic interface; it does not add characteristic or derivative capabilities.

```cpp
struct ShiftedIdealGas {
    thermo::IdealGas<double> gas{};
    double energy_shift = -1e6; // J/kg, added to e and h.

    void validate() const
    {
        gas.validate();
        krona::mpi::require(std::isfinite(energy_shift),
                            "ShiftedIdealGas: nonfinite energy reference");
    }

    template<class S>
    KOKKOS_INLINE_FUNCTION thermo::State<S> from_pT(S p, S T) const
    {
        auto q = gas.from_pT(p, T);
        q.internal_energy += static_cast<S>(energy_shift);
        q.enthalpy += static_cast<S>(energy_shift);
        return q;
    }

    template<class S>
    KOKKOS_INLINE_FUNCTION thermo::State<S> from_rhoe(S rho, S e) const
    {
        const auto q = gas.from_rhoe(rho, e - static_cast<S>(energy_shift));
        return from_pT(q.pressure, q.temperature);
    }
};

static_assert(thermo::AcousticEOS<ShiftedIdealGas, double>);
```

For conservative variables, always remove kinetic energy before calling `from_rhoe()`:

$$
e=\frac{\rho E}{\rho}-\frac{|u|^2}{2}.
$$

The conservative energy must use the same reference as the EOS. Do not assume every EOS requires `e > 0`: that is a restriction of the supplied unshifted `IdealGas`, not part of `AcousticEOS`.

### Capabilities required by consumers

| Consumer / capability | Requirements |
|---|---|
| Direct acoustic-state evaluation | `from_pT()` and/or `from_rhoe()` for the operation being performed. |
| `boundary::physical::Compressible<EOS>` | `validate()`, `from_pT()` and `from_rhoe()`. The default is `NoCharacteristics`; pressure-outlet and total-condition-inlet assignment require a compatible characteristic policy. |
| Custom boundary characteristic policy | Host `validate(const EOS&)`; device-callable `inlet(eos, context, inside, TotalConditionsInlet<S>)` and `outlet(eos, context, inside, pressure)`, returning `Primitive<S,true>`. Implement the physically appropriate closure for that EOS. |
| Supplied `PerfectGasCharacteristics` | Compatible `IdealGas<Real>`. Uses `gamma`, `gas_constant`, `from_pT`, `from_ps`, and `from_hs`; validation accepts the concrete ideal-gas type. Adding similarly named members does not make an arbitrary EOS compatible. |
| `boundary::physical::Incompressible<Material, Thermal>` | Host `validate()` and device-callable `density(S)`. Thermal mode passes temperature; nonthermal mode passes zero. The wrapper does not require the acoustic API or the thermal material's energy helpers. |
| `numerics::riemann` state adapters | Generic `from_pT()` uses the EOS's `(p,T)` conversion; generic `from_conservative()` uses `(rho,e)`. They consume density, pressure, specific internal energy and sound speed from the result. |
| Supplied Riemann `Roe` | A calorically perfect-gas state satisfying its specified `gamma` and unshifted energy relation. It checks compatibility; passing a generic EOS state does not generalize this solver. |
| `transport::euler::Thermodynamics<EOS, Domain>` | Host `validate()` and `(p,T)` evaluation. Provide a domain predicate if positive finite `p,T` alone do not characterize the EOS domain; it runs before EOS evaluation. |
| Heat capacities / transport properties | `cp()` is not part of `AcousticEOS`. Pass the required heat-capacity law/value to the transport-property model explicitly. |
| Derivative-based algorithms | Add `rhoe_derivatives()` only when needed. The provided derivative struct is available, but this capability is not required by `AcousticEOS`. |

For a custom characteristic policy, `context` is `boundary::FaceContext<S>`, and `inside` and both results are `boundary::physical::Primitive<S, true>`. The EOS is passed as `const EOS&`. Additional EOS inversions are required only if that policy's implementation uses them.

`IdealGas` has specialized Riemann adapters that avoid entropy/logarithm evaluation. They return invalid-state markers for rejected inputs rather than calling the full EOS's aborting conversion. Generic adapters retain the custom EOS's own domain/error behaviour. The `ShiftedIdealGas` example takes the generic path and is incompatible with the supplied `Roe` energy relation.

When adding an EOS, verify conversion round trips, `h = e + p/rho`, sound speed, and any supplied derivatives over its intended domain. Check that every selected consumer supports its energy reference and other physical assumptions; passing `AcousticEOS` establishes only the two C++ signatures.

## Validation, ownership and execution

The supplied models are aggregates: construction does not call `validate()`. Validate parameters before use and after changing them. `validate()` is local host work using `mpi::require`; it launches no kernels and performs no collective communication.

`IdealGas::from_pT()` rejects nonfinite/nonpositive `p,T` through `Kokkos::abort`; `from_rhoe()` first rejects nonfinite/nonpositive `rho,e`. The entropy inversions eventually call `from_pT()`, but do not separately validate every intermediate expression. `rhoe_derivatives()`, `cv()` and `cp()` do not check their inputs/parameters. There is no complete postcheck of all derived state members: keep parameters and states within the representable range of the evaluation scalar type, not merely finite at input.

The built-in models own only scalar parameters; `State` owns its seven scalar results. Evaluation allocates no arrays and retains no references to the input arguments. Model copies captured by `KOKKOS_LAMBDA` copy the parameter values. A custom tabulated model must define its own buffer ownership and lifetime.

All thermodynamic evaluation methods are inline calculations in the calling host/device context. They do not enqueue work, synchronize an execution space, update fields, or exchange ghosts. An enclosing `parallel_for` may be asynchronous; synchronize before host access, destruction/reallocation, or consumers on unrelated execution instances. On the same execution instance, dependent kernels can run in submission order without an intermediate fence. No EOS method performs an MPI barrier or halo exchange.
