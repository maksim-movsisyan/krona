# Transport properties and isotropic diffusion

This document covers `transport/properties` and `transport/diffusion/isotropic.hpp`. The first evaluates molecular transport coefficients; the second supplies coefficient providers and the local scalar law relating normal gradient to diffusive flux. Neither assembles a mesh operator.

```cpp
#include <krona/transport/properties/properties.hpp>
#include <krona/transport/diffusion/isotropic.hpp>
namespace props = krona::transport::properties;
namespace tr = krona::transport;
```

## Evaluate transport properties

`TransportProperties<Viscosity, Conductivity>` stores two policies by value. It evaluates viscosity once, passes that value to the conductivity policy, and returns `Coefficients<Real>`:

| Member | Meaning | SI units |
|---|---|---|
| `dynamic_viscosity` | Dynamic viscosity `mu`, not kinematic viscosity `mu/rho` | Pa·s |
| `thermal_conductivity` | Thermal conductivity `lambda` | W/(m·K) |

`Coefficients::valid()` requires both values to be finite and nonnegative; zero is allowed. Evaluation does not call `valid()` automatically. The module performs no unit conversion and imposes no specific state class: each policy reads only the state members it needs.

```cpp
struct TemperatureState { double temperature; };

const auto model = props::TransportProperties{
    props::SutherlandViscosity{1.716e-5, 273.15, 110.4},
    props::ConstantPrandtl{0.72, props::ConstantHeatCapacity{1005.0}}};
model.validate();

const auto coefficients = model.evaluate<double>(TemperatureState{300.0});
krona::mpi::require(coefficients.valid(), "Invalid transport properties");
// Read coefficients.dynamic_viscosity and coefficients.thermal_conductivity.
```

Call `validate()` on the host before using configured parameters. Aggregate construction does not validate them. Revalidate after parameter changes.

`model.evaluate<Real>(state)` selects floating-point arithmetic/output type explicitly. `model(state)` always uses `ComputeReal`, regardless of state or parameter storage types. The built-in policies also support `evaluate<Real>()` directly. Their parameter type defaults to `ComputeReal`; constructor deduction can select another type from the arguments.

For a ready rank-one `temperature` view with `count` entries accessible to `Exec`, evaluate a field as follows:

```cpp
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Space = typename Exec::memory_space;
Exec exec{};
Kokkos::View<props::Coefficients<Real>*, Space> values("properties", count);
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
Kokkos::parallel_for("properties/evaluate", Policy(exec, 0, count),
    KOKKOS_LAMBDA(std::size_t i) {
        values(i) = model.template evaluate<Real>(
            TemperatureState{static_cast<double>(temperature(i))});
    });
exec.fence("properties ready");
const auto observed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
for (std::size_t i = 0; i < count; ++i)
    krona::mpi::require(observed(i).valid(), "Invalid transport properties");
```

## Supplied property models

| Policy | Inputs / formula | Parameter validation |
|---|---|---|
| `ConstantViscosity{mu}` | Returns constant dynamic viscosity; ignores the state | Finite `mu >= 0` |
| `SutherlandViscosity{mu_ref, T_ref, S}` | Reads `state.temperature`; formula below | Finite `mu_ref > 0`, `T_ref > 0`, `S >= 0` |
| `ConstantConductivity{lambda}` | Returns constant conductivity; ignores state and supplied viscosity | Finite `lambda >= 0` |
| `ConstantHeatCapacity{cp}` | Specific heat at constant pressure, in J/(kg·K); ignores the state | Finite `cp > 0` |
| `ConstantPrandtl{Pr, heat_capacity}` | `lambda = mu * heat_capacity.evaluate<Real>(state) / Pr` | Finite `Pr > 0`; calls heat-capacity validation if available |

Sutherland's law is

$$
\mu(T)=\mu_{\rm ref}\left(\frac{T}{T_{\rm ref}}\right)^{3/2}
\frac{T_{\rm ref}+S}{T+S}.
$$

The implementation uses an equivalent form that avoids explicitly forming `T^(3/2)`. Temperatures are absolute. Nonfinite/nonpositive evaluation temperatures return NaN; there is no temperature clipping or automatic check of the empirical fit's applicable range.

`ConstantPrandtl` evaluates heat capacity once. A nonfinite/nonpositive `cp` returns NaN conductivity, even when viscosity is zero. Constant policies do not inspect unused state fields: for example, constant viscosity/conductivity require no temperature member.

### Using an EOS

A thermodynamic state can be used directly when it provides the needed members. An EOS is not stored or queried automatically. For a validated `IdealGas` object named `gas`:

```cpp
const auto gas_model = props::TransportProperties{
    props::SutherlandViscosity{1.716e-5, 273.15, 110.4},
    props::ConstantPrandtl{0.72, props::ConstantHeatCapacity{gas.cp()}}};
gas_model.validate();
const auto thermo_state = gas.from_pT(101325.0, 300.0);
const auto gas_coefficients = gas_model.evaluate<double>(thermo_state);
```

`ConstantHeatCapacity{gas.cp()}` copies the current value; changing `gas` later does not update the policy. For temperature-dependent heat capacity, supply a state-dependent policy.

## Add a custom property model

No inheritance or registration is required. Supply a value/device-copyable object with these `const` methods:

| Policy role | Device-callable interface |
|---|---|
| Viscosity | `template<class Real, class State> Real evaluate(const State&) const` |
| Conductivity | `template<class Real, class State> Real evaluate(const State&, Real viscosity) const` |
| Heat capacity for `ConstantPrandtl` | Same signature as viscosity; returns `cp` |

Use `KOKKOS_INLINE_FUNCTION` on device evaluation. Host `void validate() const` is optional; `TransportProperties::validate()` calls it on each policy if present. Define the state members and admissible domain your model requires. Built-in runtime domain failures propagate NaNs for the caller to detect through `Coefficients::valid()`.

For example, a custom heat-capacity law can be combined with the existing models:

```cpp
struct LinearHeatCapacity {
    template<class Real, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state) const
    {
        const Real T = static_cast<Real>(state.temperature);
        if (!Kokkos::isfinite(T) || !(T > Real{0}))
            return Kokkos::quiet_NaN_v<Real>;
        return Real{1000} + static_cast<Real>(0.1)*(T - Real{300});
    }
};

const auto custom_model = props::TransportProperties{
    props::SutherlandViscosity{1.716e-5, 273.15, 110.4},
    props::ConstantPrandtl{0.72, LinearHeatCapacity{}}};
custom_model.validate(); // LinearHeatCapacity has no configurable parameters.
```

## Isotropic diffusion law

`IsotropicDiffusion<Coefficient>` uses a callable `coefficient(context)` to implement

$$
q_n=-\Gamma\frac{\partial\phi}{\partial n}.
$$

The normal and flux follow the context's orientation. On an exterior boundary use the outward normal: positive `q_n` means transport out of the domain. For temperature with `Gamma = lambda`, heat entering the domain has negative outward flux. These are flux densities, with **no face-area multiplier**.

```cpp
struct EmptyContext {};
const EmptyContext context;
const auto law = tr::IsotropicDiffusion{tr::UniformCoefficient{2.0}};

const double flux = law.outward_flux(context, 3.0);      // -2 * 3 = -6.
const double gradient = law.normal_gradient(context, -6.0); // -(-6) / 2 = 3.
```

`outward_flux(context, gradient)` requires finite `Gamma >= 0`. `normal_gradient(context, flux)` divides by the coefficient and requires finite `Gamma > 0`. Thus zero conductivity can produce zero flux but cannot be inverted to determine a gradient. These coefficient checks use `KOKKOS_ASSERT`, which may be disabled; the law has no host `validate()` or runtime status object. Validate coefficient data through the application's error/status handling before use. It does not itself validate the supplied flux or gradient.

Diffusion arithmetic uses the common type of the coefficient result and the supplied flux/gradient argument.

## Coefficient providers

| Provider | Context / storage contract |
|---|---|
| `UniformCoefficient{value}` | Returns one scalar for any context; default value is `1`. No automatic validation. |
| `FaceCoefficient{view}` | Reads `view(context.face)`. Supply valid local face indices and a rank-one coefficient view. |
| `OwnerCellCoefficient{view}` | Reads `view(context.owner)` for an exterior face. If the context has `neighbour`, asserts that it is negative; this provider is not two-sided interior interpolation. |
| `InterpolatedCellCoefficient{view}` | Default harmonic interpolation on interior faces; owner extrapolation on exterior faces. Interior contexts need `owner`, `neighbour >= 0`, `owner_weight`, and `distance`. Values include current ghost coefficients. |
| `InterpolatedCellCoefficient{view, ArithmeticCoefficient{}}` | Same interface, with arithmetic interpolation. |

View providers require rank-one Kokkos views and support strides. The application supplies accessible buffers with sufficient extents and valid indices; the providers do not validate array lengths or exchange ghosts. `InterpolatedCellCoefficient` uses the common type of the view scalar and `context.distance`. It takes the two-sided branch only when the context supplies both `neighbour` and `owner_weight` and the neighbour is nonnegative; otherwise it returns the owner value.

For an inner face, let `w = owner_weight`, the weight used in `phi_f = w*phi_owner + (1-w)*phi_neighbour`. With positive owner/neighbor distances to the face plane, `w = d_neighbour/(d_owner + d_neighbour)`:

| Interpolation | Formula |
|---|---|
| Arithmetic | `Gamma_f = w*Gamma_owner + (1-w)*Gamma_neighbour` |
| Harmonic, series resistance | `Gamma_f = 1 / ((1-w)/Gamma_owner + w/Gamma_neighbour)` |

The reversed distance weights in the harmonic denominator are intentional: it adds owner- and neighbour-side resistances. Supply finite `w` in `[0,1]`; these helpers do not check it. At `w = 0.5`, harmonic interpolation gives `2*Gamma_owner*Gamma_neighbour/(Gamma_owner+Gamma_neighbour)`.

For the interior branch, `InterpolatedCellCoefficient` requires **both cell coefficients strictly positive and finite**, even with arithmetic interpolation. Invalid values are propagated rather than repaired. Exterior owner values are returned directly. These provider contracts are stricter than `Coefficients::valid()`, which permits zero molecular coefficients.

## Evaluate conductivity directly at a face

A provider can evaluate a property model instead of reading a precomputed conductivity array. This example reads an already prepared **face-temperature** view; temperature interpolation/boundary filling remains the application's responsibility.

```cpp
template<class TemperatureView, class Properties>
struct ConductivityAtFace {
    TemperatureView temperature;
    Properties properties;

    template<class Context>
    KOKKOS_INLINE_FUNCTION krona::ComputeReal operator()(const Context& c) const
    {
        using R = krona::ComputeReal;
        struct State { R temperature; };
        const auto coefficients = properties.template evaluate<R>(
            State{static_cast<R>(temperature(static_cast<std::size_t>(c.face)))});
        return coefficients.valid() ? coefficients.thermal_conductivity
                                    : Kokkos::quiet_NaN_v<R>;
    }
};
```

For a validated `model` and a ready rank-one `face_temperature` view:

```cpp
const auto provider = ConductivityAtFace{face_temperature, model};
const auto heat_law = tr::IsotropicDiffusion{provider};
// Inside a face kernel:
// auto q_n = heat_law.outward_flux(face_context, dT_dn);
```

This evaluates the model at the face temperature. Interpolating precomputed cell conductivities with `InterpolatedCellCoefficient` is a different operation and can produce a different result for nonlinear models. Select the intended operation explicitly. The diffusion law is scalar; it does not implement the full viscous stress tensor or construct a Laplacian discretization.

## Ownership and execution

Built-in property models own scalar parameters; `TransportProperties` owns its policies by value and `Coefficients` owns its two results. No EOS, field arrays or cached evaluations are created. View-based coefficient providers store view descriptors by value: managed views retain their allocation, while unmanaged views borrow storage. Custom policies define the ownership of any additional buffers.

`validate()` performs local host checks through `mpi::require`, with no kernel launch or collective communication. Evaluation, `valid()`, interpolation and diffusion-law methods calculate inline in their caller's host/device context: no allocation, independent launch, fence, MPI exchange or implicit field update.

To process fields, capture a validated model/provider in a Kokkos kernel, evaluate per cell or face, and write coefficients/fluxes to caller-owned outputs or consume them immediately. That enclosing launch may be asynchronous. Order dependent work on the same execution instance or establish completion before unrelated instances, host access, external MPI, or buffer destruction/reallocation. Keep all required ghost values current. The models do not synchronize or communicate on the caller's behalf.
