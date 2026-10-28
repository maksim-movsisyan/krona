# Physical boundary conditions

The physical boundary module supplies face-centroid primitive values for flow
boundaries on KRONA's grouped, stationary unstructured mesh. It supports
compressible flow and incompressible flow with or without temperature. The
same manager also exposes typed traversal for application-owned fluxes,
gradients, matrix coefficients and Jacobian contributions.

This document describes the current implementation. Setup/application examples
assume MPI and Kokkos are initialized and their lifetimes enclose all objects.
Independent examples reuse local names. They use the following aliases:

```cpp
#include <krona/boundary/physical/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/transport/isotropic_diffusion.hpp>
#include <cmath>
#include <type_traits>

namespace bc = krona::boundary;
namespace ph = krona::boundary::physical;
namespace th = krona::thermodynamics;
namespace tr = krona::transport;
using Exec = krona::execution::ExecutionSpace;
using Memory = typename Exec::memory_space;
using Real = krona::ComputeReal;
using Values = Kokkos::View<Real*, Memory>;
using Grad = Kokkos::View<Real*[3], Memory>;
using GasModel = ph::Compressible<th::IdealGas<Real>, ph::PerfectGasCharacteristics>;
```

Quick paths: [setup](#3-initial-setup-and-field-filling),
[models](#4-flow-models-and-thermodynamics),
[condition reference](#5-supported-boundary-conditions),
[traversal](#9-traversal-and-evaluation-apis),
[gradients](#11-boundary-gradients-of-primitive-variables),
[execution contracts](#14-validation-asynchronous-execution-and-lifetimes).

`Grad` is an example descriptor type. When binding existing gradient storage,
use its matching layout/view type or generalize the helper templates; no
reinterpretation of another layout through a raw pointer is intended.

## 1. Architecture and ownership

| Object | Responsibility |
|---|---|
| `PatchCatalog` | Own host patch metadata and contiguous local face intervals, including the unlabelled interval. |
| `GeometryView<Space>` | Borrow the existing cell/face geometry and face owners. |
| `FaceContext<Real>` | Describe one face: indices, position, outward unit normal, area, normal distance, tangential displacement and time. |
| A condition such as `IsothermalWall<Real>` | Store physical boundary parameters as a concrete value type. |
| `Compressible<EOS, Characteristics>` / `Incompressible<Material, Thermal>` | Supply the capabilities and explicit physical dependencies needed by the closure. |
| `PrimitiveViews` | Bind existing pressure, velocity and optional temperature components. |
| `physical::evaluate(...)` | Compute a local `Evaluation<State>` for one concrete condition. |
| `Manager<Model, Exec, Real, Custom...>` | Validate setup, bind conditions and submit typed face kernels on one execution instance. |
| Reader and sink functors | Select the interior state and consume results using application-owned storage. |

The manager owns its catalog, condition bindings, model value and execution
instance. It borrows compute geometry. It owns no solution fields, conservative
arrays, gradient arrays, transport-property arrays or persistent host mirrors.
Its copy and move operations are deleted.

Runtime selection happens on the host through a variant per patch. Each launch
receives a **concrete condition type**. There is no variant visitation, string
lookup or virtual dispatch inside a face kernel. A normal traversal submits one
kernel per nonempty local patch interval, plus the unlabelled interval if present.

Thermal wall conditions reuse the scalar module's `Relation`, `FixedValue`,
`NormalGradient` and `FixedFlux` conversion. The physical manager and scalar
manager share the boundary geometry/catalog infrastructure. Independent passive
scalars retain their own scalar conditions; the physical temperature channel is
configured as part of the physical condition.

## 2. Mesh and field contract

Boundary faces occupy `[n_inner_faces, n_faces)`. Each label occupies one
contiguous interval; unlabelled boundary faces also form one interval. The
compute mesh must attach `MeshData::FaceCells | MeshData::Geometry` and agree
with the host mesh counts checked by the manager.

For a boundary face `f`:

```text
boundary_index = f - n_inner_faces
extended_index = n_cells + boundary_index
required_field_size = n_cells + n_faces - n_inner_faces
```

`n_cells` includes owned cells and real MPI ghost cells. Boundary entries store
**values at physical face centroids**. They are not boundary ghost-cell values.
`apply_async` writes only these boundary entries; owned values, MPI ghosts and
any trailing storage remain untouched.

| Context member | Meaning |
|---|---|
| `face` | Absolute local face index. |
| `owner` | Owning real cell; boundary owners must be locally owned. |
| `boundary_index` | Index in a compact boundary-face array. |
| `extended_index` | Index in an extended cell-plus-boundary field. |
| `x`, `y`, `z` | Face centroid. |
| `nx`, `ny`, `nz` | Unit normal directed out of the owner cell. |
| `distance` | Positive normal projection of the owner-to-face displacement. |
| `tx`, `ty`, `tz` | Remaining tangential displacement. |
| `area` | Face area. |
| `time` | Finite caller-supplied time, forwarded to custom callables. |

With owner centre `x_P`, face centre `x_f` and unit normal `n`, the geometry is

\[
h=(x_f-x_P)\cdot n>0,\qquad t=x_f-x_P-hn.
\]

`primitive_views(p, velocity[, temperature])` accepts KRONA `FieldView` objects
with one pressure component, three velocity components and optionally one
temperature component. Pass `.view()` from storage objects. LayoutLeft and
LayoutRight are supported; component strides are preserved.

For split arrays or selected components, use raw rank-one Kokkos views:

```cpp
auto primitive = ph::primitive_components(p, ux, uy, uz, temperature);
// Without temperature:
auto unheated = ph::primitive_components(p, ux, uy, uz);
```

The descriptor's `read<Real>(i)` returns a small local primitive state;
`write(i, state)` writes components with explicit conversion to their storage
types. Binding these views allocates no field storage. Writable operations
require writable floating-point components accessible from `Exec`.

## 3. Initial setup and field filling

The following setup assumes `host_mesh`, `mesh_view` and an explicit `exec`
already exist. The configuration must match the mesh patch IDs and names.

```cpp
GasModel model{th::IdealGas<Real>{}, ph::PerfectGasCharacteristics{}};
const auto config = ph::read_config("boundary.toml");
ph::Manager<GasModel, Exec> boundary{
    host_mesh, mesh_view, model, config, exec
};

const auto count = boundary.required_field_size();
krona::fields::FieldStorage<Real, 1, Exec> p{
    "pressure", count, krona::fields::FieldLocation::CellExtnd, exec};
krona::fields::FieldStorage<Real, 3, Exec> velocity{
    "velocity", count, krona::fields::FieldLocation::CellExtnd, exec};
krona::fields::FieldStorage<Real, 1, Exec> temperature{
    "temperature", count, krona::fields::FieldLocation::CellExtnd, exec};

// Illustrative uniform initialization. In a solver, owned primitives are
// updated from the current solution before applying the boundary conditions.
p.fill(exec, Real{100000});
velocity.fill(exec, Real{0});
temperature.fill(exec, Real{300});

const auto primitive = ph::primitive_views(
    p.view(), velocity.view(), temperature.view());
const auto conduction = tr::IsotropicDiffusion{
    tr::UniformCoefficient<Real>{Real{2}}};

boundary.apply_async(primitive, conduction);
// Further work submitted to this exec can consume the face values.
```

The model and manager are setup objects; construct them outside the time loop.
The manager does not update primitives from conservatives or perform a halo
exchange. The application supplies current, ready input data.

Alternatively, construct without a configuration and bind conditions in C++:

```cpp
ph::Manager<GasModel, Exec> boundary{host_mesh, mesh_view, model, exec};
boundary.set(0, ph::IsothermalWall<Real>{Real{350}, {}});
boundary.set(1, ph::TotalConditionsInlet<Real>{
    Real{120000}, Real{350}, {-1, 0, 0}, false});
boundary.set(2, ph::PressureOutlet<Real>{Real{100000}, {}});
boundary.set_unlabelled(ph::AdiabaticWall<Real>{});
```

This example assumes labels 0, 1 and 2 are the complete catalog. Every labelled
patch needs a binding, including patches empty on the current rank. A global
unlabelled boundary requires `set_unlabelled` even on ranks with no local
unlabelled faces. Local empty intervals launch no kernels.

## 4. Flow models and thermodynamics

The model's static flags select capabilities at compile time:
`compressible`, `thermal` and `has_characteristics`. `thermal` determines
whether `Primitive<Real, thermal>` actually contains a temperature member.

| Model | Temperature | Characteristic inlet/outlet | Purpose |
|---|---|---|---|
| `Compressible<IdealGas<Real>, PerfectGasCharacteristics>` | Present | Supported | Calorically perfect gas with the supplied acoustic closure. |
| `Compressible<EOS, NoCharacteristics>` | Present | Unavailable | EOS conversions and conditions that do not require the characteristic policy. |
| `Incompressible<ConstantDensity<Real>>` | Absent | Unavailable | Constant-density flow without heat transport. |
| `Incompressible<ThermalConstantDensity<Real>, true>` | Present | Unavailable | Constant-density flow with heat transport. |
| `Compressible<CustomEOS, CustomCharacteristics>` | Present | Supplied by the custom policy | Extension to another acoustic thermodynamic model. |

The built-in `Compressible` always has temperature. `Incompressible` separates
the thermal capability from its material type; use matching field bindings.
The current condition implementations target these capability combinations.

### Perfect gas

`IdealGas<Real>` stores `gamma`, `gas_constant`, `reference_temperature` and
`reference_pressure`. Defaults are 1.4, 287.05, 1 and 1 respectively.

```cpp
th::IdealGas<Real> gas;
gas.gamma = static_cast<Real>(1.4);
gas.gas_constant = static_cast<Real>(287.05);
gas.validate();

const auto a = gas.from_pT(Real{100000}, Real{300});
const auto b = gas.from_rhoe(a.density, a.internal_energy);
const auto c = gas.from_ps(a.pressure, a.entropy);
const auto d = gas.from_hs(a.enthalpy, a.entropy);
const auto derivatives = gas.rhoe_derivatives(a.density, a.internal_energy);
```

`thermodynamics::State<Real>` contains pressure, temperature, density, specific
internal energy, specific enthalpy, specific entropy and sound speed. The gas
uses `e=cv*T`, `h=cp*T` and `c=sqrt(gamma*R*T)`. The reference states specify
the entropy reference. `rhoe_derivatives` returns `dp_drho_e`, `dp_de_rho`,
`dT_drho_e` and `dT_de_rho`.

The compressible wrapper also provides device-callable conversions:

```cpp
GasModel model{gas, ph::PerfectGasCharacteristics{}};
ph::Primitive<Real, true> primitive{Real{100000}, {10, 20, 30}, Real{300}};
const auto conservative = model.conservative(primitive);
const auto restored = model.primitive(conservative);
```

`Conservative` contains density, momentum per volume and `total_energy` per
volume: `rho * (e + dot(u,u)/2)`. Conversion to primitives removes kinetic
energy before calling `EOS::from_rhoe`. No conservative array is maintained by
the boundary manager. Calling EOS/conversion routines directly requires valid
parameters; call the host model validation during setup, as the manager does.

### Incompressible flow without temperature

```cpp
using LiquidModel = ph::Incompressible<th::ConstantDensity<Real>>;
LiquidModel model{th::ConstantDensity<Real>{Real{1000}}};
ph::Manager<LiquidModel, Exec> boundary{
    host_mesh, mesh_view, model, liquid_config, exec};
auto primitive = ph::primitive_views(p.view(), velocity.view());
boundary.apply_async(primitive);
```

`Primitive<Real, false>` has only pressure and velocity. No dummy temperature
allocation is needed. This material has no acoustic sound speed or artificial
acoustic EOS. `model.density(state)` returns the material density. Pressure can
be gauge pressure, including negative values. An `AdiabaticWall` in this model
provides the mechanical no-slip condition without thermal work.

### Incompressible flow with heat transport

```cpp
using HeatedLiquid = ph::Incompressible<th::ThermalConstantDensity<Real>, true>;
HeatedLiquid model{th::ThermalConstantDensity<Real>{
    Real{1000}, Real{4200}, Real{273}}};
ph::Manager<HeatedLiquid, Exec> boundary{
    host_mesh, mesh_view, model, heated_liquid_config, exec};
auto primitive = ph::primitive_views(
    p.view(), velocity.view(), temperature.view());
boundary.apply_async(primitive, conduction);
```

`ThermalConstantDensity` stores `rho`, `specific_heat` and
`reference_temperature`, with defaults 1000, 4200 and 273. It provides
`density(T)`, `internal_energy(T)=specific_heat*(T-T_ref)` and
`enthalpy(p,T)=internal_energy(T)+p/rho`. The manager imposes thermal boundary
values; the energy equation and its variable choice belong to the application.

## 5. Supported boundary conditions

The parameter names below are the canonical TOML keys. Every
`[[boundary_condition]]` entry additionally requires `patch_id`, `name` and
`type`. Vectors contain three finite components. Wall velocities default to
`[0,0,0]`. Temperatures are absolute; gas pressure must be positive. Units must
be consistent with the chosen material/EOS constants.

| C++ condition / TOML `type` | Required condition data | Optional data and defaults | Compatible models |
|---|---|---|---|
| `Symmetry` / `SYMMETRY` | None | None | All supplied models. |
| `SlipWall` / `SLIP_WALL` | None | `velocity=[0,0,0]` | All supplied models. |
| `AdiabaticWall` / `ADIABATIC_WALL` | None | `velocity=[0,0,0]` | All; thermal part active only when temperature exists. |
| `IsothermalWall` / `ISOTHERMAL_WALL` | `temperature > 0` | `velocity=[0,0,0]` | Models with temperature. |
| `HeatFluxWall` / `HEAT_FLUX_WALL` | Finite `flux`; explicit flux law when evaluating | `velocity=[0,0,0]` | Models with temperature. |
| `PressureOutlet` / `PRESSURE_OUTLET` | Finite `pressure`; positive for gas | `backflow` table; default `REJECT` | Compressible with characteristics; incompressible with/without temperature. |
| `TotalConditionsInlet` / `TOTAL_CONDITIONS_INLET` | Positive `total_pressure`, `total_temperature`; nonzero `direction` | `allow_choking=false` | Compressible with characteristics. |
| `VelocityInlet` / `VELOCITY_INLET` | `velocity`; positive `temperature` when the model is thermal | Temperature can be omitted for an unheated model. | Incompressible only in the current implementation. |
| `SupersonicInlet` / `SUPERSONIC_INLET` | Positive `pressure`, `temperature`; either `velocity` or positive `mach` plus `direction` | No independent remaining flow data. | Compressible; no characteristic policy required. |
| `SupersonicOutlet` / `SUPERSONIC_OUTLET` | None | None | Compressible; no characteristic policy required. |

Directions are normalized once by `set`/`configure`. Inlet and backflow
directions must point inward on each affected face. That orientation depends
on geometry and is checked during evaluation. A single direction may therefore
be unsuitable for a patch whose normals vary substantially.

### Wall and symmetry closures

For all walls, pressure uses `p_f = p_P + delta_p`, corresponding to the
implemented zero normal derivative approximation. This is a wall pressure
closure, not a general derivation of pressure gradients for every moving or
curved flow.

`Symmetry` removes the normal component of the reconstructed velocity.
`SlipWall` instead prescribes `u_f dot n = u_wall dot n`; only the normal wall
velocity affects its mechanical closure. Both retain reconstructed tangential
velocity and use `T_f = T_P + delta_T` when thermal.

The three no-slip wall types prescribe the complete vector
`u_f = u_wall`. `IsothermalWall` fixes the physical face temperature.
`AdiabaticWall` converts zero heat flux through the supplied law when that law
has the required interface; without a law it uses `dT/dn=0` as the isotropic
conduction shortcut. `HeatFluxWall` always requires an explicit conversion law.

### Supersonic conditions

`SupersonicInlet` requires `u_f dot n < -c_f`. An input Mach number refers to
the full speed, so `mach > 1` alone does not guarantee supersonic flow normal
to an oblique face. In C++, `mach == 0` selects the supplied velocity;
`mach > 0` selects `mach*c*direction` and ignores the velocity member. The TOML
parser accepts exactly one of those input forms.

`SupersonicOutlet` extrapolates the complete interior state and requires
`u_i dot n >= c_i`. A sonic outward state is accepted. An incoming
characteristic is a runtime error for this explicit condition type.

### Pressure outlet and reverse flow

For gas, `PressureOutlet` first classifies the interior normal velocity:

1. `u_n >= c`: extrapolate the complete interior state; ignore requested pressure.
2. `u_n < 0`: use the explicit backflow policy.
3. Otherwise: call the characteristic outlet closure. If its result reverses
   the flow, use the backflow policy as well.

For incompressible outward flow, set pressure and extrapolate velocity and
temperature with their explicit corrections. Reverse flow uses the following
additional data:

| C++ `BackflowMode` / TOML `backflow.type` | Required data | Effect |
|---|---|---|
| `Reject` / `REJECT` | None; this is the default. | Abort on reverse flow. It does not clamp velocity or silently skip the face. |
| `TotalConditions` / `TOTAL_CONDITIONS_INLET` | `total_pressure`, `total_temperature`, inward `direction`; optional `allow_choking=false` | Gas only: switch to the reservoir inlet closure. The outlet static pressure is not simultaneously imposed. |
| `DirectionTemperature` / `DIRECTION_TEMPERATURE` | Inward `direction`; `temperature` when thermal | Incompressible only: retain the interior negative normal velocity, use the prescribed direction and entering temperature; retain prescribed outlet pressure. |

For the incompressible backflow direction `d`, the implementation uses

\[
u_b=d\,\frac{u_{n,i}}{d\cdot n}.
\]

It supplies no independent backflow mass-flow specification. A gas backflow
state that is already supersonic inward is rejected by the total-condition
closure; full incoming state data would be needed instead.

## 6. Characteristic policy and its assumptions

`PerfectGasCharacteristics` supplies `inlet` and `outlet` for the built-in
calorically perfect `IdealGas`. It uses local one-dimensional normal
compatibility. Denote the inside state by `i`, the boundary state by `b`, and
the outward normal velocity by `u_n`. The adopted outgoing invariant is

\[
J_+=u_n+\frac{2c}{\gamma-1}.
\]

On the subsonic outlet branch, the policy prescribes pressure and preserves
interior entropy, tangential velocity and `J_+`. Consequently,

\[
T_b=T_i(p_b/p_i)^{(\gamma-1)/\gamma},\qquad
u_{n,b}=u_{n,i}+\frac{2(c_i-c_b)}{\gamma-1}.
\]

An unreachable low pressure on this subsonic branch is replaced by the sonic
state `u_n=c`. The resulting pressure can exceed the requested pressure.

For the total-condition inlet, direction `d` and total data `p0,T0` determine
reservoir entropy and total enthalpy. On the regular subsonic branch the speed
`V` is obtained by combining

\[
u_b=Vd,\quad c_pT_b+V^2/2=c_pT_0,\quad
p_b=p_0(T_b/T_0)^{\gamma/(\gamma-1)},\quad
u_{n,b}+2c_b/(\gamma-1)=J_{+,i}.
\]

This reduces to the quadratic solved in the implementation. Total data do not
independently prescribe speed or mass flow. If the selected inward branch
requires a critical normal state, `allow_choking=false` aborts;
`allow_choking=true` selects `u_n=-c` using reservoir data instead of the usual
interior-invariant compatibility. Data admitting no inward solution are rejected.

This policy does not guarantee a nonreflecting exit. Fixed instantaneous
pressure can reflect acoustic disturbances even when the other state variables
are reconstructed through characteristics. It also does not solve arbitrary
shock connections or include multidimensional characteristic source terms.
Viscous and thermal flux discretizations remain separate responsibilities.

The underlying perfect-gas total/static relations are summarized in
[NASA's isentropic flow equations](https://www.grc.nasa.gov/WWW/BGH/isentrop.html).
The [Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html)
describes simple-wave invariants and their distinction from shock relations.

## 7. TOML and parsing

Both entry points are local host operations with the same error convention as
the scalar module:

```cpp
const auto from_file = ph::read_config("boundary.toml");
const auto from_text = ph::parse_config(existing_text);
// Either Configuration can be passed to the constructor or configure().
```

For example, a gas configuration with three labels is:

```toml
[[boundary_condition]]
patch_id = 0
name = "heated_wall"
type = "ISOTHERMAL_WALL"
temperature = 350
velocity = [0, 0, 0]

[boundary_condition.scalar.tracer]
type = "FIXED_VALUE"
value = 0.2

[[boundary_condition]]
patch_id = 1
name = "inlet"
type = "TOTAL_CONDITIONS_INLET"
total_pressure = 120000
total_temperature = 350
direction = [-1, 0, 0]
allow_choking = false

[[boundary_condition]]
patch_id = 2
name = "outlet"
type = "PRESSURE_OUTLET"
pressure = 100000

[boundary_condition.backflow]
type = "TOTAL_CONDITIONS_INLET"
total_pressure = 120000
total_temperature = 350
direction = [-1, 0, 0]

[unlabelled]
type = "ADIABATIC_WALL"
```

For a heated incompressible inlet/outlet pair:

```toml
[[boundary_condition]]
patch_id = 0
name = "inlet"
type = "VELOCITY_INLET"
velocity = [-2, 0, 0]
temperature = 300

[[boundary_condition]]
patch_id = 1
name = "outlet"
type = "PRESSURE_OUTLET"
pressure = 0
[boundary_condition.backflow]
type = "DIRECTION_TEMPERATURE"
direction = [-1, 0, 0]
temperature = 290
```

The names, IDs and directions in these examples are illustrative and must match
the actual mesh. A supersonic input entry may instead use
`pressure`, `temperature`, `mach` and `direction`, or the same thermodynamic
data plus `velocity`.

The parser also accepts these generator spellings:

| Canonical type | Accepted alternative | Parameter aliases |
|---|---|---|
| `ADIABATIC_WALL` | `WALL_ADIABATIC` | None. |
| `ISOTHERMAL_WALL` | `WALL_ISOTHERMAL` | `T_wall` for `temperature`. |
| `TOTAL_CONDITIONS_INLET` | `INLET_TOTAL_TP` | `p0` / `T0` for total pressure / temperature. |
| `PRESSURE_OUTLET` | `OUTLET_BACK_PRESSURE` | `p_back` for `pressure`. |
| `SUPERSONIC_INLET` | `INLET_SUPERSONIC` | `p`, `T`, `M` for pressure, temperature, Mach. |
| `SUPERSONIC_OUTLET` | `OUTLET_SUPERSONIC` | None. |

Parameter aliases are accepted with either spelling of the corresponding type.
A backflow total-condition table accepts `p0`/`T0`, but its type name is
`TOTAL_CONDITIONS_INLET`. Supplying both aliases for one parameter is an error.

The parser accepts finite numbers, three-component numeric vectors and booleans
for the corresponding keys. It checks syntax and unexpected condition keys;
model compatibility and positive-parameter checks occur when conditions are
bound to a manager. Configuration values are stored as `double` and converted
to the manager's `Real` during binding, followed by preparation/validation.

`family`, `source_type`, and `global_face_count` are permitted metadata.
Independent `scalar.*` subtables are ignored by the physical parser, except
`scalar.temperature`, which is rejected as a conflict with physical temperature.
Temperature for walls and flow conditions is specified directly in the physical
entry. Assigning another scalar manager to the same temperature slots must also
be avoided by the application.

`configure` requires exactly one entry for every catalog label, matched by ID
and exact name; input order may differ from catalog order. Duplicate IDs fail.
`[unlabelled]` uses the same physical syntax. A scalar-only unlabelled table
does not supply a physical binding. With no labels, the root still needs
`boundary_condition = []`. `FARFIELD` and `USER_DEFINED` are not parsed built-ins.

## 8. Corrections and transport properties

The explicit correction for a scalar primitive is

\[
\delta\phi=t\cdot\nabla\phi_P,\qquad
\phi_f=\phi_P+\delta\phi+h\,g_n.
\]

`bc::NoCorrection` returns zero. `bc::GradientCorrection{gx,gy,gz}` reads
precomputed owner-cell gradient components and forms this scalar increment.
`ph::ZeroVelocityIncrement` returns a zero vector; `ph::VectorCorrection`
combines three scalar callables for the velocity components. Its `x`, `y`, `z`
members correspond to `u_x`, `u_y`, `u_z`, not to spatial derivative directions.

`ph::Corrections{pressure_increment, velocity_increment, temperature_increment}`
groups these helpers. The default `ph::Corrections<>` supplies zeros for all.
They calculate no gradients and perform no halo exchange.

For `Grad` arrays indexed as `(cell, derivative_direction)`, this helper binds
the three derivative components without allocating storage:

```cpp
auto scalar_increment(Grad gradient)
{
    return bc::GradientCorrection{
        Kokkos::subview(gradient, Kokkos::ALL(), 0),
        Kokkos::subview(gradient, Kokkos::ALL(), 1),
        Kokkos::subview(gradient, Kokkos::ALL(), 2)};
}
```

For five existing owned-cell gradients:

```cpp
const auto corrections = ph::Corrections{
    scalar_increment(grad_p),
    ph::VectorCorrection{
        scalar_increment(grad_u), scalar_increment(grad_v), scalar_increment(grad_w)},
    scalar_increment(grad_T)};
boundary.apply_async(primitive, conduction, corrections, time);
```

Corrections affect extrapolated pressure/temperature and slip/symmetry velocity.
A prescribed no-slip velocity or isothermal face temperature remains prescribed.
Gas characteristic inlet/outlet closures use the reader's state directly and
do not additionally apply these increments. Supersonic conditions also ignore
them. In incompressible outward `PressureOutlet`, velocity and temperature use
their increments; reverse-flow data replace that extrapolation. `VelocityInlet`
uses the pressure increment but prescribes velocity and, when thermal, temperature.

Heat flux is an **outward flux density**, without area multiplication. A law
provides a const device-callable `normal_gradient(context, flux)` returning a
value convertible to `Real`. For isotropic conduction, `q_n=-k*g_n`, so positive
outward heat flux gives a negative temperature normal derivative for `k>0`.
The coefficient is thermal conductivity, not thermal diffusivity.

No face coefficient array is needed. For example, use an existing cell field:

```cpp
const auto conduction = tr::IsotropicDiffusion{
    tr::OwnerCellCoefficient{conductivity.view().component(0)}};
boundary.apply_async(primitive, conduction, corrections, time);
```

`OwnerCellCoefficient` explicitly extrapolates the owner coefficient. Other
callable providers may compute properties from existing solution fields.
Their buffers, extents, accessibility and numerical validity are caller
contracts. In `IsotropicDiffusion::normal_gradient`, finite positive coefficient
checking uses `KOKKOS_ASSERT`; a Release build with assertions disabled does
not provide that validation guarantee.

## 9. Traversal and evaluation APIs

| API | Callback | Work performed by the manager |
|---|---|---|
| `for_each_patch(visitor)` | Host: `(PatchRange, concrete_condition)` | Check completeness and dispatch nonempty intervals. No kernel launch. |
| `for_each_face_async(visitor, time)` | Device: `(FaceContext<Real>, concrete_condition)` | Launch typed face traversal with geometry. No field reads or closure evaluation. |
| `evaluate_async(sink, reader, law, corrections, time)` | Device reader: `(context, condition)`; sink: `(context, evaluation)` | Read inside state, resolve an optional local model, evaluate the closure and invoke the sink. |
| `apply_async(fields, law, corrections, time)` | Built-in owner reader and fill sink | Evaluate from owner primitives and write physical face values to extended slots. |
| Free `ph::evaluate(condition, context, inside, model[, law, corrections])` | Called inside an existing kernel | Evaluate one local condition. No traversal, setup preparation or buffer validation. |

The default law is `bc::scalar::NoFluxLaw`; default corrections are zero and
default time is zero. An argument following an optional argument still needs
the preceding arguments explicitly supplied. For example, passing only time
to `apply_async` requires the law and corrections before it.

`evaluate_async` does **not** first call `apply_async` or modify primitive face
slots. Its reader only needs valid owned entries when it is `OwnerReader`.
Sink outputs and any larger extents it uses are the application's responsibility.

For example, record physical and exterior normal velocities without filling
the primitive boundary fields:

```cpp
struct RecordNormalVelocity {
    Kokkos::View<Real*[2], Memory> output; // At least n_boundary_faces rows.

    template<class State>
    KOKKOS_INLINE_FUNCTION void operator()(
        const bc::FaceContext<Real>& c, const ph::Evaluation<State>& result) const
    {
        const auto n = ph::normal(c);
        output(c.boundary_index, 0) = result.face.velocity.dot(n);
        output(c.boundary_index, 1) = result.exterior.velocity.dot(n);
    }
};
```

```cpp
boundary.evaluate_async(
    RecordNormalVelocity{normal_velocity}, ph::OwnerReader{primitive},
    conduction, corrections, time);
```

A custom reader can reconstruct an interior state, read a different storage
representation, or make condition-specific choices. It must return the
primitive type expected by the model. If it supplies reconstruction already,
choose corrections consistently to avoid applying the same increment twice.
An optional `reader.template validate<Exec>(owned_count)` is called on the host.
That hook is user code: keep it to descriptor checks if evaluation must stay
asynchronous. A custom validation that obtains a device reduction result on
the host can make this call block.

`for_each_face_async` is useful when the application needs the actual condition
type, for example to choose a wall relation, thermal relation or matrix block.
It does not demand a heat-flux law merely because a heat-flux patch is bound:
the callback decides whether it needs flux conversion. In contrast,
`evaluate_async` and `apply_async` require a law for any configured built-in
`HeatFluxWall`, even if that patch is locally empty.

Use const device-callable functors and accessible views in callbacks. Named
functors also avoid nested extended CUDA lambdas inside generic host visitors.
`for_each_patch` supplies ranges, not patch IDs; obtain host metadata from
`catalog()` when the application needs catalog information.

## 10. Physical face value and exterior state

`Evaluation<State>` contains two small local values:

| Member | Meaning |
|---|---|
| `face` | Physical face-centroid primitive trace, including prescribed wall velocity/temperature. This is what `apply_async` stores. |
| `exterior` | Auxiliary state for an optional two-state inviscid numerical flux. It is not a mesh cell and is not stored by `apply_async`. |

For wall/symmetry closures, the exterior thermodynamic state remains the
interior one. Velocity is reflected normally about the wall velocity:

\[
u_{ext}=u_i-2n[(u_i-u_{wall})\cdot n].
\]

Tangential velocity is retained in this auxiliary inviscid state. No-slip is
applied to `face` and to the viscous treatment. For example, at a stationary
isothermal wall with `n=(1,0,0)`, inside velocity `(10,20,30)` and temperature
300, the physical trace has velocity `(0,0,0)` and prescribed wall temperature;
the exterior has velocity `(-10,20,30)` and temperature 300. Pressure in the
exterior is also copied from inside, without the face pressure correction.
Current open-condition implementations return the same closed state in both
members. Neither member owns an array or allocates per-face memory.

The application may ignore `exterior` and compute a physical inviscid flux
directly from `face`. For a compressible state on a stationary mesh:

\[
F_n(U_b)=
\begin{pmatrix}
\rho_bu_{n,b}\\
\rho_bu_bu_{n,b}+p_bn\\
(\rho_bE_b+p_b)u_{n,b}
\end{pmatrix},\qquad E_b=e_b+|u_b|^2/2.
\]

At a stationary impermeable wall this reduces to `(0, p_b*n, 0)` per unit
area. For this inviscid contribution only wall pressure is needed. A numerical
flux `Fhat(U_inside,U_exterior,n)` is another boundary discretization and does
not generally equal `F_n(result.face)`. Its wall pressure/dissipation depend on
the selected numerical flux. The boundary module chooses neither approach and
does not multiply a flux by area, add a residual, or enforce a solver-specific
stability property.

## 11. Boundary gradients of primitive variables

The manager does not compute gradients automatically. Two useful applications
of its API are shown below. The examples assume owner gradients are already
valid on the same execution instance. Each output face has a unique index;
writing one output per boundary face needs no atomics.

### Reconstruct full gradients from evaluated face values

The following application reconstructs a normal derivative consistent with
the already evaluated face value:

\[
g_n=\frac{\phi_f-\phi_P-t\cdot\nabla\phi_P}{h}.
\]

It then replaces only the normal component of the owner gradient:

\[
\nabla\phi_f=\nabla\phi_P+n[g_n-n\cdot\nabla\phi_P].
\]

This is an explicit **gradient reconstruction choice**: tangential derivatives
are retained from the owner. For a uniform prescribed wall value, a scheme may
instead enforce zero tangential derivative of that prescribed quantity; curved
surfaces or other formulations may need further treatment. The example does
not claim an exact surface-gradient or curvature closure. It implements the
stated normal correction for each primitive component.

All five `Grad` arrays below are application-owned extended arrays with at
least `boundary.required_field_size()` rows. Their owned prefixes contain
`d/dx`, `d/dy`, `d/dz`; the callback writes only `c.extended_index`. There are
no separate geometry arrays or temporary face-gradient arrays.

```cpp
struct PrimitiveGradients {
    Grad pressure, u, v, w, temperature;
};

template<bool Thermal>
auto primitive_corrections(const PrimitiveGradients& g)
{
    const auto du = ph::VectorCorrection{
        scalar_increment(g.u), scalar_increment(g.v), scalar_increment(g.w)};
    if constexpr (Thermal)
        return ph::Corrections{
            scalar_increment(g.pressure), du, scalar_increment(g.temperature)};
    else
        return ph::Corrections{scalar_increment(g.pressure), du, bc::NoCorrection{}};
}

template<class Fields, bool FillValues = false>
struct FillPrimitiveGradients {
    Fields primitive;
    PrimitiveGradients gradient;

    KOKKOS_INLINE_FUNCTION static void write_gradient(
        Grad g, const bc::FaceContext<Real>& c, Real inside, Real face)
    {
        const auto i = static_cast<std::size_t>(c.owner);
        const Real gx = g(i, 0), gy = g(i, 1), gz = g(i, 2);
        const Real increment = c.tx*gx + c.ty*gy + c.tz*gz;
        const Real normal_derivative = (face-inside-increment)/c.distance;
        const Real change = normal_derivative-(c.nx*gx+c.ny*gy+c.nz*gz);
        g(c.extended_index, 0) = gx+c.nx*change;
        g(c.extended_index, 1) = gy+c.ny*change;
        g(c.extended_index, 2) = gz+c.nz*change;
    }

    KOKKOS_INLINE_FUNCTION void operator()(
        const bc::FaceContext<Real>& c,
        const ph::Evaluation<ph::Primitive<Real, Fields::thermal>>& result) const
    {
        const auto inside = primitive.template read<Real>(
            static_cast<std::size_t>(c.owner));
        const auto& face = result.face;
        write_gradient(gradient.pressure, c, inside.pressure, face.pressure);
        write_gradient(gradient.u, c, inside.velocity.x, face.velocity.x);
        write_gradient(gradient.v, c, inside.velocity.y, face.velocity.y);
        write_gradient(gradient.w, c, inside.velocity.z, face.velocity.z);
        if constexpr (Fields::thermal)
            write_gradient(gradient.temperature, c, inside.temperature, face.temperature);
        if constexpr (FillValues)
            primitive.write(c.extended_index, face);
    }
};
```

Bind existing arrays and submit the gradient sink directly:

```cpp
PrimitiveGradients gradients{grad_p, grad_u, grad_v, grad_w, grad_T};
using Fields = std::remove_cvref_t<decltype(primitive)>;
const auto corrections = primitive_corrections<Fields::thermal>(gradients);

boundary.evaluate_async(
    FillPrimitiveGradients<Fields>{primitive, gradients},
    ph::OwnerReader{primitive}, conduction, corrections, time);
```

For an unheated model, the `temperature` gradient descriptor may be empty;
`if constexpr` prevents its use. With the matching corrections, wall pressure
has zero normal derivative, symmetry/slip follow their velocity closure,
isothermal temperature follows its prescribed face value, and heat-flux
temperature follows the law's normal derivative. At open boundaries, the
example treats the evaluated face trace as the value constraining the normal
reconstruction; it does not derive a separate viscous open-boundary condition.

When both values and gradients are needed, the same callback can do both in
one traversal by selecting `FillValues=true`:

```cpp
primitive.template validate<Exec, true>(boundary.required_field_size());
boundary.evaluate_async(
    FillPrimitiveGradients<Fields, true>{primitive, gradients},
    ph::OwnerReader{primitive}, conduction, corrections, time);
```

Validate all gradient output extents in application setup. Manager validation
of the known correction inputs only checks the owned range; it does not prove
that these sink outputs have extended capacity. Keep the primitive arrays and
individual gradient arrays nonaliasing except for their intentional own-prefix /
boundary-suffix use. No earlier `apply_async` call is required for this example.
It specifically assumes `OwnerReader`; a reconstructed reader would require a
corresponding change to the owner-to-face gradient formula.

### Use concrete wall conditions and exact relation coefficients

`for_each_face_async` gives access to the condition type. This example computes
only the normal derivatives on mechanical walls and symmetry boundaries. It
uses the same relations that can be consumed by an implicit assembler.

```cpp
template<class Fields, class Law, class Correction>
struct WallNormalDerivatives {
    Fields primitive;
    Law law;
    Correction correction;
    Values pressure_dn;
    Grad velocity_dn;       // Compact (boundary_index, velocity_component).
    Values temperature_dn; // May be empty for an unheated model.

    template<class C>
    KOKKOS_INLINE_FUNCTION void operator()(
        const bc::FaceContext<Real>& c, const C& condition) const
    {
        constexpr bool no_slip = std::is_same_v<C, ph::AdiabaticWall<Real>> ||
            std::is_same_v<C, ph::IsothermalWall<Real>> ||
            std::is_same_v<C, ph::HeatFluxWall<Real>>;
        if constexpr (no_slip || std::is_same_v<C, ph::SlipWall<Real>> ||
                      std::is_same_v<C, ph::Symmetry>) {
            const auto inside = primitive.template read<Real>(
                static_cast<std::size_t>(c.owner));
            const auto du = correction.velocity(c);
            ph::VelocityRelation<Real> relation;
            if constexpr (no_slip)
                relation = ph::no_slip_relation(c, condition.velocity, du);
            else if constexpr (std::is_same_v<C, ph::SlipWall<Real>>)
                relation = ph::slip_relation(c, condition.velocity, du);
            else
                relation = ph::slip_relation(c, ph::Vector<Real>{}, du);

            pressure_dn(c.boundary_index) = Real{0};
            const auto gn = relation.normal_gradient(inside.velocity);
            velocity_dn(c.boundary_index, 0) = gn.x;
            velocity_dn(c.boundary_index, 1) = gn.y;
            velocity_dn(c.boundary_index, 2) = gn.z;
            if constexpr (Fields::thermal) {
                Real gn_T = 0;
                if constexpr (no_slip) {
                    const auto thermal = ph::wall_temperature_relation(
                        condition, c, static_cast<Real>(correction.temperature(c)), law);
                    gn_T = thermal.normal_gradient(inside.temperature);
                }
                temperature_dn(c.boundary_index) = gn_T;
            }
        }
    }
};
```

```cpp
boundary.for_each_face_async(
    WallNormalDerivatives{
        primitive, conduction, corrections, dp_dn, du_dn, dT_dn}, time);
```

The output arrays are allocated by the application with at least
`n_faces-n_inner_faces` rows. This callback leaves open-condition entries
untouched; initialize or handle those entries separately if needed. It supports
the listed built-in types, so custom wall types need an explicit additional
branch or their own visitor. It does not first fill primitive boundary slots.
Raw traversal places validation of its inputs, law and outputs on the caller.

## 12. Relations, implicit assembly and Jacobians

`VelocityRelation<Real>` represents two affine vector relations:

\[
u_f=A_u u_P+b_u,\qquad g_n=A_g u_P+b_g.
\]

The data members are `value_owner`, `value_constant`, `gradient_owner` and
`gradient_constant`. Matrices contain `data[3][3]`, so component coupling from
an oblique normal is retained. Call `face_value(owner)` or
`normal_gradient(owner)` to evaluate the corresponding expression.

For no-slip velocity and increment `delta_u`:

\[
A_u=0,\quad b_u=u_{wall},\quad
A_g=-I/h,\quad b_g=(u_{wall}-\delta u)/h.
\]

For slip/symmetry, let `P=I-n*n^T`:

\[
A_u=P,\quad b_u=P\delta u+n(n\cdot u_{wall}),\quad
A_g=-nn^T/h,\quad b_g=n[n\cdot(u_{wall}-\delta u)]/h.
\]

`wall_temperature_relation(condition, context, increment, law)` returns the
scalar module's affine value/normal-gradient `Relation`. It is available for
adiabatic, isothermal and heat-flux no-slip wall types. These helpers supply no
viscosity, conductivity multiplication, face area, residual sign or CSR slot.

An assembler can consume the typed condition through `for_each_face_async`,
form the required relation, and add coefficients to its own matrix. Apply the
equation's flux law and sign convention explicitly. Several faces may share an
owner row; per-cell accumulation needs atomics or an appropriate assembly plan.
An affine component normal derivative alone is not the full viscous stress
Jacobian, which also depends on the stress model and gradient reconstruction.

For a direct boundary residual `R_b=A*F(U_b(U_i))`, differentiation includes the
boundary-state derivative. For a two-state flux it includes the derivative of
the exterior mapping as well. The manager exposes the building blocks and raw
traversal, but generates neither full nonlinear characteristic Jacobians nor a
solver's matrix. Ideal-gas EOS derivatives are available separately; this does
not imply end-to-end automatic differentiation support.

## 13. Custom conditions, EOS and local models

### Custom condition

A custom condition is a small kernel-copyable object with a const
device-callable operator returning `Evaluation<Primitive<Real, Model::thermal>>`
from `(context, inside, model, law, corrections)`. Optional
`validate(model)` is called on the host by `set`.

For example, reuse the existing adiabatic wall closure with a time-dependent
prescribed velocity:

```cpp
struct AcceleratingWall {
    ph::Vector<Real> initial_velocity{}, acceleration{};

    template<class Model>
    void validate(const Model&) const
    {
        const bool finite = std::isfinite(initial_velocity.x) &&
            std::isfinite(initial_velocity.y) && std::isfinite(initial_velocity.z) &&
            std::isfinite(acceleration.x) && std::isfinite(acceleration.y) &&
            std::isfinite(acceleration.z);
        krona::mpi::require(finite, "AcceleratingWall: nonfinite parameter");
    }

    template<class Model, class Law, class Correction>
    KOKKOS_INLINE_FUNCTION auto operator()(
        const bc::FaceContext<Real>& c, ph::Primitive<Real, Model::thermal> inside,
        const Model& model, const Law& law, const Correction& correction) const
    {
        const auto velocity = initial_velocity+acceleration*c.time;
        return ph::evaluate(ph::AdiabaticWall<Real>{velocity},
                            c, inside, model, law, correction);
    }
};
```

```cpp
ph::Manager<GasModel, Exec, Real, AcceleratingWall> boundary{
    host_mesh, mesh_view, model, config, exec};
boundary.set(0, AcceleratingWall{{}, {1, 0, 0}});
boundary.apply_async(primitive, conduction, corrections, time);
```

Choose a physically appropriate wall velocity for the stationary geometry.
This example prescribes velocity; it does not move mesh coordinates or add ALE
terms. Custom types are registered in the template pack and bound in C++.
The TOML `ConfigCondition` variant has no dynamic custom factory registry.
Manager checks for built-in heat-flux conditions do not inspect a custom
condition's internal needs; custom conditions own such contracts.

### Another acoustic EOS and characteristic policy

The `AcousticEOS<EOS, Scalar>` concept describes `from_pT(p,T)` and
`from_rhoe(rho,e)` returning `thermodynamics::State<Scalar>`. This interface
does not require `gamma`, a particular energy formula or transport properties.
The compressible wrapper delegates conversions to those functions.

A custom characteristic policy supplies host `validate(eos)` and const
device-callable methods with the following call shapes:

```text
characteristics.inlet(eos, context, inside, TotalConditionsInlet<Real>)
characteristics.outlet(eos, context, inside, prescribed_pressure)
```

Both return `Primitive<Real, true>`. Their thermodynamic inversions and wave
compatibility formulas are policy-specific. Both methods must be available
for the manager's typed built-in dispatch. `PerfectGasCharacteristics::validate`
accepts the built-in `IdealGas`; supplying an unrelated EOS does not silently
make perfect-gas characteristic formulas valid for it.

With `NoCharacteristics`, wall/symmetry conditions and full supersonic
conditions remain available, provided the EOS supplies their required state
and sound-speed operations. Setting a pressure outlet or total-condition inlet
is rejected during binding.

### Spatially varying material parameters: `model.at`

The usual model is a uniform value copied into kernels. An optional provider
can expose `at(context, condition)` to return a local model instead. For example:

```cpp
struct LocalGasModel {
    static constexpr bool compressible = true;
    static constexpr bool thermal = true;
    static constexpr bool has_characteristics = true;

    GasModel base;
    Kokkos::View<const Real*, Memory> gas_constants;
    std::size_t owned_cells = 0;

    void validate() const
    {
        base.validate();
        krona::mpi::require(gas_constants.extent(0) >= owned_cells &&
            (owned_cells == 0 || gas_constants.data() != nullptr),
            "LocalGasModel: missing owned-cell parameters");
    }

    template<class R, class Condition>
    KOKKOS_INLINE_FUNCTION auto at(
        const bc::FaceContext<R>& c, const Condition&) const
    {
        auto local = base;
        local.eos.gas_constant = gas_constants(static_cast<std::size_t>(c.owner));
        return local;
    }
};
```

```cpp
LocalGasModel local_model{model, gas_constants,
    static_cast<std::size_t>(host_mesh.n_own)};
ph::Manager<LocalGasModel, Exec> boundary{
    host_mesh, mesh_view, local_model, config, exec};
boundary.apply_async(primitive, conduction, corrections, time);
```

Here `gas_constants` is an existing owned-cell Kokkos view. Its values must
already be finite and positive: this example validates the descriptor, not a
device reduction of its contents. A provider owns the full parameter-validation
policy for its data. Choosing `R` per owner illustrates dispatch; it does not
implement species transport, mixture mixing rules or interface thermodynamics.

The kernel uses `if constexpr (requires { model.at(c, condition); })`.
This is a compile-time expression-validity check. If the const call exists,
`at` executes for each evaluated face and its returned model is passed to
`ph::evaluate`. Otherwise the original model is passed directly. Standard
`Compressible` and `Incompressible` have no `at` method and use the latter path.
There is no runtime feature lookup; a provider's actual parameter reads still
cost ordinary device work. Raw `for_each_face_async` does not resolve a model.

Provider capability flags and returned state types must remain compatible.
Host condition preparation sees the provider object; custom `validate(model)`
must account for that if it expects a particular model interface. `model()`
returns a const reference to the stored model/provider. The manager supplies
no setter for replacing a uniform model; provider-backed arrays may evolve
under the caller's execution-order and lifetime contract.

## 14. Validation, asynchronous execution and lifetimes

| Operation | Validation / completion behavior |
|---|---|
| `parse_config`, `read_config` | Host parsing/I/O and structural checks. No mesh matching or device work. |
| Manager construction | Host catalog/count checks, `model.validate()`, geometry attachment and **blocking local geometry validation** on the supplied instance. |
| `set`, `set_unlabelled`, `configure` | Host parameter preparation, direction normalization and model-compatibility checks. `configure` checks full ID/name binding. No explicit execution fence. |
| `for_each_patch` | Host completeness checks and typed nonempty dispatch. |
| `for_each_face_async` | Completeness and finite-time host checks; enqueue raw typed kernels. |
| `evaluate_async` | Also check required built-in flux-law capability, known correction descriptors and optional reader validation. Enqueue evaluation and sink calls. |
| `apply_async` | Also check the primitive thermal capability, extended writable component descriptors and required size. Enqueue physical face filling. |
| Destruction | No fence and no implicit completion of queued users. |

Geometry validation checks exterior owner indices, finite geometry, unit normals,
positive areas and positive normal distances through the shared boundary code.
It returns a host reduction result and construction uses `execution::complete`.
Construct once during setup: this can wait for already submitted work on the
same instance. It is not a full mesh/topology or MPI consistency validation.

Application methods add no explicit fence, host reduction, MPI exchange,
host mirror or field-sized temporary. The `_async` contract means completion
is left to the caller; a CPU backend may nevertheless execute a kernel
synchronously. Selected closure/EOS domain errors call `Kokkos::abort` and may
surface when queued work executes or is synchronized. They do not return an
asynchronous error object. Host failures use KRONA's MPI error convention.

Known field/correction validation inspects descriptors, extents, strides,
writability and accessibility. It does not scan the solution arrays or inspect
arbitrary sinks, readers, coefficient laws and custom condition resources.
Direct `ph::evaluate` assumes prepared condition parameters, normalized
directions, valid context and admissible input state; manager-side setup checks
do not occur when calling that free function directly.

Custom host validation hooks can impose additional completion work of their
own. The absence of manager-inserted fences does not override such a hook.

Submit dependent production work to the manager's stored execution instance:

```cpp
boundary.apply_async(primitive, conduction, corrections, time);
// Submit a dependent solver kernel to boundary.execution_space().

// Only when the host or an external consumer now requires completion:
boundary.execution_space().fence("boundary values needed on host");
```

A fence completes work but does not copy device data to host. Cross-instance
dependencies, host copies and synchronization before MPI are application work.
No boundary operation implicitly refreshes owned primitives, MPI ghosts,
gradients or property arrays.

Keep the compute mesh and every referenced field/helper allocation alive until
all submitted users finish. CPU mesh storage may itself borrow `MeshPart`
arrays, so preserve their owners as required by the mesh module. A copied
descriptor or callback is not a promise to extend an asynchronous allocation's
lifetime. Do not overwrite inputs still used by queued work.

Conditions/model/helpers are copied into submitted typed functors. Changing a
value-only patch binding affects later submissions. If an old binding contains
views, keep the old backing storage alive until its queued users complete.
This contract also applies when a manager is destroyed without an explicit fence.

Each rank operates on its own local catalog and fields. Manager methods perform
no implicit MPI collective or configuration broadcast. Global patch metadata
still makes binding completeness a requirement on ranks with no local work.

## 15. Current limits and future extensions

- Geometry and catalog describe a stationary partition. Mesh adaptation,
  changed connectivity or geometry require an explicit rebuild/reattachment
  design. Wall velocity parameters alone do not implement moving-grid ALE
  fluxes, mesh-relative characteristics or the geometric conservation law.
- The supplied characteristic policy assumes constant perfect-gas heat
  capacities and local normal compatibility. General EOS policies,
  nonreflecting/relaxation outlets and multidimensional characteristic closures
  can be introduced through the explicit policy interface.
- Compressible velocity-prescribed subsonic inlets, farfield conditions and
  custom TOML factories are not built-ins in this version. A velocity-prescribed
  gas inlet needs its own thermodynamic/acoustic closure.
- The built-in primitive state contains pressure, velocity and optional
  temperature. Multispecies boundary data, multiphase variables and coupled
  composition/thermodynamic inflow treatment need further explicit contracts.
- Gradient reconstruction, viscous stresses, flux selection, residual assembly
  and nonlinear Jacobians belong to their consuming discretizations. Typed
  traversal and affine wall relations expose the needed extension points;
  filling face slots alone does not impose a matrix boundary row.
- One launch per nonempty patch preserves simple typed dispatch. Grouping
  patches or kernel fusion may help cases dominated by many tiny intervals.
- Generic custom functors/providers own their validation and buffer contracts.
  Additional asynchronous validation-result APIs could be added separately
  without putting a host reduction into the normal application path.

These are extension points of the current implementation, not additional
behavior performed by the existing manager.
