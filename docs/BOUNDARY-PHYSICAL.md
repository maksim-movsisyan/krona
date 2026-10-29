# Physical boundary conditions

`krona::boundary::physical` evaluates coupled pressure, velocity, and optional temperature conditions on external faces. It supports compressible and incompressible models. It can fill primitive boundary slots or provide boundary states to your own flux, residual, or output kernel.

It does not solve the flow equations, compute a numerical flux, move mesh geometry, or exchange MPI ghosts.

## Architecture and data flow

| Part | Responsibility |
| --- | --- |
| `Model` (`models.hpp`) | Declares compressible/thermal capabilities and supplies an EOS or incompressible material. A compressible characteristic policy closes pressure outlets and total-condition inlets. |
| Condition (`conditions.hpp`) | Holds prescribed wall, inlet, or outlet parameters. Independent of field storage. |
| `Configuration` (`config.hpp`) | Host assignments read from TOML: patch IDs, exact names, conditions, and optional unlabelled-face handling. |
| `PatchCatalog` / `GeometryView` | Copied patch metadata and borrowed external-face geometry, shared with scalar boundaries. |
| `PrimitiveViews` (`fields.hpp`) | Binds existing pressure, velocity, and optional temperature components without allocating or copying field values. |
| `evaluate()` (`evaluate.hpp`) | Device-callable evaluation of one typed condition using a face context, interior primitive state, model, and optional helpers. Returns two states. |
| `Manager<Model, Exec, Real, CustomConditions...>` | Stores assignments, validates setup, dispatches concrete condition types on the host, and launches Kokkos kernels. `Exec` and `Real` default to the project execution space and `ComputeReal`. |

The evaluation pipeline is:

1. The manager selects the condition for a patch.
2. A reader supplies the interior state for each face; `OwnerReader` reads the owner cell.
3. `evaluate()` computes the physical face state and exterior inviscid state.
4. A sink consumes the result. `apply_async()` supplies a sink that writes only the physical face state into the bound fields.

The manager owns the model and condition objects by value. It owns its small patch catalog and borrows mesh geometry. Fields are passed explicitly on each call; no field-name lookup occurs.

## Two different boundary states

`Primitive<Real, false>` contains `pressure` and `velocity`. `Primitive<Real, true>` also contains `temperature`. Velocity is `Vector<Real>{x, y, z}`.

`Evaluation<State>` contains:

| Member | Meaning |
| --- | --- |
| `face` | Physical state at the face centroid. Contains prescribed no-slip wall velocity and, where applicable, wall temperature. Use it for physical boundary values and viscous reconstruction. |
| `exterior` | Exterior inviscid state for a boundary flux computation. It is not a stored ghost-cell value or a value at a reflected cell centre. |

For a stationary isothermal no-slip wall, `face.velocity = 0` and `face.temperature = T_wall`. The exterior state retains the interior pressure and temperature and reflects only the normal velocity. Its tangential velocity remains unchanged. Thus `face` and `exterior` serve different purposes.

**`apply_async()` stores `face` only.** It does not store `exterior`, density, momentum, or energy. Use `evaluate_async()` or direct evaluation when you need those outputs. For a compressible model, `model.conservative(state)` converts a primitive state into density, momentum, and total energy; `model.primitive(state)` performs the reverse conversion.

## Mesh and field contracts

Use the host mesh and device view of the same prepared mesh. Faces must be grouped, geometry must already be computed, and `MeshStorage` must include at least `MeshData::FaceCells | MeshData::Geometry`.

External faces occupy `[n_inner_faces, n_faces)`. MPI interface faces are outside this interval and are not processed. Every external face must have an owned owner cell, finite geometry, outward unit normal, positive area, and positive projected owner-to-face distance. Manager construction validates the required geometry in the selected computation precision.

Extended primitive fields use these intervals:

| Interval | Contents |
| --- | --- |
| `[0, n_own)` | Owned-cell values. |
| `[n_own, n_cells)` | MPI ghost-cell values. |
| `[n_cells, n_cells + n_bfaces)` | Physical face values, one per external face; `n_bfaces = n_faces - n_inner_faces`. |

For external face `f`, the appended slot is `n_cells + f - n_inner_faces`. `required_field_size()` returns the required total size.

`apply_async()` requires writable pressure, three velocity components, and temperature exactly when `Model::thermal == true`. All components must be rank-one floating-point Kokkos views, accessible to `Exec`, with at least `required_field_size()` entries and valid positive strides. Split arrays, SoA fields, and strided AoS components are supported. Component types may differ from `Real`; reads and writes perform conversions. Location tags and field names are not checked.

Bind `FieldView` objects with `primitive_views(p, velocity[, T])`, or separate rank-one views with `primitive_components(p, u, v, w[, T])`. These are borrowed descriptors; use separate, non-overlapping storage for distinct writable components. The manager does not detect aliases.

`OwnerReader` reads the complete owner primitive state, even for prescribed inlets. Supply finite pressure/velocity and positive finite temperature when thermal. Compressible calculations require valid absolute pressure and thermodynamic states; incompressible pressure may be gauge pressure. `apply_async()` writes only appended boundary slots, preserving owned cells, MPI ghosts, and trailing entries.

## Fill primitive boundary values

Sequence: prepare the mesh → select a model → create the manager → assign every condition → bind extended fields → update owner values → apply the conditions.

The examples assume MPI and Kokkos are initialized and `host_mesh` is a prepared `mesh::host::MeshPart`. Shared headers and aliases:

```cpp
#include <krona/boundary/physical/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/thermodynamics/ideal_gas.hpp>
#include <krona/transport/diffusion/isotropic.hpp>
#include <type_traits>

namespace boundary = krona::boundary;
namespace ph = boundary::physical;
namespace thermo = krona::thermodynamics;
namespace device = krona::mesh::device;
namespace fields = krona::fields;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Space = typename Exec::memory_space;
using Model = ph::Compressible<thermo::IdealGas<Real>, ph::PerfectGasCharacteristics>;
using State = ph::Primitive<Real, true>;
```

This example treats all external faces as stationary isothermal walls at `350`:

```cpp
Exec exec{};
device::MeshStorage<Exec> mesh{
    host_mesh,
    {.data = device::MeshData::FaceCells | device::MeshData::Geometry}, exec};
Model model{thermo::IdealGas<Real>{}, ph::PerfectGasCharacteristics{}};
ph::Manager<Model, Exec, Real> manager{host_mesh, mesh.view(), model, exec};

for (const auto& patch : manager.catalog().patches())
    manager.set(patch.range.patch_id, ph::IsothermalWall<Real>{Real{350}, {}});
if (manager.catalog().needs_unlabelled())
    manager.set_unlabelled(ph::IsothermalWall<Real>{Real{350}, {}});

const auto size = manager.required_field_size();
fields::FieldStorage<Real, 1, Exec> pressure{
    "pressure", size, fields::FieldLocation::CellExtnd, exec};
fields::FieldStorage<Real, 3, Exec, Kokkos::LayoutRight> velocity{
    "velocity", size, fields::FieldLocation::CellExtnd, exec};
fields::FieldStorage<Real, 1, Exec> temperature{
    "temperature", size, fields::FieldLocation::CellExtnd, exec};
const auto primitive = ph::primitive_views(
    pressure.view(), velocity.view(), temperature.view());

krona::execution::complete(exec, [&] {
    pressure.fill(exec, Real{100000});
    velocity.fill(exec, Real{10}); // Example owner velocity: (10, 10, 10).
    temperature.fill(exec, Real{300});
    manager.apply_async(primitive);
}); // Initial field fills and boundary kernels are complete.
```

Every labelled patch needs a condition, including patches empty on this rank. If unlabelled faces exist locally or according to the global mesh counts, assign them explicitly with `set_unlabelled()` or configuration. Empty local ranges submit no kernels; a rank without local boundary faces is supported.

Reuse the manager and binding for later updates. After updating owner values, call `apply_async()` again. `set()` changes subsequent submissions; condition parameters are copied into submitted kernels. There is no registration-closing or separate preparation call.

Basic evaluation uses owner values only. Complete any halo exchange needed by your reconstruction or custom reader before using those inputs. This module performs no halo exchange itself.

## Select a model

| Model | Required primitive channels and capabilities |
| --- | --- |
| `Compressible<EOS, Characteristics>` | Pressure, velocity, temperature. EOS supplies thermodynamics; characteristic policy supports total inlets and pressure outlets. The example uses `IdealGas` with `PerfectGasCharacteristics`. |
| `Compressible<EOS>` | Same channels, but no characteristic closure. Suitable for walls and explicit supersonic conditions; total inlets and pressure outlets are rejected. |
| `Incompressible<thermo::ConstantDensity<Real>>` | Pressure and velocity only. Supports mechanical walls, velocity inlets, and pressure outlets. |
| `Incompressible<thermo::ThermalConstantDensity<Real>, true>` | Pressure, velocity, temperature. Also supports isothermal and heat-flux walls. |

For nonthermal models, omit temperature from the field binding. Isothermal and heat-flux conditions require a thermal model. An adiabatic wall remains a valid mechanical no-slip wall without a temperature channel.

Model parameters are validated during manager construction. The provided ideal gas requires finite `gamma > 1`, positive gas constant, and positive reference states; constant-density materials require positive density and, where applicable, positive heat capacity/reference temperature.

`PerfectGasCharacteristics` implements local one-dimensional normal compatibility for a calorically perfect gas. It is not a general multidimensional nonreflecting boundary condition. Use a compatible policy when replacing the EOS.

## Built-in conditions

The table describes zero-correction behaviour. `n` points out of the domain; normal velocity is `u · n`.

| C++ condition | Physical face state and requirements |
| --- | --- |
| `Symmetry{}` | Extrapolates pressure/temperature; removes the normal velocity. Exterior reflects the interior normal velocity. |
| `SlipWall<Real>{wall_velocity}` | Sets normal face velocity to the wall's normal velocity and retains interior tangential velocity. Pressure/temperature are extrapolated. Exterior reflects normal velocity relative to the wall. |
| `AdiabaticWall<Real>{wall_velocity}` | No-slip face velocity; extrapolated pressure. Thermal channel imposes zero heat flux with a supplied law, or zero normal temperature gradient without one. |
| `IsothermalWall<Real>{temperature, wall_velocity}` | No-slip velocity; extrapolated pressure; prescribed positive wall temperature. Requires a thermal model. |
| `HeatFluxWall<Real>{flux, wall_velocity}` | No-slip velocity; extrapolated pressure; prescribed outward heat-flux density converted through an explicit law. Requires a thermal model. |
| `PressureOutlet<Real>{pressure, backflow}` | Compressible: characteristic outlet, including supersonic-outflow and reverse-flow handling below. Incompressible: prescribed pressure and extrapolated velocity/temperature during outward flow. |
| `TotalConditionsInlet<Real>{total_pressure, total_temperature, direction, allow_choking}` | Uses total reservoir data, prescribed inward direction, and an interior characteristic to determine static state and speed. Requires a compressible characteristic model. |
| `VelocityInlet<Real>{velocity, temperature}` | Prescribes velocity and, when thermal, positive temperature; extrapolates pressure. Currently incompressible only. |
| `SupersonicInlet<Real>{pressure, temperature, velocity, mach, direction}` | Prescribes a complete compressible inflow state. `mach == 0` selects explicit velocity; positive `mach` selects `mach * sound_speed * direction`. |
| `SupersonicOutlet{}` | Returns the interior state unchanged. Requires compressible outward normal velocity at least equal to sound speed. |

Wall velocities default to zero and must be finite. Directions are normalized when assigned through `set()` or configuration, and must be finite and nonzero. Direction relative to a particular face and the actual flow regime are checked during device evaluation. Supersonic inflow requires **normal** velocity below `-sound_speed`; a large total Mach number alone is insufficient.

For inlet and outlet conditions, `face` and `exterior` are the same returned state. For no-slip walls, the exterior mirror retains the interior pressure/temperature and tangential velocity; prescribing wall temperature does not change that inviscid mirror.

### Pressure outlet, backflow, and choking

A compressible pressure outlet ignores the target pressure when the interior normal flow is already supersonic outward. Otherwise it applies the characteristic closure. With `PerfectGasCharacteristics`, sufficiently low back pressure produces a sonic outlet limit rather than enforcing an incompatible target pressure. Reverse interior flow, or a characteristic solution that reverses the flow, invokes the configured backflow policy.

An incompressible outlet checks the interior normal velocity for reverse flow. During backflow it preserves that normal velocity, aligns velocity with the supplied inward direction, sets outlet pressure, and sets temperature if thermal.

| `BackflowMode` | Required data |
| --- | --- |
| `Reject` (default) | Reverse flow terminates evaluation. There is no automatic inlet fallback. |
| `TotalConditions` | Compressible only: `reservoir` is a `TotalConditionsInlet` with positive total pressure/temperature and inward direction. |
| `DirectionTemperature` | Incompressible only: inward `direction`, plus positive `temperature` for a thermal model. |

A total-condition inlet rejects a supersonic interior inflow or incompatible reservoir data. If its solution reaches the sonic limit, `allow_choking` must explicitly permit that limit; the default is `false`.

### Heat-flux law and reconstruction corrections

`HeatFluxWall` specifies outward heat-flux **density**, without a face-area factor. Supply a device-callable law with `normal_gradient(context, flux)` returning the temperature derivative. For isotropic conduction, `q_n = -k * dT/dn`:

```cpp
const auto heat_law = krona::transport::IsotropicDiffusion{
    krona::transport::UniformCoefficient<Real>{Real{2}}};
manager.apply_async(primitive, heat_law);
exec.fence("Physical boundary values ready");
```

For example, flux `10` with `k = 2` gives `dT/dn = -5`. Heat-flux walls require a law even when their patches are empty locally. The isotropic conversion requires finite, strictly positive conductivity. Coefficient values are caller contracts, not scanned by the manager on each application. An adiabatic wall without a law uses a zero normal gradient; with a law it asks that law to convert zero flux.

The correction argument is `Corrections{pressure_provider, velocity_provider, temperature_provider}`. Defaults are zero scalar increments and zero vector increment. Each provider receives `FaceContext`; velocity returns `Vector<Real>`, the others return a scalar increment.

For a scalar, `boundary::GradientCorrection{gx, gy, gz}` computes `delta = (x_f - x_P - h*n) · grad(phi_P)`, where `h = (x_f - x_P) · n`. Components need at least `n_own` entries. `VectorCorrection{u_provider, v_provider, w_provider}` combines three scalar providers. Supply valid, current gradients before evaluation.

Pressure increments affect symmetry/slip/no-slip walls and velocity inlets. Velocity increments affect symmetry/slip face reconstruction and outward incompressible pressure outlets. Temperature increments affect extrapolated symmetry/slip states, thermal wall reconstruction, and outward incompressible pressure outlets. Prescribed no-slip velocity, prescribed inlet states, and characteristic states are not replaced by these increments. Thermal wall relations use `T_f = T_P + h*dT/dn + delta`; an isothermal face remains exactly prescribed.

## Configure from TOML

Physical condition parameters are at patch-table level, unlike the scalar module's nested field tables. Example for patch `0` named `wall` and patch `1` named `outlet`:

```toml
[[boundary_condition]]
patch_id = 0
name = "wall"
type = "ISOTHERMAL_WALL"
temperature = 350.0
velocity = [0.0, 0.0, 0.0]

[[boundary_condition]]
patch_id = 1
name = "outlet"
type = "PRESSURE_OUTLET"
pressure = 100000.0
[boundary_condition.backflow]
type = "TOTAL_CONDITIONS_INLET"
total_pressure = 120000.0
total_temperature = 320.0
direction = [-1.0, 0.0, 0.0]
allow_choking = false

[unlabelled]
type = "ADIABATIC_WALL"
```

The backflow direction here assumes an outlet normal along positive x. Adapt it to your geometry. Read and bind the configuration instead of the programmatic assignments:

```cpp
const auto config = ph::read_config("boundary.toml");
ph::Manager<Model, Exec, Real> configured{
    host_mesh, mesh.view(), model, config, exec};
configured.apply_async(primitive, heat_law);
exec.fence("Configured physical boundaries ready");
```

`parse_config(text)` parses an existing string; `manager.configure(config)` replaces assignments. `read_config()` performs local file I/O on each calling rank, with no broadcast. For root-only reading, broadcast the text yourself and parse it on each rank.

| TOML `type` | Parameters |
| --- | --- |
| `SYMMETRY` | None. |
| `SLIP_WALL`, `ADIABATIC_WALL` | Optional `velocity`, default zero. |
| `ISOTHERMAL_WALL` | `temperature`; optional `velocity`. |
| `HEAT_FLUX_WALL` | `flux`; optional `velocity`. |
| `PRESSURE_OUTLET` | `pressure`; optional `backflow` table, default rejection. |
| `TOTAL_CONDITIONS_INLET` | `total_pressure`, `total_temperature`, `direction`; optional `allow_choking = false`. |
| `VELOCITY_INLET` | `velocity`; `temperature` required when bound to a thermal model. |
| `SUPERSONIC_INLET` | `pressure`, `temperature`, and either `velocity` or positive `mach` with `direction`. Do not supply both forms. |
| `SUPERSONIC_OUTLET` | None. |

Backflow tables accept `type = "REJECT"`, `"TOTAL_CONDITIONS_INLET"` with reservoir parameters, or `"DIRECTION_TEMPERATURE"` with direction and thermal temperature. Physical type strings are case-sensitive. Recognized legacy aliases are:

| Canonical spelling | Alias |
| --- | --- |
| `ADIABATIC_WALL`, `ISOTHERMAL_WALL` | `WALL_ADIABATIC`, `WALL_ISOTHERMAL` |
| `PRESSURE_OUTLET`, `TOTAL_CONDITIONS_INLET` | `OUTLET_BACK_PRESSURE`, `INLET_TOTAL_TP` |
| `SUPERSONIC_INLET`, `SUPERSONIC_OUTLET` | `INLET_SUPERSONIC`, `OUTLET_SUPERSONIC` |
| `pressure`, `temperature` | `p_back` for outlet pressure; `p`, `T` for supersonic inflow; `T_wall` for wall temperature |
| `total_pressure`, `total_temperature`, `mach` | `p0`, `T0`, `M` |

Do not supply a parameter and its alias together. Every labelled patch must appear exactly once with matching ID and exact name; order is irrelevant. Unlabelled faces need their own physical condition. With no labelled patches, write `boundary_condition = []` explicitly.

Unexpected condition keys are rejected. Import metadata (`family`, `source_type`, `global_face_count`) and nested scalar tables are accepted, but **`scalar.temperature` conflicts with the physical temperature channel and is rejected**. Other scalar fields can coexist. Numbers/vectors must be finite; model-specific positivity and compatibility are checked when assigning conditions. Custom conditions are assigned in C++, not parsed from TOML. `FARFIELD` and `USER_DEFINED` are not implemented physical types in this parser, even if a mesh-generated template suggests them.

## Use a custom reader or sink

`evaluate_async(sink, reader, law, corrections, time)` evaluates all assigned conditions without automatically writing fields. `law`, `corrections`, and `time` default to no flux law, zero increments, and zero time.

- `reader(context, typed_condition)` runs on the device and returns `Primitive<Real, Model::thermal>`. Use `OwnerReader{primitive}` for cell-centred data, or supply your own reconstructed interior trace. If the reader defines `template<class Exec> validate(std::size_t owned_count)`, the manager calls it on the host before submission.
- `sink(context, evaluation)` runs on the device and consumes both states. Its output layout and write conflicts are your responsibility.

For example, store both states in boundary-only arrays. Define this sink at namespace scope:

```cpp
struct SaveStates {
    Kokkos::View<State*, Space> face, exterior;

    KOKKOS_INLINE_FUNCTION void operator()(
        const boundary::FaceContext<Real>& c,
        const ph::Evaluation<State>& result) const
    {
        face(c.boundary_index) = result.face;
        exterior(c.boundary_index) = result.exterior;
    }
};
```

Launch with the manager and field binding from the first example:

```cpp
const auto boundary_count = manager.required_field_size()
    - static_cast<std::size_t>(host_mesh.n_cells);
Kokkos::View<State*, Space> face_states("physical_face_states", boundary_count);
Kokkos::View<State*, Space> exterior_states("exterior_states", boundary_count);
manager.evaluate_async(SaveStates{face_states, exterior_states},
                       ph::OwnerReader{primitive}, heat_law);
exec.fence("Both boundary states ready");
```

The sink can instead call `model.conservative(result.exterior)` and feed a numerical flux. Supply the matching interior trace to that flux as well. This module does not select a flux scheme or assemble contributions for you.

## Launch your own Kokkos kernel

`for_each_patch(visitor)` calls `visitor(PatchRange, typed_condition)` on the host for each nonempty local patch, including unlabelled faces with `patch_id == -1`. The range is an absolute local face interval `[begin, end)`.

This example evaluates conditions directly inside your kernel and stores both states. It uses the same `Model` and `State` aliases as above; define it at namespace scope:

```cpp
template<class Condition, class Reader>
struct BoundaryKernel {
    boundary::GeometryView<Space> geometry;
    Condition condition;
    Model model;
    Reader reader;
    SaveStates output;
    Real time;

    KOKKOS_INLINE_FUNCTION void operator()(krona::LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto inside = reader(c, condition);
        const auto result = ph::evaluate(condition, c, inside, model);
        output(c, result);
    }
};
```

Launch it using the previously allocated outputs:

```cpp
const auto reader = ph::OwnerReader{primitive};
using Reader = std::remove_cvref_t<decltype(reader)>;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<krona::LocalIndex>>;
manager.for_each_patch([&](boundary::PatchRange range, const auto& condition) {
    using Condition = std::remove_cvref_t<decltype(condition)>;
    Kokkos::parallel_for("custom_physical_boundary",
        Policy(manager.execution_space(), range.begin, range.end),
        BoundaryKernel<Condition, Reader>{
            manager.geometry_view(), condition, manager.model(), reader,
            SaveStates{face_states, exterior_states}, Real{1.5}});
});
manager.execution_space().fence("Custom boundary kernel complete");
```

The four-argument `evaluate()` uses no flux law and zero corrections. For heat-flux conditions or reconstruction, capture helpers and call `evaluate(condition, c, inside, model, law, corrections)` instead. The shown assignments are all isothermal walls, so no heat-flux law is needed.

Direct evaluation assumes prepared condition parameters and valid geometry, states, and helper buffers. Conditions obtained from a manager have already passed host assignment checks. Your launch must provide valid extents, finite time, and any additional state checks; it does not automatically run the manager's reader/provider validation or its compressible output-pressure check.

An alternative is `for_each_face_async(sink, time)`: the manager launches kernels calling `sink(context, typed_condition)` without reading fields, evaluating a condition, or requiring a heat-flux law. It is useful for completely custom evaluation or direct access to prescribed parameters. Typed visitors/readers must compile for every condition alternative in the manager, not only currently assigned types.

Each external face has a unique boundary slot. If your kernel accumulates into owner-cell residuals, several faces may share an owner; use atomics or another safe assembly strategy.

## Add a custom condition

A custom condition implements a device-callable function taking `(context, inside, model, law, corrections)` and returning `Evaluation<Primitive<Real, Model::thermal>>`. Add its type to the manager's template arguments. An optional host `validate(model)` method is called when assigning it.

This example delegates to the built-in adiabatic wall with time-dependent wall velocity:

```cpp
struct AcceleratingWall {
    Real acceleration = 2;

    template<class R, class M, class Law, class Correction>
    KOKKOS_INLINE_FUNCTION auto operator()(
        const boundary::FaceContext<R>& c, ph::Primitive<R, M::thermal> inside,
        const M& model, const Law& law, const Correction& correction) const
    {
        const ph::Vector<R> velocity{static_cast<R>(acceleration)*c.time, R{0}, R{0}};
        return ph::evaluate(ph::AdiabaticWall<R>{velocity},
                            c, inside, model, law, correction);
    }
};
```

Create a separate manager, complete its assignments, and apply it. This example requires labelled patch `0`:

```cpp
ph::Manager<Model, Exec, Real, AcceleratingWall> custom{
    host_mesh, mesh.view(), model, exec};
for (const auto& patch : custom.catalog().patches())
    custom.set(patch.range.patch_id, ph::AdiabaticWall<Real>{});
if (custom.catalog().needs_unlabelled())
    custom.set_unlabelled(ph::AdiabaticWall<Real>{});
custom.set(0, AcceleratingWall{Real{2}});
custom.apply_async(primitive, heat_law, ph::Corrections<>{}, Real{1.5});
exec.fence("Custom wall values ready");
```

The assigned wall velocity on patch `0` is `(3, 0, 0)`. Prescribing this velocity does not move the mesh. Custom conditions must return physically valid finite states, and their captured buffers must be accessible and remain valid during execution. The manager does not automatically validate arbitrary custom parameters or recompute an EOS closure for their outputs.

Custom models declare `compressible`, `thermal`, and `has_characteristics`, provide host `validate()`, and implement the operations needed by the selected conditions. If `model.at(context, condition)` is available, the manager's `evaluate_async()` uses that device-selected local model for evaluation; this supports existing per-cell material buffers without precomputing a face model array. A manually launched kernel must perform that selection itself if needed.

## Relations for viscous or implicit work

Primitive evaluation returns states, not derivative relations. The following helpers are available separately in `relations.hpp`:

| Helper | Result |
| --- | --- |
| `no_slip_relation(context, wall_velocity, vector_increment)` | `VelocityRelation`: prescribed wall velocity and `(wall_velocity - owner_velocity - increment) / distance` normal derivative. |
| `slip_relation(context, wall_velocity, vector_increment)` | Coupled velocity relation: prescribed normal velocity with tangential reconstruction and zero tangential normal derivative. |
| `wall_temperature_relation(wall, context, scalar_increment, law)` | `scalar::Relation<Real>` for adiabatic, isothermal, or heat-flux walls. |

`VelocityRelation` stores matrices `value_owner`, `gradient_owner` and vectors `value_constant`, `gradient_constant`. Use `face_value(owner_velocity)` and `normal_gradient(owner_velocity)` to evaluate them. Slip couples components through the normal projection, so it is not three independent scalar Dirichlet conditions. These helpers do not apply viscosity, conductivity, or face-area factors, except for the supplied flux-to-gradient law used in a thermal relation.

## Face context

| Member | Meaning |
| --- | --- |
| `face`, `owner` | Absolute local face index and owned-cell index. |
| `boundary_index` | `face - n_inner_faces`, for boundary-only arrays. |
| `extended_index` | `n_cells + boundary_index`, for extended cell fields. |
| `x, y, z` | Face centroid. |
| `nx, ny, nz` | Outward unit normal. |
| `distance` | Positive normal projection `(x_f - x_P) · n`, not Euclidean distance. |
| `tx, ty, tz` | Tangential displacement `x_f - x_P - distance*n`. |
| `area` | Face area; multiply flux density by it explicitly when integrating. |
| `time` | Supplied finite time, default zero. |

## Asynchrony and lifetime

| Operation | Completion behaviour |
| --- | --- |
| `Manager(...)` | Blocking local setup: copies patch metadata, validates model/geometry, and fences its execution-space instance before returning. |
| `read_config()`, `parse_config()`, `configure()`, `set()`, `set_unlabelled()` | Complete host work before return. They do not wait for earlier boundary kernels. |
| `apply_async()` | Validates field descriptors and submits physical-face filling. No final fence. |
| `evaluate_async()` | Validates known inputs and submits reader/evaluation/sink kernels. No final fence. |
| `for_each_face_async()` | Submits raw typed callbacks. No final fence. |
| `for_each_patch()` | Completes host dispatch before return; does not wait for kernels launched by the visitor. |
| `execution_space().fence()` | Waits for work on that execution-space instance. |

These operations perform no MPI communication or barrier. Local setup completion does not wait for other ranks. A synchronous backend may finish a kernel during submission; methods without a final fence provide no portable completion guarantee.

Submit dependent producers, boundary work, and consumers on the same execution-space instance for ordering. Synchronize explicitly before dependent work on another instance or conflicting host/MPI access. Keep mesh arrays, fields, reader/sink outputs, and model/law/correction backing buffers alive until their submitted users finish. Changing host condition assignments affects later submissions; changing referenced buffer contents still requires correct execution ordering.

The manager has no destructor fence. Complete outstanding work before releasing backing data or finalizing Kokkos. Custom kernels and callbacks have the same lifetime and ordering responsibilities.
