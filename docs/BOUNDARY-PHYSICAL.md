# Physical boundary conditions

`krona::boundary::physical` evaluates coupled pressure, velocity and optional temperature conditions on exterior faces. It fills physical face slots or supplies two states to a caller's kernel. It performs no halo exchange, numerical flux evaluation or equation solve.

## Objects and ownership

| Object | Responsibility |
| --- | --- |
| `BoundaryLayout<Exec, Real>` | Owns one host patch catalog and borrows mesh geometry. Share it between scalar/physical managers using the same mesh, execution instance and computation precision. |
| `Manager<Model, Exec, Real, ExtraConditions...>` | Borrows the layout; owns its model and condition objects. Fields remain arguments to operations. |
| Numeric conditions (`conditions.hpp`) | Store prescribed values and policy flags. Existing aggregate syntax remains valid. |
| `Profiled<Condition, Provider>` (`profiles.hpp`) | Stores a provider that constructs one numeric condition at the current face/time. Factories combine individual scalar/vector providers. |
| `PrimitiveViews` (`fields.hpp`) | Binds existing pressure, three velocity components and optional temperature. No field-value copy or allocation. |
| `evaluate()` | Device-local evaluation of one condition and interior state. Returns `Evaluation{face, exterior}`. |
| `Status<Exec>` | Owns reusable device flags and a host mirror for explicitly checked submissions. |

`Primitive<Real, false>` contains pressure and velocity; `Primitive<Real, true>` also contains temperature. States remain in this module. `Vector<Real>` and `Matrix<Real>` name the common `math::Vector3` and `math::Matrix3` types.

**The two returned states have different purposes:**

| State | Meaning |
| --- | --- |
| `face` | Physical face-centred trace: includes prescribed no-slip velocity and viscous wall temperature. |
| `exterior` | Exterior inviscid state for a numerical boundary flux; not a stored ghost-cell value. |

For a stationary isothermal wall, the face has zero velocity and the wall temperature. The exterior keeps interior pressure, temperature and tangential velocity, and reverses normal velocity. `apply_async()` stores **only `face`**, never exterior or conservative states.

## Models and mesh contracts

| Model | Capabilities |
| --- | --- |
| `Compressible<EOS, Characteristics = NoCharacteristics>` | Thermal primitive states; EOS conversion. Pressure outlets and total-condition inlets additionally require a compatible characteristic policy. |
| `Incompressible<Material, Thermal = false>` | Prescribed-density material; optional temperature equation. Pressure may be gauge pressure. |

`Compressible::conservative(state)` returns density, momentum density and total-energy density. `primitive(conservative)` inverts them using the EOS. Energy references belong to the EOS. These are local device functions, not field-transfer operations.

Use host/device descriptions of the same prepared mesh. Faces must be grouped and the mesh view must contain at least `FaceCells | Geometry`. Exterior faces are `[n_inner_faces, n_faces)`; interface faces are not processed. Layout construction validates owned owners, finite geometry, outward unit normals, positive areas and positive normal owner-to-face distances.

Extended fields contain owned cells, real MPI ghosts, then one face value per exterior face. Face `f` uses slot `n_cells + f - n_inner_faces`; `required_field_size()` returns `n_cells + n_faces - n_inner_faces`.

All bound components must be accessible, rank-one floating-point Kokkos views with sufficient extent and valid storage/strides. Split arrays and strided AoS components work. Use non-overlapping writable components and do not alias outputs with concurrently read profile inputs. Names/location tags are not checked.

`OwnerReader` reads the complete owner state, including for prescribed inlets. Supply finite pressure/velocity, positive finite thermal temperature and positive compressible absolute pressure. Values read or sampled into `Real` must be representable in that type and valid for the EOS/material. Standard filling checks output representability before writing any component of a face; rounding/underflow are not diagnosed. Owned cells, MPI ghosts, other storage and trailing entries are preserved.

## Prepare and fill fields

Assume active MPI/Kokkos and a prepared `host_mesh`:

```cpp
#include <krona/boundary/physical/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/transport/diffusion/isotropic.hpp>

namespace bc = krona::boundary;
namespace ph = bc::physical;
namespace grad = krona::numerics::gradient;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Storage = krona::StorageReal;
using Space = typename Exec::memory_space;
using Model = ph::Compressible<krona::thermodynamics::IdealGas<Real>,
                               ph::PerfectGasCharacteristics>;
using State = ph::Primitive<Real, Model::thermal>;
```

Create one layout, configure the manager and bind extended fields:

```cpp
Exec exec{};
krona::mesh::device::MeshStorage<Exec> mesh(host_mesh,
    {.data = krona::mesh::device::MeshData::FaceCells |
             krona::mesh::device::MeshData::Geometry}, exec);
bc::BoundaryLayout<Exec, Real> layout(host_mesh, mesh.view(), exec);
Model model{};
ph::Manager<Model, Exec, Real> boundary(layout, model);
for (const auto& patch : layout.catalog().patches())
    boundary.set(patch.range.patch_id, ph::Symmetry{});
if (layout.catalog().needs_unlabelled()) boundary.set_unlabelled(ph::Symmetry{});
boundary.finalize();

const auto count = layout.required_field_size();
krona::fields::FieldStorage<Storage, 1, Exec> pressure(
    "p", count, krona::fields::FieldLocation::CellExtnd, exec);
krona::fields::FieldStorage<Storage, 3, Exec> velocity(
    "u", count, krona::fields::FieldLocation::CellExtnd, exec);
krona::fields::FieldStorage<Storage, 1, Exec> temperature(
    "T", count, krona::fields::FieldLocation::CellExtnd, exec);
pressure.fill_async(exec, Storage{100000});
velocity.fill_async(exec, Storage{0});
temperature.fill_async(exec, Storage{300});
const auto primitive = ph::primitive_views(pressure.view(), velocity.view(), temperature.view());
const auto heat_law = krona::transport::IsotropicDiffusion{
    krona::transport::UniformCoefficient<Real>{Real{2}}};
ph::Status<Exec> status(exec);
status.begin_async();
boundary.apply_async(primitive, status, heat_law);
status.check();
```

Assign every labelled patch, even if locally empty. Unlabelled faces need an explicit assignment when present locally or according to global mesh counts. No default condition is supplied. For a nonthermal model, bind `primitive_views(pressure, velocity)` without temperature. Separate rank-one arrays use `primitive_components(p, u, v, w[, T])`.

`set()`, `set_unlabelled()` and `configure()` invalidate finalization. Call `finalize()` after assignments, including time-step updates made through `set()`. It checks completeness/provider metadata and caches flux-law requirements; it does not scan device values, submit work, fence or communicate. Updating bound buffer contents or the evaluation time does not require refinalization.

## Built-in conditions

Let `n` be the outward unit normal, `h = (x_f - x_P)·n > 0`, and `delta` a supplied reconstruction increment. Table entries describe numeric conditions; the profile factories below produce the same closures.

| Condition | Behaviour and requirements |
| --- | --- |
| `Symmetry` | Projects reconstructed face velocity onto the tangent plane; exterior reflects normal velocity. Face pressure/thermal temperature include their correction increments. |
| `SlipWall<Real>{velocity}` | Same tangential treatment, with face normal velocity set by the wall. Exterior reflects velocity relative to the wall. |
| `AdiabaticWall<Real>{velocity}` | No-slip face velocity; zero outward heat flux. Without a supplied law this uses zero normal temperature derivative; an explicit law converts zero flux. |
| `IsothermalWall<Real>{temperature, velocity}` | No-slip velocity and positive prescribed face temperature. Filling does not read temperature correction/gradients. |
| `HeatFluxWall<Real>{flux, velocity}` | No-slip velocity and prescribed outward heat-flux density. Requires a flux-to-gradient law and a thermal model. |
| `VelocityInlet<Real>{velocity, temperature}` | Incompressible only: prescribed full velocity and thermal temperature; pressure extrapolates with correction. It does not enforce an inward velocity direction. |
| `TotalConditionsInlet<Real>{p0, T0, direction, allow_choking}` | Compressible characteristic inlet with positive total pressure/temperature and an inward direction. |
| `PressureOutlet<Real>{pressure, backflow}` | Prescribed-pressure closure; compressible flow requires characteristics. Reverse flow uses explicit backflow data. |
| `SupersonicInlet<Real>{pressure, temperature, velocity, mach, direction}` | Compressible prescribed state. `mach == 0` uses velocity; positive Mach uses `mach * sound_speed * direction`. Resulting normal inflow must be strictly supersonic. |
| `SupersonicOutlet` | Compressible extrapolation; requires outward normal speed at least equal to sound speed. |

Wall velocity is a physical velocity, not a normalized direction. Constant inlet/backflow directions are normalized when assigned to a manager; profile directions are normalized after sampling. Numeric conditions evaluated directly outside a manager require caller-normalized directions. All direction vectors must be finite and nonzero where required.

For no-slip walls, face pressure extrapolates with its correction. Wall thermal data act on the physical trace; exterior pressure/temperature remain interior. For isotropic conduction, `q_n = -k*dT/dn`: positive flux leaves the domain. `flux` is a density, without face area. A custom law supplies device `normal_gradient(context, flux)` returning a derivative; its coefficient/domain checks remain its responsibility.

### Characteristic inlet/outlet and backflow

`PerfectGasCharacteristics` is a calorically perfect-gas, local normal closure. It preserves normal compatibility `u_n + 2*a/(gamma - 1)`, not a multidimensional nonreflecting condition. A different EOS needs its own compatible policy.

The total inlet combines compatibility with reservoir entropy and total enthalpy. Supersonic interior inflow and incompatible reservoir data are rejected. `allow_choking` permits the sonic fallback when the requested compatibility has no solution on the subsonic inlet branch; it defaults to false.

A compressible pressure outlet extrapolates an already supersonic outward interior state. Otherwise it uses the characteristic closure. With `PerfectGasCharacteristics`, a target pressure that would produce a supersonic boundary state is replaced by a compatible sonic state; both its speed and pressure are determined by that limit. Interior reverse flow, or a closure that reverses flow, invokes backflow handling.

| Backflow mode | Data and action |
| --- | --- |
| `Reject` (default) | Actual reverse flow is an error. |
| `TotalConditions` | Compressible only: use the configured reservoir inlet closure. |
| `DirectionTemperature` | Incompressible only: impose the inward direction while preserving interior normal speed; use outlet pressure and prescribed thermal temperature. |

## Profiles and parameter providers

Common providers live in `boundary/profiles.hpp`. A parameter can be a number, a `math::Vector3`, `profile::Constant<T>`, a device callable `provider(context)`, or `profile::patch_values(range, view)`. A raw Kokkos view needs an explicit indexing adapter; it is not a face-context provider by itself.

Factories return `Profiled<NumericCondition, Sampler>` objects. List each distinct additional condition type once in the manager template. Sampling constructs the ordinary numeric condition; physical interpretation and validation belong to that condition, not the provider.

| Factory | Profile-capable parameters |
| --- | --- |
| `slip_wall<Real>(velocity)` | Wall velocity. |
| `adiabatic_wall<Real>(velocity = {})` | Wall velocity. |
| `isothermal_wall<Real>(temperature, velocity = {})` | Temperature and velocity. |
| `heat_flux_wall<Real>(flux, velocity = {})` | Heat flux and velocity. |
| `velocity_inlet<Real>(velocity, temperature = {})` | Velocity and thermal temperature. |
| `total_conditions_inlet<Real>(p0, T0, direction, allow_choking = false)` | Total pressure, temperature and direction; the policy flag is a bool. |
| `supersonic_inlet<Real>(pressure, temperature, velocity)` | Static pressure, temperature and explicit velocity. |
| `supersonic_mach_inlet<Real>(pressure, temperature, mach, direction)` | Static pressure, temperature, Mach and direction. |
| `pressure_outlet<Real>(pressure, backflow = {})` | Pressure and optional profiled backflow data. |
| `total_backflow<Real>(reservoir)` | Reservoir returned by `total_conditions_inlet()`, or an ordinary numeric reservoir. |
| `directional_backflow<Real>(direction, temperature = {})` | Incompressible backflow direction and thermal temperature. |

For example, define providers at namespace scope:

```cpp
struct WallTemperature {
    KOKKOS_INLINE_FUNCTION Real operator()(const bc::FaceContext<Real>& c) const
    { return Real{350} + Real{2}*c.y + Real{5}*c.time; }
};
struct WallVelocity {
    KOKKOS_INLINE_FUNCTION ph::Vector<Real> operator()(const bc::FaceContext<Real>& c) const
    { return {Real{0}, Real{10}*c.time, Real{0}}; }
};
```

This fragment requires labelled patch `0`:

```cpp
const auto wall = ph::isothermal_wall<Real>(WallTemperature{}, WallVelocity{});
ph::Manager<Model, Exec, Real, decltype(wall)> profiled(layout, model);
for (const auto& patch : layout.catalog().patches())
    profiled.set(patch.range.patch_id, ph::Symmetry{});
if (layout.catalog().needs_unlabelled()) profiled.set_unlabelled(ph::Symmetry{});
profiled.set(0, wall);
profiled.finalize();
status.begin_async();
profiled.apply_async(primitive, status, heat_law, ph::Corrections<>{}, Real{0.5});
status.check();
```

For a compact patch array, replace the `wall` declaration with:

```cpp
const auto range = layout.catalog().patches()[0].range;
Kokkos::View<Real*, Space> patch_temperature(
    "wall_temperature", static_cast<std::size_t>(range.end - range.begin));
Kokkos::deep_copy(exec, patch_temperature, Real{350});
const auto wall = ph::isothermal_wall<Real>(bc::profile::patch_values(range, patch_temperature));
```

The array must have exactly the local patch face count, including zero for an empty patch. Indexing is `context.face - range.begin`, not a global ID, cell index or boundary-slot index. Its bound range/patch ID must match the assigned patch. Const/strided scalar views and views of `math::Vector3` work. Use `catalog().unlabelled()` for unlabelled profiles.

`patch_values()` borrows pointer/stride metadata and retains no allocation ownership. Update buffers on the layout's execution instance before the next use; no rebinding/refinalization is needed. Keep them alive through completion. Constants are stored inline; custom providers may borrow their own storage. Optional `validate<Exec>(PatchRange)` hooks check host metadata only.

A generic `profiled<Condition>(provider)` accepts a device provider returning exactly that numeric/custom condition type. `sample_condition(condition, context)` exposes the raw numeric snapshot; it does not normalize or validate it. `condition_type_t<C>` identifies the underlying condition kind in typed callbacks. Standard evaluation samples each factory parameter once per face; custom callbacks must control any additional sampling they perform.

For a profiled compressible reverse-flow reservoir:

```cpp
const auto reservoir = ph::total_conditions_inlet<Real>(
    Real{130000}, Real{330}, ph::Vector<Real>{-1, 0, 0});
const auto outlet = ph::pressure_outlet<Real>(Real{90000}, ph::total_backflow<Real>(reservoir));
```

The existing Euler adapters recognize profiled conditions. `PreparedExterior` uses prepared inlet/pressure slots; `EvaluatedExterior` evaluates them again. Factory wall profiles supply their mechanical velocity without sampling temperature/heat-flux parameters for an inviscid mirror.

## Corrections and affine helpers

`Corrections{pressure, velocity, temperature}` supplies reconstruction increments. Pressure/temperature defaults use `numerics::gradient::NoCorrection`; velocity defaults to a zero vector. These are trace increments, not already-integrated fluxes.

For owned-cell temperature gradients `gx`, `gy`, `gz`:

```cpp
const auto temperature_increment = grad::GradientCorrection{grad::cell_gradients(gx, gy, gz)};
const ph::Corrections correction{
    grad::NoCorrection{}, ph::ZeroVelocityIncrement{}, temperature_increment};
status.begin_async();
boundary.apply_async(primitive, status, heat_law, correction);
status.check();
```

Projection is `(x_f - x_P - h*n)·grad(T_P)`. Views must be accessible and long enough for owned cells. `cell_gradients(field, component)` also selects the gradient module's `3*component + axis` layout. `VectorCorrection{x, y, z}` combines three scalar policies into a velocity increment; each can use the shared gradient correction. Gradient computation/exchange belongs to the application.

`no_slip_relation()` and `slip_relation()` return `VelocityRelation<Real>`: matrix/vector affine expressions for face velocity and outward normal derivative. Slip uses `I - n*n^T`, so its component coupling generally needs a full matrix. `wall_temperature_relation(numeric_wall, context, increment, law)` returns the scalar affine temperature relation. Helpers contain no area factor, allocation, launch or synchronization; direct callers supply valid geometry/parameters.

## Read interior states and consume both traces

`evaluate_async(sink, reader, status, law, correction, time)` calls device `reader(context, typed_condition)`, evaluates the closure and then calls `sink(context, evaluation)` for successful faces. Use a custom reader for reconstructed interior traces. Optional `reader.validate<Exec>(owned_count)` checks host metadata before submission.

For example, define this sink at namespace scope:

```cpp
struct SaveTraces {
    Kokkos::View<State*, Space> face, exterior;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c,
                                          const ph::Evaluation<State>& result) const
    { face(c.boundary_index) = result.face; exterior(c.boundary_index) = result.exterior; }
};
```

Then launch:

```cpp
const auto boundary_count = layout.required_field_size() - static_cast<std::size_t>(host_mesh.n_cells);
Kokkos::View<State*, Space> face_trace("face_trace", boundary_count);
Kokkos::View<State*, Space> exterior_state("exterior_state", boundary_count);
status.begin_async();
boundary.evaluate_async(SaveTraces{face_trace, exterior_state}, ph::OwnerReader{primitive}, status, heat_law);
status.check();
```

The manager checks returned states, not a custom sink's arithmetic/output buffers. Validate those yourself. Several faces can share an owner; accumulation into owner-cell arrays needs atomics or another safe strategy.

## Your own kernel

`for_each_face_async()` passes `(context, typed_condition)` to a device functor. `for_each_patch()` passes `(PatchRange, typed_condition)` to a host visitor for manual launches. Both require finalization and skip empty local patches; neither imposes a flux law, reader or diagnostic policy.

This kernel uses the same status buffer for checked direct evaluation:

```cpp
template<class Reader, class Law>
struct BoundaryKernel {
    Model model; Reader reader; Law law;
    typename ph::Status<Exec>::flags_type flags;
    Kokkos::View<State*, Space> exterior;
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c, const Condition& condition) const
    {
        bool failed = false; // Local to this executing thread.
        const ph::DeviceErrors errors{flags, &failed};
        const auto result = ph::evaluate(condition, c, reader(c, condition), model,
                                         law, ph::Corrections<>{}, errors);
        if (!failed) exterior(c.boundary_index) = result.exterior;
    }
};
```

Launch on the manager's execution instance:

```cpp
const auto reader = ph::OwnerReader{primitive};
status.begin_async();
boundary.for_each_face_async(BoundaryKernel<decltype(reader), decltype(heat_law)>{
    model, reader, heat_law, status.device_flags(exec), exterior_state}, Real{0.5});
status.check();
```

Raw callbacks must validate their buffer extents, time/helper inputs and any condition-specific requirements. Callbacks compile for every variant alternative, not only assigned types. For a custom model with `at(context, typed_condition)`, the manager selects that local model before evaluation; it must preserve the global compressible/thermal capabilities. Direct custom kernels perform this selection themselves when needed.

## A custom coupled condition

Use providers for prescribed data. Implement a custom condition when the closure depends on the interior state or changes its mathematics. It returns `Evaluation<State>` from `(context, inside, model, law, correction[, errors])`.

This example models a no-slip Newton-cooling wall with conductivity `k`, transfer coefficient `H`, and ambient temperature `T_inf`. With `alpha = H*h/k`, its trace is `(T_P + delta + alpha*T_inf)/(1 + alpha)`:

```cpp
struct ConvectiveWall {
    Real conductivity = 2, transfer = 10, ambient = 290;
    KOKKOS_INLINE_FUNCTION bool valid_parameters() const
    {
        return conductivity > Real{0} && transfer >= Real{0} && ambient > Real{0} &&
            Kokkos::isfinite(conductivity) && Kokkos::isfinite(transfer) && Kokkos::isfinite(ambient);
    }
    template<class M> void validate(const M&) const
    { krona::mpi::require(M::thermal && valid_parameters(), "ConvectiveWall: invalid model or parameters"); }
    template<class M, class Law, class Correction, class Errors>
    KOKKOS_INLINE_FUNCTION auto operator()(const bc::FaceContext<Real>& c,
        ph::Primitive<Real, M::thermal> inside, const M& model,
        const Law& law, const Correction& correction, const Errors& errors) const
    {
        if constexpr (!M::thermal) {
            errors.fail(ph::Error::Unsupported, "ConvectiveWall needs temperature");
            return ph::Evaluation{inside, inside};
        } else {
            if (!valid_parameters()) {
                errors.fail(ph::Error::Relation, "ConvectiveWall: invalid parameters");
                return ph::Evaluation{inside, inside};
            }
            const Real alpha = transfer*c.distance/conductivity;
            const Real delta = static_cast<Real>(correction.temperature(c));
            if (!Kokkos::isfinite(alpha) || !Kokkos::isfinite(delta)) {
                errors.fail(ph::Error::Relation, "ConvectiveWall: invalid relation inputs");
                return ph::Evaluation{inside, inside};
            }
            const Real wall_temperature = (inside.temperature + delta + alpha*ambient)/(Real{1} + alpha);
            return ph::evaluate(ph::IsothermalWall<Real>{wall_temperature, {}},
                                c, inside, model, law, correction, errors);
        }
    }
};
```

Register and apply it like any additional condition:

```cpp
ph::Manager<Model, Exec, Real, ConvectiveWall> coupled(layout, model);
for (const auto& patch : layout.catalog().patches()) coupled.set(patch.range.patch_id, ph::Symmetry{});
if (layout.catalog().needs_unlabelled()) coupled.set_unlabelled(ph::Symmetry{});
coupled.set(0, ConvectiveWall{}); // Requires labelled patch 0.
coupled.finalize();
status.begin_async();
coupled.apply_async(primitive, status, heat_law);
status.check();
```

Custom conditions may keep the old five-argument callable interface; a sixth error-handler argument enables deferred failure reporting. Use `errors.fail(ph::Error::..., "message")` and return an evaluation of the correct shape; failed results are not delivered to the sink. Pass the handler to nested `physical::evaluate()` calls. Conditions needing an explicit flux law specialize `ph::requires_flux_law<YourCondition>` and must still compile for unassigned alternatives with `NoFluxLaw`.

## Configuration

`read_config(path, comm)` reads TOML on rank zero, broadcasts the text and parses it on every rank. `parse_config(text)` parses locally. Every labelled patch must appear exactly once with matching ID and exact name. Unlabelled faces use `[unlabelled]`. With no labels, write `boundary_condition = []` explicitly.

```toml
[[boundary_condition]]
patch_id = 0
name = "wall"
type = "ISOTHERMAL_WALL"
temperature = 350.0
velocity = [0.0, 0.0, 0.0]

[unlabelled]
type = "SYMMETRY"
```

For numeric conditions, construct `Manager(layout, model, config)` or call `configure(config)`, then `finalize()`. Supported type names match the condition table: `SYMMETRY`, `SLIP_WALL`, `ADIABATIC_WALL`, `ISOTHERMAL_WALL`, `HEAT_FLUX_WALL`, `VELOCITY_INLET`, `TOTAL_CONDITIONS_INLET`, `PRESSURE_OUTLET`, `SUPERSONIC_INLET`, `SUPERSONIC_OUTLET`.

Canonical parameters are `pressure`, `temperature`, `velocity`, `flux`, `total_pressure`, `total_temperature`, `mach`, `direction`, `allow_choking`. Supersonic TOML accepts velocity **or** positive Mach with direction. A pressure outlet's nested `backflow` table uses type `REJECT`, `TOTAL_CONDITIONS_INLET`, or `DIRECTION_TEMPERATURE` and the corresponding parameters above.

Legacy aliases remain accepted: `WALL_ADIABATIC`, `WALL_ISOTHERMAL`, `OUTLET_BACK_PRESSURE`, `INLET_TOTAL_TP`, `INLET_SUPERSONIC`, `OUTLET_SUPERSONIC`; parameter aliases `T_wall`, `p_back`, `p`, `T`, `p0`, `T0`, `M` retain their original condition-specific meanings. Do not supply a parameter and its alias together.

Unknown condition keys/types are rejected. Import metadata (`family`, `source_type`, `global_face_count`) and other scalar tables are accepted, but `scalar.temperature` conflicts with the physical temperature channel and is rejected. `FARFIELD` and `USER_DEFINED` are not implemented parser types.

### Tabulated parameters

Each numeric scalar parameter and each component of `velocity` or `direction` accepts a number or a CSV path. The same rule applies to nested backflow parameters. `allow_choking` remains a boolean. For a Mach-based inlet, every sampled Mach number must be positive.

```toml
[[boundary_condition]]
patch_id = 0
name = "wall"
type = "ISOTHERMAL_WALL"
temperature = "profiles/wall_temperature.csv"
velocity = [0.0, "profiles/wall_uy.csv", 0.0]
```

Each CSV supplies one scalar. Its header contains `value` and one to three axes selected from `x`, `y`, `z`. The rows must cover the full Cartesian grid on the active axes, with at least two coordinates on each. Relative paths use rank zero's working directory. Each rank interpolates the profiles at its own face centroids. A centroid outside the table bounds on an active axis causes a configuration error.

Include `<krona/boundary/physical/tabulated_config.hpp>` and use the same configuration and communicator on all ranks, including ranks with empty local patches:

```cpp
const auto config = ph::read_config("bc.toml", comm);
ph::TabulatedManager<Model, Exec, Real> boundary(layout, model);
ph::configure_tabulated(boundary, config, host_mesh, comm);
boundary.finalize();
```

Pass the host mesh used to build the layout. The manager owns the sampled buffers. Complete prior boundary kernels before rebinding conditions. `configure_tabulated()` also accepts numeric conditions, and `set()` supports custom C++ providers.

### Compressible flow: pressure outlet with reservoir backflow

```toml
[[boundary_condition]]
patch_id = 0
name = "outlet"
type = "PRESSURE_OUTLET"
pressure = 100000.0           # Static outlet pressure [Pa].

[boundary_condition.backflow]
type = "TOTAL_CONDITIONS_INLET"
total_pressure = 120000.0    # Backflow reservoir total pressure [Pa].
total_temperature = 330.0   # Backflow reservoir total temperature [K].
direction = [-1.0, 0.0, 0.0]
allow_choking = false
```

For outward flow, the outlet uses `pressure` according to its local flow regime. Reverse flow uses the reservoir's total conditions and direction. The compressible model must provide a compatible characteristic policy.

### Incompressible thermal flow: pressure outlet with directed backflow

```toml
[[boundary_condition]]
patch_id = 0
name = "outlet"
type = "PRESSURE_OUTLET"
pressure = 0.0              # Outlet pressure in the solver's pressure reference [Pa].

[boundary_condition.backflow]
type = "DIRECTION_TEMPERATURE"
direction = [-1.0, 0.0, 0.0]
temperature = 300.0         # Incoming fluid temperature [K].
```

Reverse flow uses the specified direction and temperature while preserving the interior normal velocity. Omit `temperature` for a nonthermal incompressible model.

Both examples assume the outlet's outward normal points along positive `x`; the backflow direction points into the domain. Replace the patch ID/name with those from the mesh and adjust the direction for the actual geometry. Each example describes one patch; add entries for the remaining labelled patches.

### Reject reverse flow

To reject reverse flow instead, replace the nested table with:

```toml
[boundary_condition.backflow]
type = "REJECT"
```

`REJECT` is also the default when the `backflow` table is omitted.

## Asynchrony, lifetime and errors

| Operation | Completion |
| --- | --- |
| `BoundaryLayout(...)` | Local blocking geometry preparation/validation and fence. |
| `Manager(...)`, `set*()`, `configure()`, `finalize()` | Host work only; no device submission/fence or MPI communication. Validation hooks must follow that contract. |
| `parse_config()` | Local, blocking TOML parsing. |
| `read_config()` | Blocking collective TOML load, followed by local parsing. |
| `configure_tabulated()` | Collective CSV loads, local interpolation and completed host-to-device copies. |
| Profile factories and field-binding helpers | Construct parameter/descriptor objects; no field-value copies or work submission. |
| `status.begin_async()` | Queues flag reset; the previous batch must have been checked. |
| `apply_async()`, `evaluate_async()` | Host checks and kernel submission; no final fence. Require a begun status on the layout's execution instance. |
| `for_each_face_async()` | Submits callback kernels without a final fence. |
| `for_each_patch()` | Completes host dispatch, not visitor-launched kernels. |
| `status.check()` | Local blocking flag copy/fence and fatal diagnostics; no MPI collective. |

Standard checks record invalid sampled parameters, interior/returned states, BC compatibility and wall relations; affected faces skip their sink/write. No rollback occurs. Host contract failures use `mpi::require`; unexpected host exceptions are caught by `mpi::guarded`. Check before consuming a batch's results. A fence alone does not inspect its flags.

EOS/material/custom-policy routines retain their own domains and error handling; an internal abort cannot be intercepted by `Status`. Custom characteristic policies may accept an extra error-handler argument and propagate it; old signatures remain supported for valid inputs. Direct `physical::evaluate()` without an explicit handler uses `Kokkos::abort` on BC errors.

Queue dependent producers, boundary work and consumers on one execution instance, or synchronize explicitly across instances/host/MPI. `_async` provides no portable completion guarantee even if a CPU backend executes immediately. Reuse one status across several managers/submissions on that instance, then check once.

Keep layout/mesh owners and every captured field, provider, gradient, coefficient and callback buffer alive until completion. Condition objects are copied into kernels; reassignment affects later submissions but does not synchronize their backing buffers. Layout/manager destruction does not fence. Status destruction drains a pending batch to protect its flags, but neither checks errors nor protects buffers the caller has already released. Complete and check work before freeing storage or finalizing Kokkos.